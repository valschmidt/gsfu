#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a Kongsberg .kmall file to a Generic Sensor Format (.gsf) file.

Reads the source .kmall file with the sibling `kmall` package
(https://github.com/valschmidt/kmall) -- imported lazily below, and *not*
a hard dependency of GSFU.gsfu -- and writes PROCESSING_PARAMETERS,
SOUND_VELOCITY_PROFILE, SWATH_BATHYMETRY_PING (with the KMALL_SPECIFIC
sensor-specific subrecord and its TX sector array), and ATTITUDE records
via gsfu.py's write_* methods.

KMALL files can carry position (#SPO/#CPO) and attitude (#SKM) from more
than one configured sensor system (K-Controller's "Position 1/2/3..." and
"Attitude 1/2/3..."); -a/--attitude-source selects which #SKM stream (1
by default) supplies the ping's interpolated pitch/roll/heave and the
GSF_RECORD_ATTITUDE records written to the file. Ping position and
heading always come directly from the #MRZ datagram's own pingInfo (the
position/heading SIS itself used for that ping), matching how a native
GSF writer would behave -- see convert.md for the reasoning.

Known simplifications, all worth revisiting if they matter to you:
  * TideCorrector_m, DepthCorrector_m, Course_deg, and Speed_kn are always
    written as their GSF_NULL_* "not available" sentinel (not 0.0 -- see
    convert.md's "Marking a field as not available" section): MRZ carries
    no tide correction (this is a raw, uncorrected conversion) and no
    course/speed-over-ground field (only #SPO/#CPO do, and this converter
    doesn't read position datagrams -- ping position comes from MRZ itself).
  * CenterBeam is approximated as NumberBeams // 2: MRZ has no explicit
    "center beam index" field, and gsf.h defines no null sentinel for it.
  * The per-beam backscatter time series (intensity subrecord, id 21) is
    not written -- gsfu.py has no encoder for it yet (see README).
  * BEAM_FLAGS_ARRAY is not written -- MRZ's detectionType/detectionMethod
    don't map cleanly onto GSF's beam flag bit convention without more
    design work than this first cut covers.

Indexing strategy: this module builds its own lightweight, seek-based
index of the source .kmall file (mirroring gsfu.py's own index_file() --
read each datagram's small framing header, seek past its payload) rather
than using kmall.py's index_file() (which reads every datagram's full
payload just to record its type) or its decode_datagram()/skip_datagram()
sequential-scan API (which still touches every record's header twice, in
two passes). Indexing once and then seeking directly to only the needed
offsets touches strictly fewer bytes than either. kmall.py itself is not
modified.
"""
import argparse
import struct
import sys

import numpy as np
import pandas as pd

from GSFU.gsfu import (
    GSF_NULL_COURSE,
    GSF_NULL_DEPTH_CORRECTOR,
    GSF_NULL_SEP,
    GSF_NULL_SPEED,
    GSF_NULL_TIDE_CORRECTOR,
    gsf as GsfWriter,
)


def _kmall_class():
    """
    Import and return the `kmall` class from the `KMALL` package
    (https://github.com/valschmidt/kmall). The import happens inside this
    function, rather than at the top of this module, so that importing
    GSFU.kmall2gsf, or any other part of GSFU, never requires the `kmall`
    package to be installed; only code that actually converts a file
    needs it.

    :return: the `kmall` class, ready to be instantiated with a file path.

    :raises ImportError: the `kmall` package is not installed.
    """
    try:
        from KMALL.kmall import kmall as KmallReader
    except ImportError as exc:
        raise ImportError(
            "kmall2gsf.py requires the 'kmall' package "
            "(https://github.com/valschmidt/kmall, importable as `KMALL`) "
            "to read .kmall files. Install it (e.g. `pip install -e /path/to/kmall`) "
            "and try again. See convert.md for details.") from exc
    return KmallReader


###########################################################
# Lightweight, seek-based .kmall indexer
###########################################################

def index_kmall_file(K):
    """
    Walk an open .kmall file and record the offset, type, and size of
    every datagram it contains, without reading any datagram's payload.
    This reads only each datagram's eight-byte framing header (a
    four-byte total length followed by a four-byte type code, for example
    b'#MRZ') and then seeks past its payload, however large that payload
    is (for example, a #MWC water-column datagram). Note that KMALL
    stores its on-disk integers in little-endian byte order, the opposite
    of GSF's big-endian convention.

    :param K: a kmall.kmall instance already open for reading, meaning
        K.OpenFiletoRead() has already been called on it.

    :return: a list of (offset, datagram_type, num_bytes) tuples, in file
        order. num_bytes is the datagram's total length, including its
        own leading four-byte length field, so offset plus num_bytes is
        the offset of the next datagram.
    """
    K.FID.seek(0, 2)
    file_size = K.FID.tell()
    K.FID.seek(0, 0)

    # kmall.py's decode_datagram() computes (and, worse, seeks back to
    # file offset 0 while doing so) K.file_size itself the first time it's
    # called if this isn't already set -- which would silently discard
    # whatever offset the caller had just sought to. Set it now so every
    # later K.FID.seek(offset); K.decode_datagram() in this module is safe.
    K.file_size = file_size

    index = []
    while K.FID.tell() < file_size:
        offset = K.FID.tell()
        header = K.FID.read(8)
        if len(header) < 8:
            break
        num_bytes, dgm_type_bytes = struct.unpack('<I4s', header)
        dgm_type = dgm_type_bytes.decode('ascii', 'replace')
        index.append((offset, dgm_type, num_bytes))
        K.FID.seek(offset + num_bytes, 0)

    return index


###########################################################
# Attitude interpolation
###########################################################

def _circular_lerp(a, b, frac):
    """
    Linearly interpolate an angle, in degrees, from a to b, where both
    angles are on a 0 to 360 degree circle, taking the shorter way around
    the 0/360 degree boundary rather than always increasing.

    :param a: the starting angle, in degrees.
    :param b: the ending angle, in degrees.
    :param frac: how far to interpolate between a and b, from 0.0
        (returns a) to 1.0 (returns b).

    :return: the interpolated angle, in degrees, normalized to the range
        0 to 360.
    """
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return (a + diff * frac) % 360.0


def interpolate_attitude(attitude_samples, t):
    """
    Linearly interpolate pitch, roll, heave, and heading at time t from a
    sorted list of attitude samples. When t falls outside the range of
    buffered samples, this returns the first or last sample unchanged,
    rather than extrapolating beyond it. Heading is interpolated the
    short way around the 0/360 degree wraparound, using _circular_lerp().

    :param attitude_samples: a list of (time, pitch_deg, roll_deg,
        heave_m, heading_deg) tuples, sorted by time.
    :param t: the time to interpolate at, in the same units as the times
        in attitude_samples.

    :return: a tuple of (pitch_deg, roll_deg, heave_m, heading_deg), the
        interpolated attitude at time t.

    :raises ValueError: attitude_samples is empty.
    """
    if not attitude_samples:
        raise ValueError("no attitude samples available to interpolate")

    if t <= attitude_samples[0][0]:
        _, pitch, roll, heave, heading = attitude_samples[0]
        return pitch, roll, heave, heading
    if t >= attitude_samples[-1][0]:
        _, pitch, roll, heave, heading = attitude_samples[-1]
        return pitch, roll, heave, heading

    times = [s[0] for s in attitude_samples]
    i = np.searchsorted(times, t)
    t0, pitch0, roll0, heave0, heading0 = attitude_samples[i - 1]
    t1, pitch1, roll1, heave1, heading1 = attitude_samples[i]
    frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0

    return (
        pitch0 + (pitch1 - pitch0) * frac,
        roll0 + (roll1 - roll0) * frac,
        heave0 + (heave1 - heave0) * frac,
        _circular_lerp(heading0, heading1, frac),
    )


###########################################################
# KMALL -> GSF field mapping
###########################################################

def mrz_to_kmall_specific(mrz):
    """
    Build the KMALL_SPECIFIC scalar field dictionary and the list of
    per-transmit-sector dictionaries that a GSF ping's sensor-specific
    subrecord needs, directly from one MRZ datagram (as returned by
    kmall.kmall.read_EMdgmMRZ()). The field names this returns match the
    key names GSFU.gsfu._decode_kmall_specific() decodes from a GSF file,
    just read here from kmall.py's own field names instead of from GSF
    wire bytes.

    :param mrz: one decoded #MRZ datagram, as returned by
        kmall.kmall.read_EMdgmMRZ().

    :return: a tuple of (kmall_specific, tx_sectors). kmall_specific is a
        dictionary of scalar KMALL_SPECIFIC fields. tx_sectors is a list
        of dictionaries, one per transmit sector.
    """
    header = mrz['header']
    cmn = mrz['cmnPart']
    info = mrz['pingInfo']

    kmall_specific = {
        'DgmType': 1,  # #MRZ
        'DgmVersion': header['dgmVersion'],
        'SystemID': header['systemID'],
        'EchoSounderID': header['echoSounderID'],
        'NumBytesCmnPart': cmn['numBytesCmnPart'],
        'PingCnt': cmn['pingCnt'],
        'RxFansPerPing': cmn['rxFansPerPing'],
        'RxFanIndex': cmn['rxFanIndex'],
        'SwathsPerPing': cmn['swathsPerPing'],
        'SwathAlongPosition': cmn['swathAlongPosition'],
        'TxTransducerInd': cmn['txTransducerInd'],
        'RxTransducerInd': cmn['rxTransducerInd'],
        'NumRxTransducers': cmn['numRxTransducers'],
        'AlgorithmType': cmn['algorithmType'],
        'NumBytesInfoData': info['numBytesInfoData'],
        'PingRate_Hz': info['pingRate_Hz'],
        'BeamSpacing': info['beamSpacing'],
        'DepthMode': info['depthMode'],
        'SubDepthMode': info['subDepthMode'],
        'DistanceBtwSwath': info['distanceBtwSwath'],
        'DetectionMode': info['detectionMode'],
        'PulseForm': info['pulseForm'],
        'FrequencyMode_Hz': info['frequencyMode_Hz'],
        'FreqRangeLowLim_Hz': info['freqRangeLowLim_Hz'],
        'FreqRangeHighLim_Hz': info['freqRangeHighLim_Hz'],
        'MaxTotalTxPulseLength_sec': info['maxTotalTxPulseLength_sec'],
        'MaxEffTxPulseLength_sec': info['maxEffTxPulseLength_sec'],
        'MaxEffTxBandWidth_Hz': info['maxEffTxBandWidth_Hz'],
        'AbsCoeff_dBPerkm': info['absCoeff_dBPerkm'],
        'PortSectorEdge_deg': info['portSectorEdge_deg'],
        'StarbSectorEdge_deg': info['starbSectorEdge_deg'],
        'PortMeanCov_deg': info['portMeanCov_deg'],
        'StarbMeanCov_deg': info['stbdMeanCov_deg'],
        'PortMeanCov_m': info['portMeanCov_m'],
        'StarbMeanCov_m': info['starbMeanCov_m'],
        'ModeAndStabilisation': info['modeAndStabilisation'],
        'RuntimeFilter1': info['runtimeFilter1'],
        'RuntimeFilter2': info['runtimeFilter2'],
        'PipeTrackingStatus': info['pipeTrackingStatus'],
        'TransmitArraySizeUsed_deg': info['transmitArraySizeUsed_deg'],
        'ReceiveArraySizeUsed_deg': info['receiveArraySizeUsed_deg'],
        'TransmitPower_dB': info['transmitPower_dB'],
        'SLrampUpTimeRemaining': info['SLrampUpTimeRemaining'],
        'YawAngle_deg': info['yawAngle_deg'],
        'NumBytesPerTxSector': info['numBytesPerTxSector'],
        'HeadingVessel_deg': info['headingVessel_deg'],
        'SoundSpeedAtTxDepth_mPerSec': info['soundSpeedAtTxDepth_mPerSec'],
        'TxTransducerDepth_m': info['txTransducerDepth_m'],
        'ZWaterLevelReRefPoint_m': info['z_waterLevelReRefPoint_m'],
        'XKmallToAll_m': info['x_kmallToall_m'],
        'YKmallToAll_m': info['y_kmallToall_m'],
        'LatLongInfo': info['latLongInfo'],
        'PosSensorStatus': info['posSensorStatus'],
        'AttitudeSensorStatus': info['attitudeSensorStatus'],
        'Latitude_deg': info['latitude_deg'],
        'Longitude_deg': info['longitude_deg'],
        'EllipsoidHeightReRefPoint_m': info['ellipsoidHeightReRefPoint_m'],
        'NumBytesRxInfo': mrz['rxInfo']['numBytesRxInfo'],
        'NumSoundingsMaxMain': mrz['rxInfo']['numSoundingsMaxMain'],
        'NumSoundingsValidMain': mrz['rxInfo']['numSoundingsValidMain'],
        'NumBytesPerSounding': mrz['rxInfo']['numBytesPerSounding'],
        'WCSampleRate': mrz['rxInfo']['WCSampleRate'],
        'SeabedImageSampleRate': mrz['rxInfo']['seabedImageSampleRate'],
        'BSnormal_dB': mrz['rxInfo']['BSnormal_dB'],
        'BSoblique_dB': mrz['rxInfo']['BSoblique_dB'],
        'ExtraDetectionAlarmFlag': mrz['rxInfo']['extraDetectionAlarmFlag'],
        'NumExtraDetections': mrz['rxInfo']['numExtraDetections'],
        'NumBytesPerClass': mrz['rxInfo']['numBytesPerClass'],
    }

    tx_sectors = [
        {
            'TxSectorNumb': s['txSectorNumb'],
            'TxArrNumber': s['txArrNumber'],
            'TxSubArray': s['txSubArray'],
            'SectorTransmitDelay_sec': s['sectorTransmitDelay_sec'],
            'TiltAngleReTx_deg': s['tiltAngleReTx_deg'],
            'TxNominalSourceLevel_dB': s['txNominalSourceLevel_dB'],
            'TxFocusRange_m': s['txFocusRange_m'],
            'CentreFreq_Hz': s['centreFreq_Hz'],
            'SignalBandWidth_Hz': s['signalBandWidth_Hz'],
            'TotalSignalLength_sec': s['totalSignalLength_sec'],
            'PulseShading': s['pulseShading'],
            'SignalWaveForm': s['signalWaveForm'],
        }
        for s in _listofdicts(mrz['txSectorInfo'])
    ]

    return kmall_specific, tx_sectors


def _listofdicts(dict_of_lists):
    """
    Convert a dictionary of equal-length lists into a list of per-item
    dictionaries. The `kmall` package stores repeated substructures, such
    as transmit sectors or soundings, as a dictionary whose values are
    lists (one list per field, all the same length); this converts that
    representation into the more convenient one dictionary per item, used
    by the transmit-sector loop in mrz_to_kmall_specific().

    :param dict_of_lists: a dictionary mapping field name to a list of
        that field's values, one entry per item, all lists the same
        length. An empty or falsy value is treated as zero items.

    :return: a list of dictionaries, one per item, each mapping the same
        field names to that item's single value.
    """
    if not dict_of_lists:
        return []
    keys = list(dict_of_lists.keys())
    n = len(dict_of_lists[keys[0]])
    return [{k: dict_of_lists[k][i] for k in keys} for i in range(n)]


def mrz_to_beams(mrz):
    """
    Build the dictionary of per-beam arrays a GSF ping's 'Beams' entry
    needs, from one MRZ datagram's soundings (which kmall.py stores as a
    dictionary of equal-length lists, one entry per beam). Only the
    subset of GSF beam array subrecords with a reasonably direct MRZ
    equivalent is populated; see this module's own docstring, under
    "Known simplifications", for exactly which ones are deliberately left
    out and why.

    :param mrz: one decoded #MRZ datagram, as returned by
        kmall.kmall.read_EMdgmMRZ().

    :return: a dictionary mapping each populated GSF beam array's column
        label (for example 'Depth_m') to a numpy array of that array's
        per-beam values.
    """
    soundings = mrz['sounding']
    twtt = np.asarray(soundings['twoWayTravelTime_sec'], dtype=np.float64) \
        + np.asarray(soundings['twoWayTravelTimeCorrection_sec'], dtype=np.float64)

    return {
        'Depth_m': np.asarray(soundings['z_reRefPoint_m'], dtype=np.float64),
        'AcrossTrack_m': np.asarray(soundings['y_reRefPoint_m'], dtype=np.float64),
        'AlongTrack_m': np.asarray(soundings['x_reRefPoint_m'], dtype=np.float64),
        'TravelTime_s': twtt,
        'BeamAngle_deg': np.asarray(soundings['beamAngleReRx_deg'], dtype=np.float64),
        'MeanCalAmplitude_dB': np.asarray(soundings['reflectivity2_dB'], dtype=np.float64),
        'QualityFactor': np.asarray(soundings['qualityFactor'], dtype=np.float64),
        'SectorNumber': np.asarray(soundings['txSectorNumb'], dtype=np.float64),
        'DetectionInfo': np.asarray(soundings['detectionType'], dtype=np.float64),
        'VerticalError_m': np.asarray(soundings['detectionUncertaintyVer_m'], dtype=np.float64),
        'HorizontalError_m': np.asarray(soundings['detectionUncertaintyHor_m'], dtype=np.float64),
    }


def mrz_to_ping_scalars(mrz, pitch_deg, roll_deg, heave_m):
    """
    Build the dictionary of fixed scalar ping fields that
    gsf.write_swath_bathymetry_ping() expects, from one MRZ datagram.
    Position and heading are read directly from the MRZ datagram's own
    pingInfo structure, the same values the sonar's own positioning
    system used for this ping. Pitch, roll, and heave are not read from
    MRZ directly; the caller supplies its own already-interpolated
    attitude values instead, typically produced by
    interpolate_attitude(). Fields that MRZ does not carry, such as tide
    and draft correction or course and speed over ground, are written as
    their GSF_NULL_* "not available" sentinel value rather than as zero,
    since zero would otherwise be indistinguishable from a genuine,
    corrected value of zero; see this project's convert.md, under
    "Marking a field as not available", for why this distinction matters.

    :param mrz: one decoded #MRZ datagram, as returned by
        kmall.kmall.read_EMdgmMRZ().
    :param pitch_deg: this ping's interpolated pitch, in degrees.
    :param roll_deg: this ping's interpolated roll, in degrees.
    :param heave_m: this ping's interpolated heave, in meters.

    :return: a dictionary of the fixed scalar ping fields, in the same
        shape gsf.write_swath_bathymetry_ping() expects as part of its
        `record` argument.
    """
    header = mrz['header']
    info = mrz['pingInfo']
    number_beams = mrz['rxInfo']['numSoundingsMaxMain'] + mrz['rxInfo']['numExtraDetections']

    return {
        'PingTime': header['dgtime'],
        'Longitude_deg': info['longitude_deg'],
        'Latitude_deg': info['latitude_deg'],
        'NumberBeams': number_beams,
        'CenterBeam': number_beams // 2,  # MRZ has no explicit center-beam index
        'PingFlags': 0,
        # Not present in MRZ (raw conversion, no tide/draft model applied)
        # -- written as the GSF_NULL_* "not available" sentinel, not 0.0,
        # since 0.0 is itself a valid corrector value and would otherwise
        # be indistinguishable from "corrected with zero offset". See
        # convert.md's "Marking a field as not available" section.
        'TideCorrector_m': GSF_NULL_TIDE_CORRECTOR,
        'DepthCorrector_m': GSF_NULL_DEPTH_CORRECTOR,
        'Heading_deg': info['headingVessel_deg'],
        'Pitch_deg': pitch_deg,
        'Roll_deg': roll_deg,
        'Heave_m': heave_m,
        # Course/speed-over-ground are only in #SPO/#CPO, which this
        # converter doesn't read -- likewise null, not 0.0.
        'Course_deg': GSF_NULL_COURSE,
        'Speed_kn': GSF_NULL_SPEED,
        'Height_m': info['ellipsoidHeightReRefPoint_m'],
        'SEP_m': GSF_NULL_SEP,
        # gsf.h defines no null sentinel for GPSTideCorrector_m; 0.0 is
        # the best available default (also a real, achievable value).
        'GPSTideCorrector_m': 0.0,
    }


def _lenient_parse_kv_text(text):
    """
    Parse a "key=value" text blob into a dictionary, silently skipping
    any entry that does not contain an "=" character. This is the text
    format KMALL install and runtime parameter datagrams carry, with
    entries separated by some mix of commas, semicolons, and newlines
    depending on the datagram type.

    The `kmall` package has its own parsers for this same text, in its
    translate_installation_parameters_todict() and
    translate_runtime_parameters_todict() functions, but those assume
    every entry is well formed and raise an exception on the first one
    that is not; this has been observed to happen with real KMALL sample
    files. This function is used as a fallback when those functions raise
    (see convert()), so that one malformed entry does not abort an entire
    conversion.

    :param text: the raw key/value parameter text to parse.

    :return: a dictionary of the successfully parsed key/value pairs.
        Malformed entries are silently omitted, not raised as an error.
    """
    params = {}
    for chunk in text.replace('\n', ',').replace(';', ',').split(','):
        entry = chunk.strip()
        if '=' in entry:
            k, v = entry.split('=', 1)
            params[k.strip()] = v.strip()
    return params


###########################################################
# Conversion driver
###########################################################

def convert(kmall_path, gsf_path, attitude_source=1, verbose=False):
    """
    Convert one .kmall file to a .gsf file, writing PROCESSING_PARAMETERS,
    SOUND_VELOCITY_PROFILE, SWATH_BATHYMETRY_PING, and ATTITUDE records.
    See this module's own docstring for the full list of GSF record types
    written and the known simplifications this conversion makes.

    :param kmall_path: the path to the source .kmall file to read.
    :param gsf_path: the path to the .gsf file to write.
    :param attitude_source: which #SKM sensorSystem stream supplies this
        file's interpolated ping attitude and its GSF_RECORD_ATTITUDE
        records. This is one-indexed, matching how K-Controller numbers
        its ATTI_1/ATTI_2/... installation parameters.
    :param verbose: if True, print progress messages while converting.

    :raises ImportError: the `kmall` package is not installed.
    :raises ValueError: no #SKM datagram in the source file matches
        attitude_source.
    """
    KmallReader = _kmall_class()
    K = KmallReader(kmall_path)
    K.OpenFiletoRead()

    index = index_kmall_file(K)
    if verbose:
        print("Indexed %d datagrams in %s" % (len(index), kmall_path))

    offsets_by_type = {}
    for offset, dgm_type, _num_bytes in index:
        offsets_by_type.setdefault(dgm_type, []).append(offset)

    wanted_sensor_system = attitude_source - 1

    # Pass 1: IIP/IOP (first occurrence of each), every #SVP, and every
    # #SKM sample for the selected attitude source, buffered so it's
    # available before any ping needs to interpolate into it.
    iip_params = {}
    iop_params = {}
    svp_datagrams = []
    attitude_samples = []  # (dgtime, pitch_deg, roll_deg, heave_m, heading_deg)
    found_attitude_systems = set()

    # NOTE: we deliberately call read_EMdgmIIP()/read_EMdgmIOP() directly
    # with translate=False here, rather than K.read_datagram() (which
    # always requests translate=True). kmall.py's own translate_*_todict()
    # parsers raise ValueError on a malformed "no '=' " entry -- a bug
    # observed in real sample files -- and we're not allowed to fix
    # kmall.py itself. Reading raw text and parsing it ourselves with
    # _lenient_parse_kv_text() (which skips malformed entries instead of
    # raising) sidesteps that bug entirely.
    for offset in offsets_by_type.get('#IIP', [])[:1]:
        K.FID.seek(offset)
        K.decode_datagram()
        dg = K.read_EMdgmIIP(translate=False)
        iip_params = _lenient_parse_kv_text(dg['install_txt'])

    for offset in offsets_by_type.get('#IOP', [])[:1]:
        K.FID.seek(offset)
        K.decode_datagram()
        dg = K.read_EMdgmIOP(translate=False)
        iop_params = _lenient_parse_kv_text(dg['runtime_txt'])

    for offset in offsets_by_type.get('#SVP', []):
        K.FID.seek(offset)
        K.decode_datagram()
        K.read_datagram()
        svp_datagrams.append(K.datagram_data)

    for offset in offsets_by_type.get('#SKM', []):
        K.FID.seek(offset)
        K.decode_datagram()
        K.read_datagram()
        dg = K.datagram_data
        found_attitude_systems.add(dg['infoPart']['sensorSystem'])
        if dg['infoPart']['sensorSystem'] != wanted_sensor_system:
            continue
        s = dg['sample']['KMdefault']
        for i in range(len(s['dgtime'])):
            attitude_samples.append(
                (s['dgtime'][i], s['pitch_deg'][i], s['roll_deg'][i], s['heave_m'][i], s['heading_deg'][i]))

    if offsets_by_type.get('#SKM') and not attitude_samples:
        raise ValueError(
            "no #SKM attitude data found for attitude_source=%d (sensorSystem=%d); "
            "sensorSystem values present in this file: %s" %
            (attitude_source, wanted_sensor_system, sorted(found_attitude_systems)))

    attitude_samples.sort(key=lambda s: s[0])

    G = GsfWriter(gsf_path)
    G.write_header()

    if iip_params or iop_params:
        param_time = attitude_samples[0][0] if attitude_samples else 0.0
        G.write_processing_parameters({**iip_params, **iop_params}, param_time=param_time)

    for svp in svp_datagrams:
        obs_time = svp['datetime'] if svp['time_sec'] else svp['header']['dgtime']
        G.write_sound_velocity_profile(
            observation_time=obs_time, application_time=svp['header']['dgtime'],
            latitude_deg=svp['latitude_deg'], longitude_deg=svp['longitude_deg'],
            depth_m=svp['sensorData']['depth_m'],
            sound_speed_mPerSec=svp['sensorData']['soundVelocity_mPerSec'])

    for offset in offsets_by_type.get('#SKM', []):
        K.FID.seek(offset)
        K.decode_datagram()
        K.read_datagram()
        dg = K.datagram_data
        if dg['infoPart']['sensorSystem'] != wanted_sensor_system:
            continue
        s = dg['sample']['KMdefault']
        G.write_attitude(
            attitude_time=s['dgtime'], pitch_deg=s['pitch_deg'], roll_deg=s['roll_deg'],
            heave_m=s['heave_m'], heading_deg=s['heading_deg'])

    # Pass 2: pings, interpolating attitude from the pass-1 buffer.
    ping_count = 0
    for offset in offsets_by_type.get('#MRZ', []):
        K.FID.seek(offset)
        K.decode_datagram()
        K.read_datagram()
        mrz = K.datagram_data

        pitch, roll, heave, _heading = interpolate_attitude(attitude_samples, mrz['header']['dgtime'])

        kmall_specific, tx_sectors = mrz_to_kmall_specific(mrz)
        record = mrz_to_ping_scalars(mrz, pitch, roll, heave)
        record['Beams'] = mrz_to_beams(mrz)
        record['SensorSpecificID'] = 156
        record['SensorSpecific'] = {**kmall_specific, 'TxSectors': pd.DataFrame(tx_sectors)}

        G.write_swath_bathymetry_ping(record)
        ping_count += 1
        if verbose and ping_count % 100 == 0:
            print("  wrote %d pings" % ping_count)

    G.closeFile()
    K.closeFile()

    if verbose:
        print("Wrote %d pings, %d attitude records, %d SVP records to %s" %
              (ping_count, len(offsets_by_type.get('#SKM', [])), len(svp_datagrams), gsf_path))

    return ping_count


###########################################################
# Command line interface
###########################################################

def main(args=None):
    """
    Command line entry point for the kmall2gsf.py script. Parses
    command-line arguments and calls convert() to perform the actual
    conversion, printing a message and returning a nonzero exit code
    instead of raising if the `kmall` package is not installed or if
    conversion fails.

    :param args: the command-line arguments to parse, not including the
        program name. If left as None, sys.argv[1:] is used.

    :return: the process exit code: 0 on success, 1 if conversion failed.
    """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Convert a Kongsberg .kmall file to a Generic Sensor Format (.gsf) file.")
    parser.add_argument('-f', action='store', dest='kmall_filename', required=True,
                         help="The .kmall file to convert.")
    parser.add_argument('-o', action='store', dest='gsf_filename', required=True,
                         help="The .gsf file to write.")
    parser.add_argument('-a', '--attitude-source', action='store', type=int, default=1,
                         dest='attitude_source', metavar='N',
                         help="Which #SKM attitude source to use (1-indexed, matching "
                              "K-Controller's ATTI_1/ATTI_2/... installation parameters). "
                              "Default: 1.")
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False,
                         help="Print progress while converting.")
    parsed = parser.parse_args(args)

    try:
        convert(parsed.kmall_filename, parsed.gsf_filename,
                attitude_source=parsed.attitude_source, verbose=parsed.verbose)
    except (ImportError, ValueError) as exc:
        print("Error: %s" % exc)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
