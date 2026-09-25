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
    Lazily import the `kmall` class from the `KMALL` package
    (https://github.com/valschmidt/kmall, importable name is `KMALL`,
    uppercase) so that importing GSFU.kmall2gsf -- or any other part of
    GSFU -- never requires it to be installed.
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
    Walk an open .kmall file and record (offset, datagram_type, num_bytes)
    for every datagram, reading only each datagram's 8-byte framing header
    (4-byte total length + 4-byte type, e.g. b'#MRZ') and seeking past its
    payload -- no datagram payload is read here, however large (a #MWC
    water-column datagram, say). KMALL's on-disk integers are little-
    endian, unlike GSF's big-endian ("network byte order") convention.

    :param K: a kmall.kmall instance already open for reading
        (K.OpenFiletoRead() already called).
    :return: list of (offset, datagram_type, num_bytes) in file order.
        num_bytes is the datagram's total length, including its own
        leading 4-byte length field (so offset + num_bytes is the next
        datagram's offset).
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
    """ Linearly interpolate an angle in degrees from `a` to `b` (0-360),
    taking the shorter way around the 0/360 boundary. """
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return (a + diff * frac) % 360.0


def interpolate_attitude(attitude_samples, t):
    """
    Linearly interpolate pitch/roll/heave/heading at time `t` from a
    sorted list of (time, pitch_deg, roll_deg, heave_m, heading_deg)
    samples. Clamps to the first/last sample when `t` is outside the
    buffered range (rather than extrapolating), and interpolates heading
    the short way around the 0/360 wraparound.

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
    Build the KMALL_SPECIFIC scalar dict and TX sector list gsfu.py's
    write_swath_bathymetry_ping() expects, directly from an MRZ datagram
    (as returned by kmall.kmall.read_EMdgmMRZ()) -- the field-for-field
    inverse of GSFU.gsfu._decode_kmall_specific()'s key names, just
    sourced from kmall.py's own field names instead of GSF wire bytes.

    :return: (kmall_specific: dict, tx_sectors: list[dict])
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
    """ kmall.py stores repeated substructures (sectors, soundings) as a
    dict of equal-length lists; convert back to a list of per-item dicts
    for the sector loop above. """
    if not dict_of_lists:
        return []
    keys = list(dict_of_lists.keys())
    n = len(dict_of_lists[keys[0]])
    return [{k: dict_of_lists[k][i] for k in keys} for i in range(n)]


def mrz_to_beams(mrz):
    """
    Build the beams dict write_swath_bathymetry_ping() expects from an
    MRZ datagram's soundings (a dict of equal-length lists, one entry per
    beam). Only the subset of GSF beam array subrecords with a reasonably
    direct MRZ equivalent is populated -- see the module docstring's
    "Known simplifications" for what's deliberately omitted.
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
    Build the ping scalars dict write_swath_bathymetry_ping() expects.
    Position and heading come directly from MRZ's own pingInfo (the
    values SIS itself used for this ping); pitch/roll/heave are the
    caller's already-interpolated attitude (see interpolate_attitude()).
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
    Parse a "key=value" text blob (KMALL install/runtime parameter text,
    entries separated by some mix of ','/';'/newlines depending on
    datagram type) into a dict, silently skipping any entry that doesn't
    contain '=' -- unlike kmall.py's own translate_installation_
    parameters_todict()/translate_runtime_parameters_todict(), which
    assume every entry is well-formed and raise on the first one that
    isn't. Used as a fallback when those raise (see convert()) so one
    malformed entry doesn't abort the whole conversion.
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
    Convert one .kmall file to a .gsf file.

    :param attitude_source: which #SKM sensorSystem stream (1-indexed, as
        numbered in K-Controller's ATTI_1/ATTI_2/... installation
        parameters) supplies interpolated ping attitude and the
        GSF_RECORD_ATTITUDE records written to the file.
    :raises ValueError: no #SKM datagram in the file matches
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
    """ Command line script entry point. """
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
