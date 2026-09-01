#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A python class (and command line utility) to index, read, and write
Generic Sensor Format (GSF) sonar data files.

The physical, on-disk record encoding implemented here (the record size
field, the packed data identifier word, and the optional checksum) and the
record type constants/descriptions below are taken from the reference C
implementation of the GSF library ("gsflib"), Copyright 2019 Leidos, Inc.,
distributed under the LGPL 2.1:

    https://github.com/Spatialnetics/gsflib
    (source/gsf/gsf.h, source/gsf/gsf.c)

Comments copied or closely paraphrased from gsf.h are so noted, so that
readers already familiar with gsflib recognize the record layout, names,
and semantics used here. No gsflib source code is reused -- this is an
independent, pure-Python re-implementation of the file format, chosen over
wrapping gsflib's compiled libgsf (as the "gsfpy" python package does)
because the prebuilt libgsf shared libraries are not available/loadable on
every platform (e.g. gsfpy's bundled libgsf is a Linux-only shared object
and cannot be loaded on macOS or Windows).

The field-level record decoders (see "Field-level record decoding" below)
are ported the same way, from gsf_dec.c's gsfDecode* and Decode*Array
functions -- the wire-format details there (byte order, scaling, signedness)
are not derivable from the gsf.h struct definitions alone, since those
describe the decoded in-memory form, not the packed on-disk encoding.

Verified current against GSF v3.11 (2026-09-01): gsf.h and gsf_dec.c from
the official v3.11 distribution are byte-identical (modulo CRLF/LF) to the
Spatialnetics/gsflib source above, so no decoder here has drifted from
current gsflib. The v3.11 change summary documents exactly one decode
behavior change since v3.10 -- beam_angle_forward (ping subrecord 18) was
briefly, mistakenly encoded signed for about 8 months in v3.10 only, then
reverted -- which this code does not special-case (see the comment at
_PING_ARRAY_SUBRECORDS[18]): a genuine v3.10 file decodes that one field
incorrectly, by design, rather than adding version-specific handling for a
short-lived encoder bug. Every other version-conditional GSF wire-format
difference this code is aware of (the ping height/SEP/GPS-tide-corrector
fields, present only at major_version > 2) is handled dynamically from the
file's own GSF_RECORD_HEADER version string, with no user-specified version
required.
"""
import argparse
import datetime
import os
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants translated from gsf.h
# ---------------------------------------------------------------------------

#: Largest ever expected record size. (gsf.h: GSF_MAX_RECORD_SIZE)
GSF_MAX_RECORD_SIZE = 524288

#: Fixed width, in bytes, of the version string carried in a
#: GSF_RECORD_HEADER record. (gsf.h: GSF_VERSION_SIZE)
GSF_VERSION_SIZE = 12

#: Size, in bytes, of a record's fixed on-disk framing: the 4-byte record
#: size field plus the 4-byte data identifier word.
GSF_RECORD_FRAMING_SIZE = 8


class FileMode(IntEnum):
    """ Define the GSF data file access flags. (gsf.h) """
    GSF_CREATE = 1
    GSF_READONLY = 2
    GSF_UPDATE = 3
    GSF_READONLY_INDEX = 4
    GSF_UPDATE_INDEX = 5
    GSF_APPEND = 6


class SeekOption(IntEnum):
    """ Define options for sequential access GSF file pointer manipulation. (gsf.h) """
    GSF_REWIND = 1
    GSF_END_OF_FILE = 2
    GSF_PREVIOUS_RECORD = 3


#: Specify a key to allow reading the next record, no matter what it is. (gsf.h: GSF_NEXT_RECORD)
GSF_NEXT_RECORD = 0


class RecordType(IntEnum):
    """
    Specify the GSF record data type numbers, for registry number zero.
    (gsf.h: GSF_RECORD_* defines)
    """
    GSF_RECORD_HEADER = 1
    GSF_RECORD_SWATH_BATHYMETRY_PING = 2
    GSF_RECORD_SOUND_VELOCITY_PROFILE = 3
    GSF_RECORD_PROCESSING_PARAMETERS = 4
    GSF_RECORD_SENSOR_PARAMETERS = 5
    GSF_RECORD_COMMENT = 6
    GSF_RECORD_HISTORY = 7
    GSF_RECORD_NAVIGATION_ERROR = 8        # 10/19/98 This record is obsolete
    GSF_RECORD_SWATH_BATHY_SUMMARY = 9
    GSF_RECORD_SINGLE_BEAM_PING = 10
    GSF_RECORD_HV_NAVIGATION_ERROR = 11    # This record replaces GSF_RECORD_NAVIGATION_ERROR
    GSF_RECORD_ATTITUDE = 12


#: Number of currently defined record data types (including 0, which is used
#: in the indexing for ping records which contain scale factor subrecords).
#: (gsf.h: NUM_REC_TYPES)
NUM_REC_TYPES = 13

#: Human readable description of each record type, adapted from the
#: struct-level comments in gsf.h documenting each record's data structure.
RECORD_TYPE_DESCRIPTIONS = {
    RecordType.GSF_RECORD_HEADER:
        "GSF header record. Identifies the GSF version used to create the file.",
    RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING:
        "Data structure for a ping from a swath bathymetric system.",
    RecordType.GSF_RECORD_SOUND_VELOCITY_PROFILE:
        "Sound velocity profile record.",
    RecordType.GSF_RECORD_PROCESSING_PARAMETERS:
        "Internal record structure for processing parameters.",
    RecordType.GSF_RECORD_SENSOR_PARAMETERS:
        "Sensor parameters record.",
    RecordType.GSF_RECORD_COMMENT:
        "Comment record.",
    RecordType.GSF_RECORD_HISTORY:
        "History record.",
    RecordType.GSF_RECORD_NAVIGATION_ERROR:
        "Navigation error record. (Obsolete; replaced by GSF_RECORD_HV_NAVIGATION_ERROR.)",
    RecordType.GSF_RECORD_SWATH_BATHY_SUMMARY:
        "Swath bathymetry summary record.",
    RecordType.GSF_RECORD_SINGLE_BEAM_PING:
        "Single beam ping record.",
    RecordType.GSF_RECORD_HV_NAVIGATION_ERROR:
        ("Horizontal/Vertical navigation error record. Replaces "
         "GSF_RECORD_NAVIGATION_ERROR. (The HV stands for Horizontal and Vertical.)"),
    RecordType.GSF_RECORD_ATTITUDE:
        "Attitude record: one or more time-tagged pitch/roll/heave/heading measurements.",
}


# ---------------------------------------------------------------------------
# GSF error conditions (gsf.h error codes). The C library returns these as
# integer error codes via a global gsfError; here they are raised as
# exceptions instead.
# ---------------------------------------------------------------------------

class GSFError(Exception):
    """ Base class for errors raised while reading or writing a GSF file. """


class GSFRecordSizeError(GSFError):
    """ gsf.h: GSF_RECORD_SIZE_ERROR -- a record's declared size is <= 8
    bytes, or greater than GSF_MAX_RECORD_SIZE. """


class GSFUnrecognizedRecordIDError(GSFError):
    """ gsf.h: GSF_UNRECOGNIZED_RECORD_ID -- recordID is not a value between
    1 and NUM_REC_TYPES-1. """


class GSFPartialRecordAtEndOfFileError(GSFError):
    """ gsf.h: GSF_PARTIAL_RECORD_AT_END_OF_FILE -- fewer bytes remain in the
    file than the current record declares. """


class GSFChecksumFailureError(GSFError):
    """ gsf.h: GSF_CHECKSUM_FAILURE -- a record's stored checksum does not
    match the computed checksum of its data. """


def gsf_checksum(data):
    """
    Compute the GSF record checksum: the modulo-32 byte-wise sum of `data`.

    Ported from gsf.c's gsfChecksum(): "This function computes and returns
    the modulo-32 form byte-wise sum of the num_bytes starting at buff."
    Used to verify (or generate) the optional 4-byte checksum that precedes
    a record's payload when the data identifier's checksum flag is set.

    :param data: bytes -- the record payload, NOT including the checksum
        field itself.
    :return: int -- the computed checksum.
    """
    return sum(data) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# gsfDataID -- gsf.h t_gsfDataID
# ---------------------------------------------------------------------------

@dataclass
class gsfDataID:
    """
    Mirrors gsf.h's t_gsfDataID, "the GSF Data Identifier structure".

    On disk this is packed into a single 4-byte big-endian ("network byte
    order", per gsf.c) word immediately following the 4-byte record size
    field. gsflib unpacks it (gsf.c, gsfUnpackStream) as:

        checksumFlag = did & 0x80000000            # bit 31       (boolean)
        reserved     = (did & 0x7FC00000) >> 22     # bits 22-30   (9 bits)
        recordID     = did & 0x003FFFFF             # bits 00-21
                       #  bits 00-11 => data type number
                       #  bits 12-22 => registry number
    """
    checksumFlag: bool         # boolean
    reserved: int              # up to 9 bits
    recordID: int              # bits 00-11 => data type number; bits 12-22 => registry number
    record_number: int = 0     # specifies the nth occurrence of record type specified by
                                # recordID; relevant only for direct access; counts from 1


###########################################################
# Field-level record decoding (subset)
#
# Ported from gsf_dec.c: gsfDecodeHeader, gsfDecodeSwathBathySummary,
# gsfDecodeSwathBathymetryPing, DecodeScaleFactors, the per-width/
# per-signedness Decode*Array family, gsfDecodeSoundVelocityProfile,
# gsfDecodeProcessingParameters/gsfDecodeSensorParameters,
# gsfDecodeComment, gsfDecodeHistory, gsfDecodeNavigationError,
# gsfDecodeHVNavigationError, gsfDecodeAttitude, and the fixed portion
# of gsfDecodeSinglebeam.
#
# Deliberately NOT decoded (reported as a byte count instead): the
# per-sensor "_SPECIFIC" subrecords (~30 vendor-specific sonar payloads),
# gsflib's own optional RLE array compression (DecodeCompressedArray),
# the 2-bit packed GSF_SWATH_BATHY_SUBRECORD_QUALITY_FLAGS_ARRAY, and the
# per-beam variable-length GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY.
###########################################################

#: Identifies the scale factors subrecord within a ping's subrecord stream.
#: (gsf.h: GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS)
_SUBRECORD_SCALE_FACTORS = 100
#: (gsf.h: GSF_SWATH_BATHY_SUBRECORD_BEAM_FLAGS_ARRAY) -- raw bytes, no scale factor.
_SUBRECORD_BEAM_FLAGS_ARRAY = 16

# Beam-array subrecord id -> (attribute name, column label, is signed on disk).
# See gsf.h's GSF_SWATH_BATHY_SUBRECORD_* defines for the id values, and
# gsf_dec.c's switch on subrecord_id (in gsfDecodeSwathBathymetryPing) for
# which Decode*Array (signed vs. unsigned) each one calls. Every entry here
# decodes as: value = raw_int / multiplier - offset, using whichever of the
# 1/2/4-byte {signed,unsigned} widths subrecord_size / number_beams implies
# (subrecord ids 22, 23, 25 are additionally truncated to an integer, per
# DecodeFromByteToUnsignedShortArray).
_PING_ARRAY_SUBRECORDS = {
    1: ("depth", "Depth_m", False),
    2: ("across_track", "AcrossTrack_m", True),
    3: ("along_track", "AlongTrack_m", True),
    4: ("travel_time", "TravelTime_s", False),
    5: ("beam_angle", "BeamAngle_deg", True),
    6: ("mc_amplitude", "MeanCalAmplitude_dB", True),
    7: ("mr_amplitude", "MeanRelAmplitude_dB", False),
    8: ("echo_width", "EchoWidth_s", False),
    9: ("quality_factor", "QualityFactor", False),
    10: ("receive_heave", "ReceiveHeave_m", True),
    11: ("depth_error", "DepthError_m", False),           # obsolete
    12: ("across_track_error", "AcrossTrackError_m", False),  # obsolete
    13: ("along_track_error", "AlongTrackError_m", False),    # obsolete
    14: ("nominal_depth", "NominalDepth_m", False),
    # 15 GSF_SWATH_BATHY_SUBRECORD_QUALITY_FLAGS_ARRAY: 2-bit packed, not decoded.
    # 16 GSF_SWATH_BATHY_SUBRECORD_BEAM_FLAGS_ARRAY: handled separately (no scale factor).
    17: ("signal_to_noise", "SignalToNoise_dB", True),
    # 18 beam_angle_forward: unsigned in every GSF version except v3.10,
    # where it was briefly (and, per the v3.11 change summary, mistakenly)
    # encoded signed for about 8 months before being reverted. Decoded
    # unsigned unconditionally here -- correct for v3.09 and earlier and for
    # v3.11+, silently wrong only for a file actually written by a v3.10
    # library. Deliberately not special-cased for that narrow window.
    18: ("beam_angle_forward", "BeamAngleForward_deg", False),
    19: ("vertical_error", "VerticalError_m", False),
    20: ("horizontal_error", "HorizontalError_m", False),
    # 21 GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY: variable length per beam, not decoded.
    22: ("sector_number", "SectorNumber", False),
    23: ("detection_info", "DetectionInfo", False),
    24: ("incident_beam_adj", "IncidentBeamAdj_deg", True),
    25: ("system_cleaning", "SystemCleaning", False),
    26: ("doppler_corr", "DopplerCorr", True),
    27: ("sonar_vert_uncert", "SonarVertUncert_m", False),
    28: ("sonar_horz_uncert", "SonarHorzUncert_m", False),
    29: ("detection_window", "DetectionWindow_s", False),
    30: ("mean_abs_coeff", "MeanAbsCoeff", False),
    31: ("TVG_dB", "TVG_dB", False),
}

#: Subrecord ids decoded into an integer rather than a double (gsflib decodes
#: these into an unsigned short field via DecodeFromByteToUnsignedShortArray).
_PING_ARRAY_INTEGER_SUBRECORDS = {22, 23, 25}

_ARRAY_DTYPE = {
    (1, False): '>u1', (1, True): '>i1',
    (2, False): '>u2', (2, True): '>i2',
    (4, False): '>u4', (4, True): '>i4',
}

#: Identifies the KMALL (Kongsberg SIS 5 / .kmall-derived) sensor-specific
#: subrecord within a ping's subrecord stream. (gsf.h: GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC)
_SUBRECORD_KMALL_SPECIFIC = 156
#: Identifies the per-beam backscatter time series subrecord.
#: (gsf.h: GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY)
_SUBRECORD_INTENSITY_SERIES_ARRAY = 21

#: Every vendor sensor-specific ("_SPECIFIC") ping subrecord id and name,
#: from gsf.h's GSF_SWATH_BATHY_SUBRECORD_* defines. Used to label subrecords
#: that have no field-level decoder here (only GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC,
#: id 156, is decoded -- see _decode_kmall_specific()) with their proper
#: name instead of a bare numeric id.
_SENSOR_SPECIFIC_SUBRECORD_NAMES = {
    102: "SEABEAM_SPECIFIC",
    103: "EM12_SPECIFIC",
    104: "EM100_SPECIFIC",
    105: "EM950_SPECIFIC",
    106: "EM121A_SPECIFIC",
    107: "EM121_SPECIFIC",
    108: "SASS_SPECIFIC",              # obsolete
    109: "SEAMAP_SPECIFIC",
    110: "SEABAT_SPECIFIC",
    111: "EM1000_SPECIFIC",
    112: "TYPEIII_SEABEAM_SPECIFIC",   # obsolete
    113: "SB_AMP_SPECIFIC",
    114: "SEABAT_II_SPECIFIC",
    115: "SEABAT_8101_SPECIFIC",
    116: "SEABEAM_2112_SPECIFIC",
    117: "ELAC_MKII_SPECIFIC",
    118: "EM3000_SPECIFIC",
    119: "EM1002_SPECIFIC",
    120: "EM300_SPECIFIC",
    121: "CMP_SASS_SPECIFIC",
    122: "RESON_8101_SPECIFIC",
    123: "RESON_8111_SPECIFIC",
    124: "RESON_8124_SPECIFIC",
    125: "RESON_8125_SPECIFIC",
    126: "RESON_8150_SPECIFIC",
    127: "RESON_8160_SPECIFIC",
    128: "EM120_SPECIFIC",
    129: "EM3002_SPECIFIC",
    130: "EM3000D_SPECIFIC",
    131: "EM3002D_SPECIFIC",
    132: "EM121A_SIS_SPECIFIC",
    133: "EM710_SPECIFIC",
    134: "EM302_SPECIFIC",
    135: "EM122_SPECIFIC",
    136: "GEOSWATH_PLUS_SPECIFIC",
    137: "KLEIN_5410_BSS_SPECIFIC",
    138: "RESON_7125_SPECIFIC",
    139: "EM2000_SPECIFIC",
    140: "EM300_RAW_SPECIFIC",
    141: "EM1002_RAW_SPECIFIC",
    142: "EM2000_RAW_SPECIFIC",
    143: "EM3000_RAW_SPECIFIC",
    144: "EM120_RAW_SPECIFIC",
    145: "EM3002_RAW_SPECIFIC",
    146: "EM3000D_RAW_SPECIFIC",
    147: "EM3002D_RAW_SPECIFIC",
    148: "EM121A_SIS_RAW_SPECIFIC",
    149: "EM2040_SPECIFIC",
    150: "DELTA_T_SPECIFIC",
    151: "R2SONIC_2022_SPECIFIC",
    152: "R2SONIC_2024_SPECIFIC",
    153: "R2SONIC_2020_SPECIFIC",
    155: "RESON_TSERIES_SPECIFIC",     # 154 is unused/reserved in gsf.h
    _SUBRECORD_KMALL_SPECIFIC: "KMALL_SPECIFIC",
    157: "ME70BO_SPECIFIC",
}


def _decode_kmall_imagery_specific(payload, pos):
    """
    Decode the KMALL sensor-specific portion of a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord. Ported from
    gsf_dec.c's DecodeKMALLImagerySpecific(): entirely spare/reserved for
    the KMALL sensor, so this exists only to advance past its 64 bytes.

    :return: bytes_consumed (always 64).
    """
    return 64


def _decode_kmall_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC subrecord (id 156):
    Kongsberg SIS 5 / .kmall-derived per-ping sensor metadata, carried over
    from the originating #MRZ datagram's header/cmnPart/pingInfo, plus its
    per-transmit-sector and per-extra-detection-class arrays. Ported from
    gsf_dec.c's DecodeKMALLSpecific().

    :return: (scalars: dict, sector_rows: list[dict], class_rows: list[dict],
        bytes_consumed: int)
    """
    start = pos
    s = {}

    s['GSFKMALLVersion'] = payload[pos]; pos += 1
    s['DgmType'] = payload[pos]; pos += 1
    s['DgmVersion'] = payload[pos]; pos += 1
    s['SystemID'] = payload[pos]; pos += 1
    (s['EchoSounderID'],) = struct.unpack_from('>H', payload, pos); pos += 2
    pos += 8  # spare1

    (s['NumBytesCmnPart'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (s['PingCnt'],) = struct.unpack_from('>H', payload, pos); pos += 2
    s['RxFansPerPing'] = payload[pos]; pos += 1
    s['RxFanIndex'] = payload[pos]; pos += 1
    s['SwathsPerPing'] = payload[pos]; pos += 1
    s['SwathAlongPosition'] = payload[pos]; pos += 1
    s['TxTransducerInd'] = payload[pos]; pos += 1
    s['RxTransducerInd'] = payload[pos]; pos += 1
    s['NumRxTransducers'] = payload[pos]; pos += 1
    s['AlgorithmType'] = payload[pos]; pos += 1
    pos += 16  # spare2

    (s['NumBytesInfoData'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (raw,) = struct.unpack_from('>I', payload, pos); s['PingRate_Hz'] = raw / 1.0e5; pos += 4
    s['BeamSpacing'] = payload[pos]; pos += 1
    s['DepthMode'] = payload[pos]; pos += 1
    s['SubDepthMode'] = payload[pos]; pos += 1
    s['DistanceBtwSwath'] = payload[pos]; pos += 1
    s['DetectionMode'] = payload[pos]; pos += 1
    s['PulseForm'] = payload[pos]; pos += 1
    (raw,) = struct.unpack_from('>i', payload, pos); s['FrequencyMode_Hz'] = float(raw); pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['FreqRangeLowLim_Hz'] = raw / 1.0e3; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['FreqRangeHighLim_Hz'] = raw / 1.0e3; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['MaxTotalTxPulseLength_sec'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['MaxEffTxPulseLength_sec'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['MaxEffTxBandWidth_Hz'] = raw / 1.0e3; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['AbsCoeff_dBPerkm'] = raw / 1.0e3; pos += 4
    (raw,) = struct.unpack_from('>h', payload, pos); s['PortSectorEdge_deg'] = raw / 1.0e2; pos += 2
    (raw,) = struct.unpack_from('>h', payload, pos); s['StarbSectorEdge_deg'] = raw / 1.0e2; pos += 2
    # gsf_dec.c reads a port/starboard mean-coverage-in-degrees pair here,
    # then immediately reads and overwrites it with a second pair a few
    # lines later (an apparent copy/paste artifact in the reference
    # decoder) -- the first pair's bytes are consumed but its decoded
    # values are discarded. Preserved here to keep byte alignment correct.
    pos += 4
    (raw,) = struct.unpack_from('>h', payload, pos); s['PortMeanCov_deg'] = raw / 1.0e2; pos += 2
    (raw,) = struct.unpack_from('>h', payload, pos); s['StarbMeanCov_deg'] = raw / 1.0e2; pos += 2
    (raw,) = struct.unpack_from('>h', payload, pos); s['PortMeanCov_m'] = float(raw); pos += 2
    (raw,) = struct.unpack_from('>h', payload, pos); s['StarbMeanCov_m'] = float(raw); pos += 2
    s['ModeAndStabilisation'] = payload[pos]; pos += 1
    s['RuntimeFilter1'] = payload[pos]; pos += 1
    (s['RuntimeFilter2'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (s['PipeTrackingStatus'],) = struct.unpack_from('>i', payload, pos); pos += 4
    (raw,) = struct.unpack_from('>H', payload, pos); s['TransmitArraySizeUsed_deg'] = raw / 1.0e3; pos += 2
    (raw,) = struct.unpack_from('>H', payload, pos); s['ReceiveArraySizeUsed_deg'] = raw / 1.0e3; pos += 2
    (raw,) = struct.unpack_from('>h', payload, pos); s['TransmitPower_dB'] = raw / 1.0e2; pos += 2
    (s['SLrampUpTimeRemaining'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (raw,) = struct.unpack_from('>i', payload, pos); s['YawAngle_deg'] = raw / 1.0e6; pos += 4
    (num_tx_sectors,) = struct.unpack_from('>H', payload, pos); pos += 2
    s['NumTxSectors'] = num_tx_sectors
    (s['NumBytesPerTxSector'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (raw,) = struct.unpack_from('>i', payload, pos); s['HeadingVessel_deg'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['SoundSpeedAtTxDepth_mPerSec'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['TxTransducerDepth_m'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['ZWaterLevelReRefPoint_m'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['XKmallToAll_m'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['YKmallToAll_m'] = raw / 1.0e6; pos += 4
    s['LatLongInfo'] = payload[pos]; pos += 1
    s['PosSensorStatus'] = payload[pos]; pos += 1
    s['AttitudeSensorStatus'] = payload[pos]; pos += 1
    (raw,) = struct.unpack_from('>i', payload, pos); s['Latitude_deg'] = raw / 1.0e7; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['Longitude_deg'] = raw / 1.0e7; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['EllipsoidHeightReRefPoint_m'] = raw / 1.0e3; pos += 4
    pos += 32  # spare3

    sector_rows = []
    for _ in range(min(num_tx_sectors, 9)):  # gsf.h: GSF_MAX_KMALL_SECTORS
        row = {}
        row['TxSectorNumb'] = payload[pos]; pos += 1
        row['TxArrNumber'] = payload[pos]; pos += 1
        row['TxSubArray'] = payload[pos]; pos += 1
        (raw,) = struct.unpack_from('>i', payload, pos); row['SectorTransmitDelay_sec'] = raw / 1.0e6; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['TiltAngleReTx_deg'] = raw / 1.0e6; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['TxNominalSourceLevel_dB'] = raw / 1.0e6; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['TxFocusRange_m'] = raw / 1.0e3; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['CentreFreq_Hz'] = raw / 1.0e3; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['SignalBandWidth_Hz'] = raw / 1.0e3; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['TotalSignalLength_sec'] = raw / 1.0e6; pos += 4
        row['PulseShading'] = payload[pos]; pos += 1
        row['SignalWaveForm'] = payload[pos]; pos += 1
        (raw,) = struct.unpack_from('>i', payload, pos); row['HighVoltageLevel_dB'] = raw / 1.0e6; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['SectorTrackingCorr_dB'] = raw / 1.0e6; pos += 4
        (raw,) = struct.unpack_from('>i', payload, pos); row['EffectiveSignalLength_sec'] = raw / 1.0e6; pos += 4
        pos += 8  # spare1
        sector_rows.append(row)

    (s['NumBytesRxInfo'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (s['NumSoundingsMaxMain'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (s['NumSoundingsValidMain'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (s['NumBytesPerSounding'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (wc_int,) = struct.unpack_from('>i', payload, pos); pos += 4
    (wc_frac,) = struct.unpack_from('>I', payload, pos); pos += 4
    s['WCSampleRate'] = wc_int + wc_frac / 1.0e9
    (sb_int,) = struct.unpack_from('>i', payload, pos); pos += 4
    (sb_frac,) = struct.unpack_from('>I', payload, pos); pos += 4
    s['SeabedImageSampleRate'] = sb_int + sb_frac / 1.0e9
    (raw,) = struct.unpack_from('>i', payload, pos); s['BSnormal_dB'] = raw / 1.0e6; pos += 4
    (raw,) = struct.unpack_from('>i', payload, pos); s['BSoblique_dB'] = raw / 1.0e6; pos += 4
    (s['ExtraDetectionAlarmFlag'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (s['NumExtraDetections'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_extra_classes,) = struct.unpack_from('>H', payload, pos); pos += 2
    s['NumExtraDetectionClasses'] = num_extra_classes
    (s['NumBytesPerClass'],) = struct.unpack_from('>H', payload, pos); pos += 2
    pos += 32  # spare4

    class_rows = []
    for _ in range(min(num_extra_classes, 11)):  # gsf.h: GSF_MAX_KMALL_EXTRA_CLASSES
        (num_in_class,) = struct.unpack_from('>H', payload, pos); pos += 2
        alarm_flag = payload[pos]; pos += 1
        pos += 32  # spare
        class_rows.append({'NumExtraDetInClass': num_in_class, 'AlarmFlag': alarm_flag})

    pos += 32  # spare5

    return s, sector_rows, class_rows, pos - start


def _decode_brb_intensity(payload, pos, num_beams, sensor_id):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord: a
    per-beam receive-beam backscatter time series. Ported from gsf_dec.c's
    DecodeBRBIntensity(). Only the KMALL sensor-imagery format (sensor_id
    156) is supported -- every other vendor's imagery-specific block has a
    different, sensor-dependent size, which would misalign every beam
    after it; unsupported sensors return None rather than guess.

    :param sensor_id: the vendor "_SPECIFIC" subrecord id most recently
        seen for this ping (identifies which sensor-imagery format, if
        any, precedes the per-beam samples).
    :return: (header: dict, beam_rows: list[dict] with 'SampleCount',
        'DetectSample', 'StartRangeSamples', 'Samples' (a list of ints),
        bytes_consumed), or None if the sensor-imagery format isn't
        supported.
    """
    if num_beams <= 0:
        return None

    start = pos
    bits_per_sample = payload[pos]; pos += 1
    (applied_corrections,) = struct.unpack_from('>I', payload, pos); pos += 4
    pos += 16  # spare

    if sensor_id == _SUBRECORD_KMALL_SPECIFIC:
        pos += _decode_kmall_imagery_specific(payload, pos)
    else:
        return None

    bytes_per_sample = bits_per_sample // 8
    header = {'BitsPerSample': bits_per_sample, 'AppliedCorrections': applied_corrections}

    beam_rows = []
    for _beam in range(num_beams):
        (sample_count, detect_sample, start_range_samples) = struct.unpack_from('>3H', payload, pos)
        pos += 6
        pos += 6  # spare

        if bits_per_sample == 12:
            samples = []
            i = 0
            while i < sample_count:
                b0, b1, b2 = payload[pos], payload[pos + 1], payload[pos + 2]
                samples.append((b0 << 4) | (b1 >> 4))
                if i + 1 < sample_count:
                    samples.append(((b1 & 0x0F) << 8) | b2)
                pos += 3
                i += 2
        elif bytes_per_sample in (1, 2, 4):
            dtype = {1: '>u1', 2: '>u2', 4: '>u4'}[bytes_per_sample]
            samples = list(np.frombuffer(payload, dtype=dtype, count=sample_count, offset=pos))
            pos += sample_count * bytes_per_sample
        else:
            return None

        beam_rows.append({
            'SampleCount': sample_count,
            'DetectSample': detect_sample,
            'StartRangeSamples': start_range_samples,
            'Samples': samples,
        })

    return header, beam_rows, pos - start


def _gsf_timestamp(sec, nsec):
    """ Convert a GSF (seconds, nanoseconds)-since-epoch pair to a UTC datetime. """
    return datetime.datetime.fromtimestamp(sec + nsec / 1.0e9, tz=datetime.timezone.utc)


def _decode_scale_factors(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS subrecord.
    Ported from gsf_dec.c's DecodeScaleFactors().

    :return: (scale_table, bytes_consumed) where scale_table maps
        subrecordID -> (multiplier, offset, compressionFlag).
    """
    num_subrecords, = struct.unpack_from('>I', payload, pos)
    p = pos + 4
    table = {}
    for _ in range(num_subrecords):
        word, = struct.unpack_from('>I', payload, p)
        subrecord_id = (word >> 24) & 0xFF
        compression_flag = (word >> 16) & 0xFF
        p += 4
        multiplier, = struct.unpack_from('>I', payload, p)   # unsigned
        offset, = struct.unpack_from('>i', payload, p + 4)   # signed
        p += 8
        table[subrecord_id] = (float(multiplier), float(offset), compression_flag)
    return table, p - pos


def _decode_ping_array(payload, pos, size, num_beams, multiplier, offset, signed, truncate_to_int):
    """
    Decode one beam-array subrecord's raw bytes into engineering units,
    vectorized with numpy: value = raw_int / multiplier - offset. The
    on-disk width (1, 2, or 4 bytes per beam) is inferred from
    size / num_beams, matching gsf_dec.c's primary (uncompressed) decode
    path. Returns None if the width can't be determined (size isn't an
    exact multiple of num_beams, or resolves to an unsupported width).
    """
    if num_beams <= 0 or size % num_beams != 0:
        return None
    bytes_per_value = size // num_beams
    dtype = _ARRAY_DTYPE.get((bytes_per_value, signed))
    if dtype is None:
        return None

    raw = np.frombuffer(payload, dtype=dtype, count=num_beams, offset=pos)
    values = raw.astype(np.float64) / multiplier - offset
    if truncate_to_int:
        values = np.trunc(values).astype(np.int64)
    return values


def _decode_swath_bathymetry_ping(payload, major_version, scale_factors, decode_intensity=False):
    """
    Decode a GSF_RECORD_SWATH_BATHYMETRY_PING payload: the fixed-format
    scalar fields, followed by the variable subrecord stream (scale
    factors, per-beam arrays, and the vendor sensor-specific subrecord).
    Ported from gsf_dec.c's gsfDecodeSwathBathymetryPing().

    :param scale_factors: dict mapping subrecordID -> (multiplier, offset,
        compressionFlag), carried in from the caller and updated in place
        when this ping carries its own GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS
        subrecord -- pings after the first do not always repeat their scale
        factors, and are expected to reuse whatever this file last decoded.
    :param decode_intensity: if True, fully decode the per-beam
        GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (the
        backscatter time series) into tables['IntensityTimeSeries'] --
        this can be tens of thousands of samples per ping, so it's opt-in
        and normally left False (used by gsf.print_intensity_series(), not
        the default gsf.print_records() path).
    :return: (scalars: dict, tables: dict[str, pandas.DataFrame], notes: list[str])
    """
    scalars = {}
    notes = []

    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['PingTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (lon_raw, lat_raw) = struct.unpack_from('>2i', payload, 8)
    scalars['Longitude_deg'] = lon_raw / 1.0e7
    scalars['Latitude_deg'] = lat_raw / 1.0e7

    (number_beams, center_beam, ping_flags, reserved) = struct.unpack_from('>4H', payload, 16)
    scalars['NumberBeams'] = number_beams
    scalars['CenterBeam'] = center_beam
    scalars['PingFlags'] = ping_flags

    (tide_raw,) = struct.unpack_from('>h', payload, 24)
    scalars['TideCorrector_m'] = tide_raw / 100.0

    (depth_corr_raw,) = struct.unpack_from('>i', payload, 26)
    scalars['DepthCorrector_m'] = depth_corr_raw / 100.0

    (heading_raw,) = struct.unpack_from('>H', payload, 30)
    scalars['Heading_deg'] = heading_raw / 100.0

    (pitch_raw, roll_raw, heave_raw) = struct.unpack_from('>3h', payload, 32)
    scalars['Pitch_deg'] = pitch_raw / 100.0
    scalars['Roll_deg'] = roll_raw / 100.0
    scalars['Heave_m'] = heave_raw / 100.0

    (course_raw, speed_raw) = struct.unpack_from('>2H', payload, 38)
    scalars['Course_deg'] = course_raw / 100.0
    scalars['Speed_kn'] = speed_raw / 100.0

    pos = 42
    if major_version > 2:
        (height_raw, sep_raw, gps_tide_raw) = struct.unpack_from('>3i', payload, pos)
        scalars['Height_m'] = height_raw / 1000.0
        scalars['SEP_m'] = sep_raw / 1000.0
        scalars['GPSTideCorrector_m'] = gps_tide_raw / 1000.0
        pos += 14  # 3 x 4-byte fields, plus 2 spare bytes

    beam_columns = {}
    tables = {}
    sensor_id = None

    while len(payload) - pos > 4:
        word, = struct.unpack_from('>I', payload, pos)
        subrecord_id = (word >> 24) & 0xFF
        subrecord_size = word & 0x00FFFFFF
        pos += 4

        if subrecord_id == _SUBRECORD_SCALE_FACTORS:
            table, _consumed = _decode_scale_factors(payload, pos)
            scale_factors.clear()
            scale_factors.update(table)

        elif subrecord_id == _SUBRECORD_BEAM_FLAGS_ARRAY:
            values = np.frombuffer(payload, dtype='>u1', count=number_beams, offset=pos) \
                if subrecord_size == number_beams else None
            if values is not None:
                beam_columns['BeamFlags'] = values
            else:
                notes.append("BeamFlags (%d bytes) not decoded: unexpected size" % subrecord_size)

        elif subrecord_id in _PING_ARRAY_SUBRECORDS:
            _attr, label, signed = _PING_ARRAY_SUBRECORDS[subrecord_id]
            sf = scale_factors.get(subrecord_id)
            if sf is None:
                notes.append("%s (%d bytes) not decoded: no scale factors available" % (label, subrecord_size))
            else:
                multiplier, offset, _flags = sf
                values = _decode_ping_array(
                    payload, pos, subrecord_size, number_beams, multiplier, offset,
                    signed, subrecord_id in _PING_ARRAY_INTEGER_SUBRECORDS)
                if values is None:
                    notes.append("%s (%d bytes) not decoded: unsupported encoding" % (label, subrecord_size))
                else:
                    beam_columns[label] = values

        elif subrecord_id == _SUBRECORD_KMALL_SPECIFIC:
            sensor_id = subrecord_id
            try:
                kmall_scalars, sector_rows, class_rows, _consumed = _decode_kmall_specific(payload, pos)
            except (struct.error, IndexError) as exc:
                notes.append("KMALL_SPECIFIC (156, %d bytes) not decoded: %s" % (subrecord_size, exc))
            else:
                scalars.update({"KMALL." + k: v for k, v in kmall_scalars.items()})
                if sector_rows:
                    sectors = pd.DataFrame(sector_rows)
                    sectors.index.name = 'Sector'
                    tables['TxSectors'] = sectors
                if class_rows:
                    classes = pd.DataFrame(class_rows)
                    classes.index.name = 'ExtraDetectionClass'
                    tables['ExtraDetectionClasses'] = classes

        elif subrecord_id == _SUBRECORD_INTENSITY_SERIES_ARRAY:
            if not decode_intensity:
                notes.append(
                    "IntensityTimeSeries (21, %d bytes) not decoded here: "
                    "use gsf.print_intensity_series() / -I" % subrecord_size)
            else:
                try:
                    decoded = _decode_brb_intensity(payload, pos, number_beams, sensor_id)
                except (struct.error, IndexError) as exc:
                    notes.append("IntensityTimeSeries (21, %d bytes) not decoded: %s" % (subrecord_size, exc))
                else:
                    if decoded is None:
                        notes.append(
                            "IntensityTimeSeries (21, %d bytes) not decoded: unsupported sensor imagery "
                            "format (sensor_id=%s)" % (subrecord_size, sensor_id))
                    else:
                        _header, beam_rows, _consumed = decoded
                        series = pd.DataFrame(beam_rows)
                        series.index.name = 'Beam'
                        tables['IntensityTimeSeries'] = series

        elif subrecord_id in _SENSOR_SPECIFIC_SUBRECORD_NAMES:
            sensor_id = subrecord_id
            notes.append("%s (%d, %d bytes) not decoded" %
                         (_SENSOR_SPECIFIC_SUBRECORD_NAMES[subrecord_id], subrecord_id, subrecord_size))

        else:
            notes.append("subrecord id %d (%d bytes) not decoded" % (subrecord_id, subrecord_size))

        pos += subrecord_size

    if beam_columns:
        beams = pd.DataFrame(beam_columns)
        beams.index.name = 'Beam'
        tables['Beams'] = beams

    return scalars, tables, notes


def _decode_single_beam_ping(payload):
    """
    Decode the fixed-format portion of a GSF_RECORD_SINGLE_BEAM_PING
    payload. Ported from gsf_dec.c's gsfDecodeSinglebeam(); the
    sensor-specific tail (echosounder-dependent) is not decoded.
    """
    scalars = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['PingTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (lon_raw, lat_raw) = struct.unpack_from('>2i', payload, 8)
    scalars['Longitude_deg'] = lon_raw / 1.0e7
    scalars['Latitude_deg'] = lat_raw / 1.0e7

    (tide_raw,) = struct.unpack_from('>h', payload, 16)
    scalars['TideCorrector_m'] = tide_raw / 100.0

    (depth_corr_raw,) = struct.unpack_from('>i', payload, 18)
    scalars['DepthCorrector_m'] = depth_corr_raw / 100.0

    (heading_raw,) = struct.unpack_from('>H', payload, 22)
    scalars['Heading_deg'] = heading_raw / 100.0

    (pitch_raw, roll_raw, heave_raw) = struct.unpack_from('>3h', payload, 24)
    scalars['Pitch_deg'] = pitch_raw / 100.0
    scalars['Roll_deg'] = roll_raw / 100.0
    scalars['Heave_m'] = heave_raw / 100.0

    (depth_raw,) = struct.unpack_from('>i', payload, 30)
    scalars['Depth_m'] = depth_raw / 100.0

    (ssc_raw,) = struct.unpack_from('>h', payload, 34)
    scalars['SoundSpeedCorrection_m'] = ssc_raw / 100.0

    (pos_type,) = struct.unpack_from('>H', payload, 36)
    scalars['PositioningSystemType'] = pos_type

    remaining = len(payload) - 38
    notes = []
    if remaining > 4:
        notes.append("sensor-specific data (%d bytes) not decoded" % remaining)

    return scalars, {}, notes


def _decode_swath_bathy_summary(payload):
    """ Ported from gsf_dec.c's gsfDecodeSwathBathySummary(). """
    scalars = {}
    (start_sec, start_nsec, end_sec, end_nsec) = struct.unpack_from('>4I', payload, 0)
    scalars['StartTime'] = _gsf_timestamp(start_sec, start_nsec).isoformat()
    scalars['EndTime'] = _gsf_timestamp(end_sec, end_nsec).isoformat()

    (min_lat_raw, min_lon_raw, max_lat_raw, max_lon_raw) = struct.unpack_from('>4i', payload, 16)
    scalars['MinLatitude_deg'] = min_lat_raw / 1.0e7
    scalars['MinLongitude_deg'] = min_lon_raw / 1.0e7
    scalars['MaxLatitude_deg'] = max_lat_raw / 1.0e7
    scalars['MaxLongitude_deg'] = max_lon_raw / 1.0e7

    (min_depth_raw, max_depth_raw) = struct.unpack_from('>2i', payload, 32)
    scalars['MinDepth_m'] = min_depth_raw / 100.0
    scalars['MaxDepth_m'] = max_depth_raw / 100.0

    return scalars, {}, []


def _decode_sound_velocity_profile(payload):
    """ Ported from gsf_dec.c's gsfDecodeSoundVelocityProfile(). """
    scalars = {}
    (obs_sec, obs_nsec, app_sec, app_nsec) = struct.unpack_from('>4I', payload, 0)
    scalars['ObservationTime'] = _gsf_timestamp(obs_sec, obs_nsec).isoformat()
    scalars['ApplicationTime'] = _gsf_timestamp(app_sec, app_nsec).isoformat()

    (lon_raw, lat_raw) = struct.unpack_from('>2i', payload, 16)
    scalars['Longitude_deg'] = lon_raw / 1.0e7
    scalars['Latitude_deg'] = lat_raw / 1.0e7

    (number_points,) = struct.unpack_from('>I', payload, 24)
    scalars['NumberPoints'] = number_points

    raw = np.frombuffer(payload, dtype='>u4', count=2 * number_points, offset=28)
    depth = raw[0::2].astype(np.float64) / 100.0
    sound_speed = raw[1::2].astype(np.float64) / 100.0
    table = pd.DataFrame({'Depth_m': depth, 'SoundSpeed_mPerSec': sound_speed})
    table.index.name = 'Point'

    return scalars, {'Profile': table}, []


def _decode_name_value_parameters(payload):
    """
    Decode a GSF_RECORD_PROCESSING_PARAMETERS or GSF_RECORD_SENSOR_PARAMETERS
    payload -- both share the same wire format. Ported from gsf_dec.c's
    gsfDecodeProcessingParameters() / gsfDecodeSensorParameters(). Each
    parameter is already a "NAME=VALUE" string, so this maps directly onto
    scalar key/value output.

    In practice, encoders commonly include a trailing NUL byte within a
    parameter's counted size (as a C-string terminator baked into the
    file); that byte is stripped here rather than surfaced as a literal
    '\\x00' in the printed value.
    """
    scalars = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['ParamTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (number_parameters,) = struct.unpack_from('>H', payload, 8)
    pos = 10
    for _ in range(number_parameters):
        (size,) = struct.unpack_from('>h', payload, pos)
        pos += 2
        text = payload[pos:pos + size].decode('ascii', 'replace').rstrip('\x00')
        pos += size
        name, sep, value = text.partition('=')
        scalars[name if sep else text] = value if sep else ''

    return scalars, {}, []


def _decode_comment(payload):
    """ Ported from gsf_dec.c's gsfDecodeComment(). """
    scalars = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['CommentTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (length,) = struct.unpack_from('>I', payload, 8)
    scalars['Comment'] = payload[12:12 + length].decode('ascii', 'replace')

    return scalars, {}, []


def _decode_history(payload):
    """ Ported from gsf_dec.c's gsfDecodeHistory(). """
    scalars = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['HistoryTime'] = _gsf_timestamp(sec, nsec).isoformat()

    pos = 8
    for key in ('HostName', 'OperatorName', 'CommandLine', 'Comment'):
        (length,) = struct.unpack_from('>H', payload, pos)
        pos += 2
        scalars[key] = payload[pos:pos + length].decode('ascii', 'replace')
        pos += length

    return scalars, {}, []


def _decode_navigation_error(payload):
    """ Ported from gsf_dec.c's gsfDecodeNavigationError() (obsolete record). """
    scalars = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['NavErrorTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (record_id,) = struct.unpack_from('>I', payload, 8)
    scalars['RecordID'] = record_id

    (lon_err_raw, lat_err_raw) = struct.unpack_from('>2i', payload, 12)
    scalars['LongitudeError_m'] = lon_err_raw / 10.0
    scalars['LatitudeError_m'] = lat_err_raw / 10.0

    return scalars, {}, []


def _decode_hv_navigation_error(payload):
    """ Ported from gsf_dec.c's gsfDecodeHVNavigationError(). """
    scalars = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['NavErrorTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (record_id,) = struct.unpack_from('>I', payload, 8)
    scalars['RecordID'] = record_id

    (horiz_err_raw, vert_err_raw) = struct.unpack_from('>2i', payload, 12)
    scalars['HorizontalError_m'] = horiz_err_raw / 1000.0
    scalars['VerticalError_m'] = vert_err_raw / 1000.0

    (sep_unc_raw,) = struct.unpack_from('>H', payload, 20)
    scalars['SEPUncertainty_m'] = sep_unc_raw / 100.0

    # 2 spare bytes at 22-23, then the position type string.
    (length,) = struct.unpack_from('>H', payload, 24)
    scalars['PositionType'] = payload[26:26 + length].decode('ascii', 'replace')

    return scalars, {}, []


def _decode_attitude(payload):
    """ Ported from gsf_dec.c's gsfDecodeAttitude(). """
    (base_sec, base_nsec) = struct.unpack_from('>2I', payload, 0)
    (num_measurements,) = struct.unpack_from('>H', payload, 8)

    times = []
    pitch = np.empty(num_measurements)
    roll = np.empty(num_measurements)
    heave = np.empty(num_measurements)
    heading = np.empty(num_measurements)

    pos = 10
    for i in range(num_measurements):
        # time_offset (u16), pitch/roll/heave (s16 each), heading (u16).
        (time_offset_raw, pitch_raw, roll_raw, heave_raw, heading_raw) = \
            struct.unpack_from('>H3hH', payload, pos)
        times.append(_gsf_timestamp(base_sec, base_nsec + int(round((time_offset_raw / 1000.0) * 1e9))))
        pitch[i] = pitch_raw / 100.0
        roll[i] = roll_raw / 100.0
        heave[i] = heave_raw / 100.0
        heading[i] = heading_raw / 100.0
        pos += 10

    table = pd.DataFrame({
        'Time': times,
        'Pitch_deg': pitch,
        'Roll_deg': roll,
        'Heave_m': heave,
        'Heading_deg': heading,
    })
    table.index.name = 'Measurement'

    return {'NumMeasurements': num_measurements}, {'Measurements': table}, []


def _decode_record(record_id, payload, major_version, scale_factors, decode_intensity=False):
    """
    Dispatch to a field-level decoder for the given recordID, if one is
    available.

    :param decode_intensity: forwarded to _decode_swath_bathymetry_ping();
        ignored for every other record type.
    :return: (scalars: dict, tables: dict[str, pandas.DataFrame], notes: list[str]),
        or None if there is no decoder for this record type.
    """
    if record_id == RecordType.GSF_RECORD_HEADER:
        return {'Version': payload[:GSF_VERSION_SIZE].split(b'\x00', 1)[0].decode('ascii', 'replace')}, {}, []
    if record_id == RecordType.GSF_RECORD_SWATH_BATHY_SUMMARY:
        return _decode_swath_bathy_summary(payload)
    if record_id == RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING:
        return _decode_swath_bathymetry_ping(payload, major_version, scale_factors, decode_intensity)
    if record_id == RecordType.GSF_RECORD_SOUND_VELOCITY_PROFILE:
        return _decode_sound_velocity_profile(payload)
    if record_id in (RecordType.GSF_RECORD_PROCESSING_PARAMETERS, RecordType.GSF_RECORD_SENSOR_PARAMETERS):
        return _decode_name_value_parameters(payload)
    if record_id == RecordType.GSF_RECORD_COMMENT:
        return _decode_comment(payload)
    if record_id == RecordType.GSF_RECORD_HISTORY:
        return _decode_history(payload)
    if record_id == RecordType.GSF_RECORD_NAVIGATION_ERROR:
        return _decode_navigation_error(payload)
    if record_id == RecordType.GSF_RECORD_HV_NAVIGATION_ERROR:
        return _decode_hv_navigation_error(payload)
    if record_id == RecordType.GSF_RECORD_SINGLE_BEAM_PING:
        return _decode_single_beam_ping(payload)
    if record_id == RecordType.GSF_RECORD_ATTITUDE:
        return _decode_attitude(payload)
    return None


def _gsf_major_version(version_string, default=3):
    """ Parse the major version number out of a GSF_RECORD_HEADER version
    string (e.g. "GSF-v03.09" -> 3), falling back to `default` if it can't
    be parsed (e.g. no header decoded yet). """
    if version_string:
        try:
            return int(version_string.split('-v', 1)[1].split('.', 1)[0])
        except (IndexError, ValueError):
            pass
    return default


def resolve_record_type(value):
    """
    Resolve a record type given as a RecordType, an int recordID, or a name
    string into a RecordType. Name strings may be short ("COMMENT") or full
    ("GSF_RECORD_COMMENT"), and are matched case-insensitively.

    :param value: None, a RecordType, an int, or a name string.
    :return: None if `value` is None, otherwise the matching RecordType.
    :raises ValueError: `value` is a string that names no known record type.
        The message lists every valid name.
    """
    if value is None or isinstance(value, RecordType):
        return value
    if isinstance(value, int):
        return RecordType(value)

    name = str(value).upper()
    if not name.startswith('GSF_RECORD_'):
        name = 'GSF_RECORD_' + name
    try:
        return RecordType[name]
    except KeyError:
        valid = ", ".join(rt.name for rt in RecordType)
        raise ValueError("Unknown record type: %s\nValid types: %s" % (value, valid))


class gsf():
    """
    A class for indexing, reading, and writing Generic Sensor Format (GSF)
    data files.

    Modeled after the ``kmall`` class in kmall.py: a lightweight,
    dependency-free (no compiled GSF library required) sequential reader
    built on Python's ``struct`` module, plus a pandas-based file index.
    """

    def __init__(self, filename=None):
        self.verbose = 0
        self.filename = filename
        self.FID = None
        self.file_size = None
        self.Index = None

        #: GSF version string read from the file's GSF_RECORD_HEADER record
        #: (e.g. "GSF-v03.09"), set by index_file().
        self.gsfVersion = None

    ###########################################################
    # File open/close utilities (mirrors kmall.py's OpenFiletoRead/closeFile)
    ###########################################################

    def OpenFiletoRead(self, inputfilename=None):
        """ Open a GSF data file for reading. """
        if self.filename is None:
            if inputfilename is None:
                print("No file name specified")
                sys.exit(1)
            else:
                filetoopen = inputfilename
        else:
            filetoopen = self.filename

        if self.verbose >= 1:
            print("Opening: %s to read" % filetoopen)

        self.FID = open(filetoopen, "rb")

    def OpenFiletoWrite(self, inputfilename=None):
        """ Open a GSF data file for writing. """
        if self.filename is None:
            if inputfilename is None:
                print("No file name specified")
                sys.exit(1)
            else:
                filetoopen = inputfilename
        else:
            filetoopen = self.filename

        if self.verbose >= 1:
            print("Opening: %s to write" % filetoopen)

        self.FID = open(filetoopen, "wb")

    def closeFile(self):
        """ Close the file. """
        if self.FID is not None:
            self.FID.close()

    ###########################################################
    # Low level record framing
    ###########################################################

    def read_record_header(self):
        """
        Read the 8-byte record framing (size field + data identifier word)
        for the record at the current file position, leaving the file
        positioned at the start of the record's payload (immediately after
        the data identifier word).

        This reproduces the framing decoded by gsflib's gsfUnpackStream()
        in gsf.c, without decoding the record payload itself.

        :return: None at a clean end of file (no bytes remain to be read),
            otherwise a tuple of (dataSize, readSize, gsfDataID) where:
              dataSize is the value of the on-disk record "size" field --
                the number of bytes in the record payload only. This does
                NOT include the 4-byte checksum word, even when the
                checksum flag is set (per gsf.c's gsfUnpackStream, which
                reads that word separately, ahead of the payload);
              readSize is dataSize, plus 4 additional bytes if the checksum
                flag is set (i.e. the number of bytes remaining to be
                read/skipped, after this framing, to reach the next
                record).
        :raises GSFPartialRecordAtEndOfFileError: fewer than 8 bytes remain,
            or the declared record does not fully fit within the file.
        :raises GSFRecordSizeError: the declared size is <= 8 bytes or
            greater than GSF_MAX_RECORD_SIZE.
        :raises GSFUnrecognizedRecordIDError: recordID is not a value
            between 1 and NUM_REC_TYPES - 1.
        """
        if self.file_size is None:
            self.file_size = os.fstat(self.FID.fileno()).st_size

        offset = self.FID.tell()
        header = self.FID.read(GSF_RECORD_FRAMING_SIZE)

        if len(header) == 0:
            return None

        if len(header) < GSF_RECORD_FRAMING_SIZE:
            raise GSFPartialRecordAtEndOfFileError(
                "Partial record header at byte offset %d in %s" % (offset, self.filename))

        # Read the data size, and GSF ID fields. GSF byte order = network
        # byte order (big-endian).
        dataSize, did = struct.unpack('>II', header)

        # Convert the packed data identifier word into a gsfDataID, per
        # gsf.c's gsfUnpackStream().
        checksumFlag = bool(did & 0x80000000)
        reserved = (did & 0x7FC00000) >> 22
        recordID = did & 0x003FFFFF

        # If there is a checksum, we'll read/skip four additional bytes.
        readSize = dataSize + 4 if checksumFlag else dataSize

        if readSize <= 8 or readSize > GSF_MAX_RECORD_SIZE:
            raise GSFRecordSizeError(
                "Record at byte offset %d in %s has an invalid size (%d bytes)"
                % (offset, self.filename, readSize))

        if recordID < 1 or recordID >= NUM_REC_TYPES:
            raise GSFUnrecognizedRecordIDError(
                "Record at byte offset %d in %s has an unrecognized recordID (%d)"
                % (offset, self.filename, recordID))

        if offset + GSF_RECORD_FRAMING_SIZE + readSize > self.file_size:
            raise GSFPartialRecordAtEndOfFileError(
                "Record at byte offset %d in %s declares %d bytes, which "
                "extends past the end of the file" % (offset, self.filename, readSize))

        data_id = gsfDataID(checksumFlag=checksumFlag, reserved=reserved, recordID=recordID)
        return dataSize, readSize, data_id

    ###########################################################
    # Indexing
    ###########################################################

    def index_file(self):
        """
        Index a GSF file: walk every record in the file, recording its
        type, byte offset, and size, without decoding the record payload.

        Builds self.Index, a pandas DataFrame with one row per record
        (columns: RecordType, RecordID, ByteOffset, RecordSize, TotalBytes,
        ChecksumFlag). Mirrors kmall.py's index_file().
        """
        if self.FID is None:
            self.OpenFiletoRead()
        else:
            self.closeFile()  # forces flushing.
            self.OpenFiletoRead()

        # Get the size of the file.
        self.FID.seek(0, 2)
        self.file_size = self.FID.tell()
        self.FID.seek(0, 0)

        if self.verbose == 1:
            print("Filesize: %d" % self.file_size)

        self.gsfVersion = None

        record_type = []
        record_id = []
        byte_offset = []
        record_size = []
        total_bytes = []
        checksum_flag = []

        while self.FID.tell() < self.file_size:
            offset = self.FID.tell()
            header = self.read_record_header()
            if header is None:
                break
            dataSize, readSize, data_id = header

            # Opportunistically capture the GSF version string out of the
            # header record while we're already positioned at its payload.
            if data_id.recordID == RecordType.GSF_RECORD_HEADER and self.gsfVersion is None:
                version_bytes = self.FID.read(min(GSF_VERSION_SIZE, dataSize))
                self.gsfVersion = version_bytes.split(b'\x00', 1)[0].decode('ascii', 'replace')

            # Skip/seek past the rest of this record's payload, regardless
            # of whether we read part of it above, to land exactly on the
            # next record.
            self.FID.seek(offset + GSF_RECORD_FRAMING_SIZE + readSize, 0)

            byte_offset.append(offset)
            record_id.append(int(data_id.recordID))
            try:
                record_type.append(RecordType(data_id.recordID).name)
            except ValueError:
                record_type.append("UNKNOWN(%d)" % data_id.recordID)
            record_size.append(dataSize)
            total_bytes.append(GSF_RECORD_FRAMING_SIZE + readSize)
            checksum_flag.append(data_id.checksumFlag)

            if self.verbose:
                print("RECORD_TYPE: %s,\tOFFSET: %d,\tSIZE: %d" %
                      (record_type[-1], byte_offset[-1], total_bytes[-1]))

        self.Index = pd.DataFrame({
            'RecordType': record_type,
            'RecordID': record_id,
            'ByteOffset': byte_offset,
            'RecordSize': record_size,
            'TotalBytes': total_bytes,
            'ChecksumFlag': checksum_flag,
        })
        self.Index['RecordType'] = self.Index['RecordType'].astype('category')

        unreadBytes = self.file_size - int(self.Index['TotalBytes'].sum())
        if unreadBytes != 0:
            print()
            print("   *** WARNING! %d bytes were not accounted for in this file! ***" % unreadBytes)
            print()

        if self.verbose >= 2:
            print(self.Index)

    ###########################################################
    # Reporting
    ###########################################################

    def report_record_types(self):
        """
        Print (and return) a summary of the file's index: the number of
        records of each type, and how many bytes they consume. Mirrors
        kmall.py's report_packet_types(), and is invoked by the -V command
        line flag.
        """
        if self.Index is None:
            self.index_file()

        grouped = self.Index.groupby('RecordType', observed=True)
        summary = pd.DataFrame({
            'Count': grouped['RecordType'].count(),
            'Total Bytes': grouped['TotalBytes'].sum(),
            'Min Bytes': grouped['TotalBytes'].min(),
            'Max Bytes': grouped['TotalBytes'].max(),
        })
        summary['% of File'] = (summary['Total Bytes'] / self.file_size * 100).round(2)
        summary = summary[summary['Count'] > 0].sort_values('Total Bytes', ascending=False)

        print("File: %s" % self.filename)
        print("GSF Version: %s" % (self.gsfVersion or "unknown"))
        print("File size: %d bytes" % self.file_size)
        print("Total records: %d" % len(self.Index))
        print()
        print(summary.to_string())

        return summary

    ###########################################################
    # Debugging
    ###########################################################

    def print_records(self, record_type=None):
        """
        Debugging utility: walk the file and print each record. Where a
        field-level decoder is available (see "Field-level record
        decoding" above -- currently every record type except the
        obsolete/rare ones and the beam-array subrecords noted there),
        scalar fields are printed as "key : value" pairs, and any
        per-beam/per-point/per-measurement data is printed as a table,
        one row per beam/point/measurement. Record types with no decoder
        fall back to rendering the raw payload as ASCII text (non-printable
        bytes shown as '.').

        :param record_type: optional record type to restrict output to --
            a RecordType, its integer recordID, or a name string (short,
            e.g. "COMMENT", or full, e.g. "GSF_RECORD_COMMENT";
            case-insensitive). Prints every record if omitted.
        :raises ValueError: record_type is a string that names no known
            record type.
        """
        record_type = resolve_record_type(record_type)

        if self.FID is None:
            self.OpenFiletoRead()
        else:
            self.closeFile()  # forces flushing.
            self.OpenFiletoRead()

        self.FID.seek(0, 2)
        self.file_size = self.FID.tell()
        self.FID.seek(0, 0)

        # GSF_RECORD_SWATH_BATHYMETRY_PING decode needs the file's major
        # version (to know whether the height/SEP/GPS-tide-corrector fields
        # are present) and the most recently seen scale factors (a ping
        # need not repeat them if they haven't changed since an earlier
        # ping) -- both carried across the loop below.
        major_version = _gsf_major_version(self.gsfVersion)
        scale_factors = {}

        while self.FID.tell() < self.file_size:
            offset = self.FID.tell()
            header = self.read_record_header()
            if header is None:
                break
            dataSize, readSize, data_id = header

            is_header_record = data_id.recordID == RecordType.GSF_RECORD_HEADER
            matches = record_type is None or data_id.recordID == record_type

            if matches or is_header_record:
                if data_id.checksumFlag:
                    self.FID.seek(4, 1)
                payload = self.FID.read(dataSize)

                if is_header_record:
                    self.gsfVersion = payload[:GSF_VERSION_SIZE].split(b'\x00', 1)[0].decode('ascii', 'replace')
                    major_version = _gsf_major_version(self.gsfVersion)

                if matches:
                    try:
                        name = RecordType(data_id.recordID).name
                    except ValueError:
                        name = "UNKNOWN(%d)" % data_id.recordID

                    print("=== %s  offset=%d  size=%d ===" %
                          (name, offset, GSF_RECORD_FRAMING_SIZE + readSize))

                    decoded = None
                    try:
                        decoded = _decode_record(data_id.recordID, payload, major_version, scale_factors)
                    except (struct.error, IndexError) as exc:
                        print("  # decode failed (%s); showing raw text" % exc)

                    if decoded is None:
                        text = ''.join(chr(b) if 32 <= b < 127 else '.' for b in payload)
                        print(text)
                    else:
                        scalars, tables, notes = decoded
                        if scalars:
                            width = max(len(k) for k in scalars)
                            for k, v in scalars.items():
                                print("  %-*s : %s" % (width, k, v))
                        for note in notes:
                            print("  # %s" % note)
                        for label, table in tables.items():
                            if table is not None and len(table):
                                print("-- %s --" % label)
                                print(table.to_string())
                    print()

            self.FID.seek(offset + GSF_RECORD_FRAMING_SIZE + readSize, 0)

    def print_intensity_series(self):
        """
        Debugging utility: walk the file's GSF_RECORD_SWATH_BATHYMETRY_PING
        records and print each ping's per-beam backscatter time series (the
        GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord, decoded
        by _decode_brb_intensity()) to stdout as CSV: one row per beam,
        giving the beam index, sample count, bottom-detect sample index,
        start-range sample index, then every raw sample value in that
        beam's time series.

        Currently only the KMALL (Kongsberg SIS 5 / EM2040-and-newer)
        sensor-imagery format is decoded; pings from other sensors, or that
        carry no intensity series subrecord at all, are noted and skipped
        rather than guessed at.
        """
        if self.FID is None:
            self.OpenFiletoRead()
        else:
            self.closeFile()  # forces flushing.
            self.OpenFiletoRead()

        self.FID.seek(0, 2)
        self.file_size = self.FID.tell()
        self.FID.seek(0, 0)

        major_version = _gsf_major_version(self.gsfVersion)
        scale_factors = {}

        while self.FID.tell() < self.file_size:
            offset = self.FID.tell()
            header = self.read_record_header()
            if header is None:
                break
            dataSize, readSize, data_id = header

            is_header_record = data_id.recordID == RecordType.GSF_RECORD_HEADER
            is_ping = data_id.recordID == RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING

            if is_header_record or is_ping:
                if data_id.checksumFlag:
                    self.FID.seek(4, 1)
                payload = self.FID.read(dataSize)

                if is_header_record:
                    self.gsfVersion = payload[:GSF_VERSION_SIZE].split(b'\x00', 1)[0].decode('ascii', 'replace')
                    major_version = _gsf_major_version(self.gsfVersion)

                if is_ping:
                    try:
                        scalars, tables, notes = _decode_swath_bathymetry_ping(
                            payload, major_version, scale_factors, decode_intensity=True)
                    except (struct.error, IndexError) as exc:
                        print("# ping offset=%d: decode failed (%s)" % (offset, exc))
                    else:
                        series = tables.get('IntensityTimeSeries')
                        if series is None:
                            note = next((n for n in notes if 'IntensityTimeSeries' in n), None)
                            print("# ping offset=%d ping_time=%s: %s" %
                                  (offset, scalars.get('PingTime', '?'),
                                   note or "no intensity time series subrecord"))
                        else:
                            print("# ping offset=%d ping_time=%s" % (offset, scalars.get('PingTime', '?')))
                            print("# Beam,SampleCount,DetectSample,StartRangeSamples,Sample0,Sample1,...")
                            for beam, row in series.iterrows():
                                fields = [str(beam), str(row['SampleCount']), str(row['DetectSample']),
                                          str(row['StartRangeSamples'])]
                                fields.extend(str(v) for v in row['Samples'])
                                print(",".join(fields))

            self.FID.seek(offset + GSF_RECORD_FRAMING_SIZE + readSize, 0)


###########################################################
# Command line interface
###########################################################

def _record_type_help_text():
    """
    Build a reference listing of every GSF_RECORD_* type and its
    description (from RECORD_TYPE_DESCRIPTIONS), for use as the -p
    argument's record type reference and printed via -h.
    """
    lines = ["record types (short or full name accepted for -p, e.g. -p COMMENT):", ""]
    short_names = [rt.name.replace('GSF_RECORD_', '') for rt in RecordType]
    width = max(len(name) for name in short_names)
    for rt, short in zip(RecordType, short_names):
        lines.append("  %-*s  %s" % (width, short, RECORD_TYPE_DESCRIPTIONS.get(rt, "")))
    return "\n".join(lines)


def main(args=None):
    """ Command line script entry point. """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="A python script (and class) for indexing, reading, "
                     "and writing Generic Sensor Format (GSF) data files.",
        epilog=_record_type_help_text(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', action='store', dest='gsf_filename',
                         help="The path and filename to parse.")
    parser.add_argument('-V', action='store_true', dest='verify',
                         default=False,
                         help="Index the file and print a summary of its record types "
                              "(count and bytes consumed by each).")
    parser.add_argument('-p', action='store', nargs='?', const='ALL', default=None,
                         dest='print_records', metavar='RECORDTYPE',
                         help="Print records to stdout as ASCII text, for debugging. "
                              "With no value, prints every record; optionally restrict "
                              "to one record type, e.g. -p COMMENT or -p GSF_RECORD_COMMENT.")
    parser.add_argument('-I', action='store_true', dest='print_intensity',
                         default=False,
                         help="Print each swath bathymetry ping's per-beam backscatter "
                              "time series (the intensity series subrecord) to stdout as "
                              "CSV, one row per beam. Currently decoded only for the KMALL "
                              "(Kongsberg SIS 5) sensor-imagery format.")
    parser.add_argument('-v', action='count', dest='verbose', default=0,
                         help="Increasingly verbose output (e.g. -v -vv), for debugging use -vv")
    parsed = parser.parse_args(args)

    if parsed.gsf_filename is None:
        parser.print_help()
        return 1

    G = gsf(parsed.gsf_filename)
    G.verbose = parsed.verbose

    if parsed.print_intensity:
        G.print_intensity_series()
        return 0

    if parsed.print_records is not None:
        record_type = None if parsed.print_records == 'ALL' else parsed.print_records
        try:
            G.print_records(record_type=record_type)
        except ValueError as exc:
            print(exc)
            return 1
        return 0

    G.index_file()

    if parsed.verify:
        G.report_record_types()
    else:
        print("Indexed %d records in %s" % (len(G.Index), parsed.gsf_filename))

    return 0


if __name__ == '__main__':
    sys.exit(main())
