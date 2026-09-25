#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A python class (and command line utility) to index, read, and write Generic
Sensor Format (GSF) sonar data files.

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
# Every per-sensor ping-level "_SPECIFIC" subrecord (all 55 of gsf.h's
# GSF_SWATH_BATHY_SUBRECORD_*_SPECIFIC ids) is decoded via the
# _PING_SENSOR_SPECIFIC_CODECS registry (KMALL_SPECIFIC, id 156, through a
# thin adapter over its own standalone _decode_kmall_specific()/
# _encode_kmall_specific() -- see the registry's docstring below).
# Likewise every single-beam
# sensor-specific tail (_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS) and the
# per-beam intensity-series imagery preamble (_decode_brb_intensity() and
# its per-family _decode_*_imagery_specific() helpers) are fully decoded.
#
# Deliberately NOT decoded (reported as a byte count instead): gsflib's
# own optional RLE array compression (DecodeCompressedArray).
###########################################################

#: Identifies the scale factors subrecord within a ping's subrecord stream.
#: (gsf.h: GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS)
_SUBRECORD_SCALE_FACTORS = 100
#: (gsf.h: GSF_SWATH_BATHY_SUBRECORD_BEAM_FLAGS_ARRAY) -- raw bytes, no scale factor.
_SUBRECORD_BEAM_FLAGS_ARRAY = 16
#: (gsf.h: GSF_SWATH_BATHY_SUBRECORD_QUALITY_FLAGS_ARRAY) -- 2-bit packed, no scale factor.
_SUBRECORD_QUALITY_FLAGS_ARRAY = 15

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
    # 15 GSF_SWATH_BATHY_SUBRECORD_QUALITY_FLAGS_ARRAY: handled separately, 2-bit
    #    packed (no scale factor) -- see _decode_quality_flags_array().
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
#: from gsf.h's GSF_SWATH_BATHY_SUBRECORD_* defines. Used to resolve a
#: decoded ping's SensorSpecificID to a human-readable family name (e.g.
#: for print_records()), and to label subrecords not present in
#: _PING_SENSOR_SPECIFIC_CODECS (id 154 is unused/reserved in gsf.h).
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

#: Registry of ping-level sensor-specific ("_SPECIFIC") subrecord codecs,
#: keyed by gsf.h's GSF_SWATH_BATHY_SUBRECORD_* id: {subrecord_id: (family
#: label, decode_fn, encode_fn)}. Several ids can share one struct/codec
#: pair (e.g. gsf.h's t_gsfEM3Specific -- and so gsfu.py's
#: _decode_em3_specific()/_encode_em3_specific() -- covers EM3000, EM1002,
#: EM300, EM120, EM3002, EM3000D, EM3002D, EM121A_SIS, and EM2000).
#:
#: decode_fn(payload, pos) -> (fields: dict, tables: dict[str,
#: pandas.DataFrame], bytes_consumed). encode_fn(subrecord_id, fields,
#: tables=None) -> bytes, including its own 4-byte subrecord id+size word
#: (subrecord_id is passed through since one encode_fn may need to stamp
#: any of several ids). Table values are pandas.DataFrames on both sides --
#: a decode_fn's returned DataFrame can be handed straight to the matching
#: encode_fn's `tables` with no conversion. Almost every family always
#: returns an empty `tables` dict; only EM4, EM3Raw, EM3, and KMALL ever
#: populate one.
#:
#: KMALL_SPECIFIC (id 156) is registered via a thin adapter pair
#: (_decode_kmall_specific_adapter()/_encode_kmall_specific_adapter(), just
#: below _encode_kmall_specific()) that repackages its own standalone
#: _decode_kmall_specific()/_encode_kmall_specific() functions -- which
#: predate this registry and keep their own two-named-table signature --
#: into this registry's generic contract.
_PING_SENSOR_SPECIFIC_CODECS = {}


def _decode_elac_mkii_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_ELAC_MKII_SPECIFIC subrecord (id
    117): Elac MkII multibeam sensor metadata. Ported from gsf_dec.c's
    DecodeElacMkIISpecific().

    Untested against a verified GSF file: no sample data containing an
    ELAC_MKII_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    mode = payload[pos]; pos += 1
    (ping_num,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sound_vel,) = struct.unpack_from('>H', payload, pos); pos += 2
    (pulse_length,) = struct.unpack_from('>H', payload, pos); pos += 2
    receiver_gain_stbd = payload[pos]; pos += 1
    receiver_gain_port = payload[pos]; pos += 1
    (reserved,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {
        'Mode': mode,
        'PingNumber': ping_num,
        'SoundVelocity_mps': sound_vel,
        'PulseLength_hundredth_ms': pulse_length,
        'ReceiverGainStbd_dB': receiver_gain_stbd,
        'ReceiverGainPort_dB': receiver_gain_port,
        'Reserved': reserved,
    }
    return fields, {}, pos - start


def _encode_elac_mkii_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_ELAC_MKII_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_elac_mkii_specific(). Ported from gsf_enc.c's
    EncodeElacMkIISpecific().
    """
    g = fields.get
    body = struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', int(g('SoundVelocity_mps', 0)))
    body += struct.pack('>H', int(g('PulseLength_hundredth_ms', 0)))
    body += struct.pack('>B', int(g('ReceiverGainStbd_dB', 0)) & 0xFF)
    body += struct.pack('>B', int(g('ReceiverGainPort_dB', 0)) & 0xFF)
    body += struct.pack('>H', int(g('Reserved', 0)))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[117] = ("ElacMkII", _decode_elac_mkii_specific, _encode_elac_mkii_specific)


def _decode_seabeam_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_SPECIFIC subrecord (id
    102): 16-beam SeaBeam sensor metadata. Ported from gsf_dec.c's
    DecodeSeabeamSpecific().

    Untested against a verified GSF file: no sample data containing a
    SEABEAM_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (eclipse_time,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {'EclipseTime_tenths_s': eclipse_time}
    return fields, {}, pos - start


def _encode_seabeam_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_seabeam_specific(). Ported from gsf_enc.c's
    EncodeSeabeamSpecific().
    """
    body = struct.pack('>H', int(fields.get('EclipseTime_tenths_s', 0)))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[102] = ("SeaBeam", _decode_seabeam_specific, _encode_seabeam_specific)


def _decode_em12_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM12_SPECIFIC subrecord (id 103):
    Simrad EM12 sensor metadata. Ported from gsf_dec.c's
    DecodeEM12Specific().

    Untested against a verified GSF file: no sample data containing an
    EM12_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    resolution = payload[pos]; pos += 1
    ping_quality = payload[pos]; pos += 1
    (sound_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    pos += 32  # spare

    fields = {
        'PingNumber': ping_number,
        'Resolution': resolution,
        'PingQuality': ping_quality,
        'SoundVelocity_mps': sound_velocity_raw / 10.0,
        'Mode': mode,
    }
    return fields, {}, pos - start


def _encode_em12_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM12_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_em12_specific(). Ported from gsf_enc.c's EncodeEM12Specific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>B', int(g('Resolution', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PingQuality', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('SoundVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += b'\x00' * 32  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[103] = ("EM12", _decode_em12_specific, _encode_em12_specific)


def _decode_em100_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM100_SPECIFIC subrecord (id 104):
    Simrad EM100 sensor metadata. Ported from gsf_dec.c's
    DecodeEM100Specific().

    Untested against a verified GSF file: no sample data containing an
    EM100_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (ship_pitch_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (transducer_pitch_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    power = payload[pos]; pos += 1
    attenuation = payload[pos]; pos += 1
    tvg = payload[pos]; pos += 1
    pulse_length = payload[pos]; pos += 1
    (counter,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {
        'ShipPitch_deg': ship_pitch_raw / 100.0,
        'TransducerPitch_deg': transducer_pitch_raw / 100.0,
        'Mode': mode,
        'Power': power,
        'Attenuation': attenuation,
        'TVG': tvg,
        'PulseLength': pulse_length,
        'Counter': counter,
    }
    return fields, {}, pos - start


def _encode_em100_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM100_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_em100_specific(). Ported from gsf_enc.c's
    EncodeEM100Specific().
    """
    g = fields.get
    body = struct.pack('>h', _gsf_round(g('ShipPitch_deg', 0.0) * 100.0))
    body += struct.pack('>h', _gsf_round(g('TransducerPitch_deg', 0.0) * 100.0))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Power', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Attenuation', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TVG', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PulseLength', 0)) & 0xFF)
    body += struct.pack('>H', int(g('Counter', 0)))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[104] = ("EM100", _decode_em100_specific, _encode_em100_specific)


def _decode_cmp_sass_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_CMP_SASS_SPECIFIC subrecord (id
    121): Compressed SASS (BOSDAT) sensor metadata. Ported from
    gsf_dec.c's DecodeCmpSassSpecific().

    Untested against a verified GSF file: no sample data containing a
    CMP_SASS_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (lfreq_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (lntens_raw,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {
        'SurfaceSoundVelocity_ftps': lfreq_raw / 10.0,
        'Heave_ftps': lntens_raw / 10.0,
    }
    return fields, {}, pos - start


def _encode_cmp_sass_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_CMP_SASS_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_cmp_sass_specific(). Ported from gsf_enc.c's
    EncodeCmpSassSpecific().
    """
    g = fields.get
    body = struct.pack('>H', _gsf_round(g('SurfaceSoundVelocity_ftps', 0.0) * 10.0))
    body += struct.pack('>H', _gsf_round(g('Heave_ftps', 0.0) * 10.0))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[121] = ("CmpSass", _decode_cmp_sass_specific, _encode_cmp_sass_specific)


def _decode_em950_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM950_SPECIFIC or
    _EM1000_SPECIFIC subrecord (ids 105/111 -- identical wire format, one
    struct/decoder shared by both in gsflib). Ported from gsf_dec.c's
    DecodeEM950Specific().

    Untested against a verified GSF file: no sample data containing an
    EM950_SPECIFIC/EM1000_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    (ping_quality,) = struct.unpack_from('>b', payload, pos); pos += 1
    (ship_pitch_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (transducer_pitch_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {
        'PingNumber': ping_number,
        'Mode': mode,
        'PingQuality': ping_quality,
        'ShipPitch_deg': ship_pitch_raw / 100.0,
        'TransducerPitch_deg': transducer_pitch_raw / 100.0,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
    }
    return fields, {}, pos - start


def _encode_em950_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM950_SPECIFIC or _EM1000_SPECIFIC
    subrecord (subrecord_id selects which -- see _decode_em950_specific()),
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_em950_specific(). Ported from gsf_enc.c's EncodeEM950Specific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>b', int(g('PingQuality', 0)))
    body += struct.pack('>h', _gsf_round(g('ShipPitch_deg', 0.0) * 100.0))
    body += struct.pack('>h', _gsf_round(g('TransducerPitch_deg', 0.0) * 100.0))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[105] = ("EM950", _decode_em950_specific, _encode_em950_specific)
_PING_SENSOR_SPECIFIC_CODECS[111] = ("EM1000", _decode_em950_specific, _encode_em950_specific)


def _decode_em121a_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM121A_SPECIFIC or _EM121_SPECIFIC
    subrecord (ids 106/107 -- identical wire format, one struct/decoder
    shared by both in gsflib). Ported from gsf_dec.c's
    DecodeEM121ASpecific().

    Untested against a verified GSF file: no sample data containing an
    EM121A_SPECIFIC/EM121_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    valid_beams = payload[pos]; pos += 1
    pulse_length = payload[pos]; pos += 1
    beam_width = payload[pos]; pos += 1
    tx_power = payload[pos]; pos += 1
    tx_status = payload[pos]; pos += 1
    rx_status = payload[pos]; pos += 1
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {
        'PingNumber': ping_number,
        'Mode': mode,
        'ValidBeams': valid_beams,
        'PulseLength': pulse_length,
        'BeamWidth': beam_width,
        'TxPower': tx_power,
        'TxStatus': tx_status,
        'RxStatus': rx_status,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
    }
    return fields, {}, pos - start


def _encode_em121a_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM121A_SPECIFIC or _EM121_SPECIFIC
    subrecord (subrecord_id selects which -- see _decode_em121a_specific()),
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_em121a_specific(). Ported from gsf_enc.c's
    EncodeEM121ASpecific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>B', int(g('ValidBeams', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PulseLength', 0)) & 0xFF)
    body += struct.pack('>B', int(g('BeamWidth', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TxPower', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TxStatus', 0)) & 0xFF)
    body += struct.pack('>B', int(g('RxStatus', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[106] = ("EM121A", _decode_em121a_specific, _encode_em121a_specific)
_PING_SENSOR_SPECIFIC_CODECS[107] = ("EM121", _decode_em121a_specific, _encode_em121a_specific)


def _decode_seamap_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEAMAP_SPECIFIC subrecord (id 109):
    SeaMap/SeaMap-II swath interferometric sonar metadata. Ported from
    gsf_dec.c's DecodeSeaMapSpecific().

    gsf_dec.c's DecodeSeaMapSpecific() takes an extra GSF_FILE_TABLE *ft
    argument used only to gate one historical bug fix: for GSF files
    written by gsflib older than v2.7, the pointer advance after
    pressureDepth was missing (a documented bug -- see the "JSB
    11/08/2007" comment in gsf_dec.c/gsf_enc.c), so pressureDepth's raw
    bytes were immediately overwritten by altitude's decode on those old
    files. This library doesn't track per-file gsflib-version state the
    way gsflib's GSF_FILE_TABLE does, and only ever writes modern
    (GSF_VERSION, > v2.7) files, so this always takes the "fixed" branch
    (advances past pressureDepth normally) -- correct for any file this
    library itself writes, and for any real file from a non-ancient
    gsflib, but not bug-for-bug compatible with a SeaMap subrecord from a
    GSF file older than v2.7.

    Untested against a verified GSF file: no sample data containing a
    SEAMAP_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    # 11 raw u16 words, file order: portTransmitter[0,1], stbdTransmitter[0,1],
    # portGain, stbdGain, portPulseLength, stbdPulseLength, pressureDepth,
    # altitude, temperature.
    values = struct.unpack_from('>11H', payload, pos)
    pos += 22

    keys = (
        'PortTransmitter0', 'PortTransmitter1', 'StbdTransmitter0', 'StbdTransmitter1',
        'PortGain', 'StbdGain', 'PortPulseLength', 'StbdPulseLength',
        'PressureDepth', 'Altitude', 'Temperature',
    )
    fields = {key: value / 10.0 for key, value in zip(keys, values)}
    return fields, {}, pos - start


def _encode_seamap_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEAMAP_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_seamap_specific() (see its docstring re: the pre-v2.7
    pressureDepth quirk this doesn't replicate). Ported from gsf_enc.c's
    EncodeSeaMapSpecific().
    """
    g = fields.get
    keys = (
        'PortTransmitter0', 'PortTransmitter1', 'StbdTransmitter0', 'StbdTransmitter1',
        'PortGain', 'StbdGain', 'PortPulseLength', 'StbdPulseLength',
        'PressureDepth', 'Altitude', 'Temperature',
    )
    body = b''.join(struct.pack('>H', _gsf_round(g(key, 0.0) * 10.0)) for key in keys)
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[109] = ("SeaMap", _decode_seamap_specific, _encode_seamap_specific)


def _decode_seabat_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_SPECIFIC subrecord (id 110):
    Reson SeaBat sensor metadata. Ported from gsf_dec.c's
    DecodeSeaBatSpecific().

    Untested against a verified GSF file: no sample data containing a
    SEABAT_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    sonar_range = payload[pos]; pos += 1
    transmit_power = payload[pos]; pos += 1
    receive_gain = payload[pos]; pos += 1

    fields = {
        'PingNumber': ping_number,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
        'Mode': mode,
        'SonarRange_m': sonar_range,
        'TransmitPower': transmit_power,
        'ReceiveGain': receive_gain,
    }
    return fields, {}, pos - start


def _encode_seabat_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_seabat_specific(). Ported from gsf_enc.c's EncodeSeaBatSpecific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>B', int(g('SonarRange_m', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TransmitPower', 0)) & 0xFF)
    body += struct.pack('>B', int(g('ReceiveGain', 0)) & 0xFF)
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[110] = ("SeaBat", _decode_seabat_specific, _encode_seabat_specific)


def _decode_sb_amp_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SB_AMP_SPECIFIC subrecord (id 113):
    SeaBeam amplitude sensor metadata. Ported from gsf_dec.c's
    DecodeSBAmpSpecific().

    Untested against a verified GSF file: no sample data containing a
    SB_AMP_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    hour = payload[pos]; pos += 1
    minute = payload[pos]; pos += 1
    second = payload[pos]; pos += 1
    hundredths = payload[pos]; pos += 1
    (block_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (avg_gate_depth,) = struct.unpack_from('>h', payload, pos); pos += 2

    fields = {
        'Hour': hour,
        'Minute': minute,
        'Second': second,
        'Hundredths': hundredths,
        'BlockNumber': block_number,
        'AvgGateDepth': avg_gate_depth,
    }
    return fields, {}, pos - start


def _encode_sb_amp_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SB_AMP_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_sb_amp_specific(). Ported from gsf_enc.c's EncodeSBAmpSpecific().
    """
    g = fields.get
    body = struct.pack('>B', int(g('Hour', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Minute', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Second', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Hundredths', 0)) & 0xFF)
    body += struct.pack('>I', int(g('BlockNumber', 0)))
    body += struct.pack('>h', int(g('AvgGateDepth', 0)))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[113] = ("SBAmp", _decode_sb_amp_specific, _encode_sb_amp_specific)


def _decode_seabat_ii_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_II_SPECIFIC subrecord (id
    114): Reson SeaBat II sensor metadata (replaces SEABAT_SPECIFIC as of
    GSF_1.04). Ported from gsf_dec.c's DecodeSeaBatIISpecific().

    Untested against a verified GSF file: no sample data containing a
    SEABAT_II_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sonar_range,) = struct.unpack_from('>H', payload, pos); pos += 2
    (transmit_power,) = struct.unpack_from('>H', payload, pos); pos += 2
    (receive_gain,) = struct.unpack_from('>H', payload, pos); pos += 2
    fore_aft_bw = payload[pos]; pos += 1
    athwart_bw = payload[pos]; pos += 1
    pos += 4  # spare

    fields = {
        'PingNumber': ping_number,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
        'Mode': mode,
        'SonarRange_m': sonar_range,
        'TransmitPower': transmit_power,
        'ReceiveGain': receive_gain,
        'ForeAftBW_deg': fore_aft_bw / 10.0,
        'AthwartBW_deg': athwart_bw / 10.0,
    }
    return fields, {}, pos - start


def _encode_seabat_ii_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_II_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_seabat_ii_specific(). Ported from gsf_enc.c's
    EncodeSeaBatIISpecific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>H', int(g('Mode', 0)))
    body += struct.pack('>H', int(g('SonarRange_m', 0)))
    body += struct.pack('>H', int(g('TransmitPower', 0)))
    body += struct.pack('>H', int(g('ReceiveGain', 0)))
    body += struct.pack('>B', _gsf_round(g('ForeAftBW_deg', 0.0) * 10.0) & 0xFF)
    body += struct.pack('>B', _gsf_round(g('AthwartBW_deg', 0.0) * 10.0) & 0xFF)
    body += b'\x00' * 4  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[114] = ("SeaBatII", _decode_seabat_ii_specific, _encode_seabat_ii_specific)


def _decode_seabeam_2112_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_2112_SPECIFIC subrecord (id
    116): SeaBeam 2112/36 sensor metadata. Ported from gsf_dec.c's
    DecodeSeaBeam2112Specific(). `Mode` is a raw bitmask (see gsf.h's
    GSF_2112_* macros just after t_gsfSeaBeam2112Specific) -- not split
    into separate booleans here, matching how this library treats other
    bitmask fields (e.g. KMALL's PingFlags).

    Untested against a verified GSF file: no sample data containing a
    SEABEAM_2112_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    mode = payload[pos]; pos += 1
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    ssv_source = payload[pos]; pos += 1
    ping_gain = payload[pos]; pos += 1
    pulse_width = payload[pos]; pos += 1
    transmitter_attenuation = payload[pos]; pos += 1
    number_algorithms = payload[pos]; pos += 1
    algorithm_order = payload[pos:pos + 4].decode('ascii', 'replace').rstrip('\x00'); pos += 4
    pos += 2  # spare

    fields = {
        'Mode': mode,
        'SurfaceVelocity_mps': (surface_velocity_raw + 130000) / 100.0,
        'SsvSource': ssv_source,
        'PingGain_dB': ping_gain,
        'PulseWidth_ms': pulse_width,
        'TransmitterAttenuation_dB': transmitter_attenuation,
        'NumberAlgorithms': number_algorithms,
        'AlgorithmOrder': algorithm_order,
    }
    return fields, {}, pos - start


def _encode_seabeam_2112_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_2112_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_seabeam_2112_specific(). Ported from gsf_enc.c's
    EncodeSeaBeam2112Specific().
    """
    g = fields.get
    body = struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 1300.0) * 100.0 - 130000))
    body += struct.pack('>B', int(g('SsvSource', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PingGain_dB', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PulseWidth_ms', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TransmitterAttenuation_dB', 0)) & 0xFF)
    body += struct.pack('>B', int(g('NumberAlgorithms', 0)) & 0xFF)
    body += g('AlgorithmOrder', '').encode('ascii')[:4].ljust(4, b'\x00')
    body += b'\x00' * 2  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[116] = ("SeaBeam2112", _decode_seabeam_2112_specific, _encode_seabeam_2112_specific)


def _decode_seabat8101_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_8101_SPECIFIC subrecord (id
    115). Ported from gsf_dec.c's DecodeSeaBat8101Specific().

    Untested against a verified GSF file: no sample data containing a
    SEABAT_8101_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (rng,) = struct.unpack_from('>H', payload, pos); pos += 2
    (power,) = struct.unpack_from('>H', payload, pos); pos += 2
    (gain,) = struct.unpack_from('>H', payload, pos); pos += 2
    (pulse_width,) = struct.unpack_from('>H', payload, pos); pos += 2
    tvg_spreading = payload[pos]; pos += 1
    tvg_absorption = payload[pos]; pos += 1
    fore_aft_bw_raw = payload[pos]; pos += 1
    athwart_bw_raw = payload[pos]; pos += 1
    (range_filt_min,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_filt_max,) = struct.unpack_from('>H', payload, pos); pos += 2
    (depth_filt_min,) = struct.unpack_from('>H', payload, pos); pos += 2
    (depth_filt_max,) = struct.unpack_from('>H', payload, pos); pos += 2
    projector = payload[pos]; pos += 1
    pos += 4  # spare

    fields = {
        'PingNumber': ping_number,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
        'Mode': mode,
        'Range_m': rng,
        'Power': power,
        'Gain': gain,
        'PulseWidth_us': pulse_width,
        'TvgSpreading': tvg_spreading,
        'TvgAbsorption': tvg_absorption,
        'ForeAftBW_deg': fore_aft_bw_raw / 10.0,
        'AthwartBW_deg': athwart_bw_raw / 10.0,
        'RangeFiltMin': range_filt_min,
        'RangeFiltMax': range_filt_max,
        'DepthFiltMin': depth_filt_min,
        'DepthFiltMax': depth_filt_max,
        'Projector': projector,
    }
    return fields, {}, pos - start


def _encode_seabat8101_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_8101_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_seabat8101_specific(). Ported from gsf_enc.c's
    EncodeSeaBat8101Specific(). The reference encoder rounds
    fore_aft_bw/athwart_bw with a plain `+ 0.5` truncating cast rather
    than the +/-0.501 convention used elsewhere; functionally equivalent
    since these fields are never negative, so this uses the standard
    _gsf_round() convention for consistency with the rest of this module.
    """
    g = fields.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>H', int(g('Mode', 0)))
    body += struct.pack('>H', int(g('Range_m', 0)))
    body += struct.pack('>H', int(g('Power', 0)))
    body += struct.pack('>H', int(g('Gain', 0)))
    body += struct.pack('>H', int(g('PulseWidth_us', 0)))
    body += struct.pack('>B', int(g('TvgSpreading', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TvgAbsorption', 0)) & 0xFF)
    body += struct.pack('>B', _gsf_round(g('ForeAftBW_deg', 0.0) * 10.0) & 0xFF)
    body += struct.pack('>B', _gsf_round(g('AthwartBW_deg', 0.0) * 10.0) & 0xFF)
    body += struct.pack('>H', int(g('RangeFiltMin', 0)))
    body += struct.pack('>H', int(g('RangeFiltMax', 0)))
    body += struct.pack('>H', int(g('DepthFiltMin', 0)))
    body += struct.pack('>H', int(g('DepthFiltMax', 0)))
    body += struct.pack('>B', int(g('Projector', 0)) & 0xFF)
    body += b'\x00' * 4  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[115] = ("SeaBat8101", _decode_seabat8101_specific, _encode_seabat8101_specific)


def _decode_reson8100_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_RESON_8101/8111/8124/8125/8150/8160_
    SPECIFIC subrecord (ids 122-127 -- one struct/decoder shared by the
    whole Reson 8100 family). Ported from gsf_dec.c's
    DecodeReson8100Specific().

    Note: ProjectorAngle is kept as the raw on-disk "degrees * 100" integer,
    NOT divided down to degrees -- gsf_dec.c itself stores it that way (the
    struct field is a plain `int`, unlike surface_velocity/fore_aft_bw/
    etc., which the decoder does descale into `double` fields), despite the
    comment there describing the encoding. Preserved as-is for fidelity.

    Untested against a verified GSF file: no sample data containing a
    Reson 8100-family SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (latency,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sonar_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sonar_model,) = struct.unpack_from('>H', payload, pos); pos += 2
    (frequency,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sample_rate,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_rate,) = struct.unpack_from('>H', payload, pos); pos += 2
    (mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (rng,) = struct.unpack_from('>H', payload, pos); pos += 2
    (power,) = struct.unpack_from('>H', payload, pos); pos += 2
    (gain,) = struct.unpack_from('>H', payload, pos); pos += 2
    (pulse_width,) = struct.unpack_from('>H', payload, pos); pos += 2
    tvg_spreading = payload[pos]; pos += 1
    tvg_absorption = payload[pos]; pos += 1
    fore_aft_bw_raw = payload[pos]; pos += 1
    athwart_bw_raw = payload[pos]; pos += 1
    projector_type = payload[pos]; pos += 1
    (projector_angle,) = struct.unpack_from('>h', payload, pos); pos += 2
    (range_filt_min,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_filt_max,) = struct.unpack_from('>H', payload, pos); pos += 2
    (depth_filt_min,) = struct.unpack_from('>H', payload, pos); pos += 2
    (depth_filt_max,) = struct.unpack_from('>H', payload, pos); pos += 2
    filters_active = payload[pos]; pos += 1
    (temperature,) = struct.unpack_from('>H', payload, pos); pos += 2
    (beam_spacing_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    pos += 2  # spare

    fields = {
        'Latency_ms': latency,
        'PingNumber': ping_number,
        'SonarID': sonar_id,
        'SonarModel': sonar_model,
        'Frequency_kHz': frequency,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
        'SampleRate_Hz': sample_rate,
        'PingRate_mHz': ping_rate,
        'Mode': mode,
        'Range_m': rng,
        'Power': power,
        'Gain': gain,
        'PulseWidth_us': pulse_width,
        'TvgSpreading': tvg_spreading,
        'TvgAbsorption': tvg_absorption,
        'ForeAftBW_deg': fore_aft_bw_raw / 10.0,
        'AthwartBW_deg': athwart_bw_raw / 10.0,
        'ProjectorType': projector_type,
        'ProjectorAngle': projector_angle,
        'RangeFiltMin': range_filt_min,
        'RangeFiltMax': range_filt_max,
        'DepthFiltMin': depth_filt_min,
        'DepthFiltMax': depth_filt_max,
        'FiltersActive': filters_active,
        'Temperature_tenth_degC': temperature,
        'BeamSpacing_deg': beam_spacing_raw / 10000.0,
    }
    return fields, {}, pos - start


def _encode_reson8100_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_RESON_8101/8111/8124/8125/8150/8160_
    SPECIFIC subrecord, including its own 4-byte subrecord id+size word:
    the inverse of _decode_reson8100_specific(). Ported from gsf_enc.c's
    EncodeReson8100Specific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('Latency_ms', 0)))
    body += struct.pack('>I', int(g('PingNumber', 0)))
    body += struct.pack('>I', int(g('SonarID', 0)))
    body += struct.pack('>H', int(g('SonarModel', 0)))
    body += struct.pack('>H', int(g('Frequency_kHz', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>H', int(g('SampleRate_Hz', 0)))
    body += struct.pack('>H', int(g('PingRate_mHz', 0)))
    body += struct.pack('>H', int(g('Mode', 0)))
    body += struct.pack('>H', int(g('Range_m', 0)))
    body += struct.pack('>H', int(g('Power', 0)))
    body += struct.pack('>H', int(g('Gain', 0)))
    body += struct.pack('>H', int(g('PulseWidth_us', 0)))
    body += struct.pack('>B', int(g('TvgSpreading', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TvgAbsorption', 0)) & 0xFF)
    body += struct.pack('>B', _gsf_round(g('ForeAftBW_deg', 0.0) * 10.0) & 0xFF)
    body += struct.pack('>B', _gsf_round(g('AthwartBW_deg', 0.0) * 10.0) & 0xFF)
    body += struct.pack('>B', int(g('ProjectorType', 0)) & 0xFF)
    body += struct.pack('>h', int(g('ProjectorAngle', 0)))
    body += struct.pack('>H', int(g('RangeFiltMin', 0)))
    body += struct.pack('>H', int(g('RangeFiltMax', 0)))
    body += struct.pack('>H', int(g('DepthFiltMin', 0)))
    body += struct.pack('>H', int(g('DepthFiltMax', 0)))
    body += struct.pack('>B', int(g('FiltersActive', 0)) & 0xFF)
    body += struct.pack('>H', int(g('Temperature_tenth_degC', 0)))
    body += struct.pack('>H', _gsf_round(g('BeamSpacing_deg', 0.0) * 10000.0))
    body += b'\x00' * 2  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


for _id in (122, 123, 124, 125, 126, 127):
    _PING_SENSOR_SPECIFIC_CODECS[_id] = ("Reson8100", _decode_reson8100_specific, _encode_reson8100_specific)
del _id


def _decode_reson7125_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_RESON_7125_SPECIFIC subrecord (id
    138). Ported from gsf_dec.c's DecodeReson7100Specific(). Bitmask
    fields (ControlFlags, TransmitFlags, ReceiveFlags) are kept as a
    single raw int, not split into booleans, matching this codebase's
    convention for similar fields elsewhere (e.g. SeaBeam2112's Mode).

    Note: gsf_dec.c's own DecodeReson7100Specific() has a latent bug for
    tx_pulse_reserved -- it reads from a stale `stemp` (a leftover 2-byte
    local from an earlier field) instead of the 4-byte `ltemp` it just
    loaded, so the real reference decoder misreads this field. Since the
    field is documented as reserved/unused, this port decodes it
    correctly (the actual 4 bytes at that wire position) rather than
    replicating the bug -- there's no meaningful data to lose either way.

    Untested against a verified GSF file: no sample data containing a
    RESON_7125_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (no nested arrays), bytes_consumed).
    """
    start = pos
    (protocol_version,) = struct.unpack_from('>H', payload, pos); pos += 2
    (device_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    pos += 16  # reserved_1
    (major_serial_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (minor_serial_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (multi_ping_seq,) = struct.unpack_from('>H', payload, pos); pos += 2
    (frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sample_rate_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (receiver_bandwidth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_width_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_type_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_envlp_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_envlp_param_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_reserved,) = struct.unpack_from('>I', payload, pos); pos += 4
    (max_ping_rate_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_period_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (range_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (power_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (gain_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (control_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_steer_vert_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (projector_steer_horz_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (projector_bw_vert_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (projector_bw_horz_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (projector_focal_pt_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_weight_window_type,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_weight_window_param,) = struct.unpack_from('>I', payload, pos); pos += 4
    (transmit_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    (hydrophone_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_weight_window_type,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_weight_window_param,) = struct.unpack_from('>I', payload, pos); pos += 4
    (receive_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_beam_width_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_filt_min_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_filt_max_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (depth_filt_min_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (depth_filt_max_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (absorption_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sound_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (spreading_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    raw_data_from_7027 = payload[pos]; pos += 1
    pos += 15  # reserved_2
    sv_source = payload[pos]; pos += 1
    layer_comp_flag = payload[pos]; pos += 1
    pos += 8  # reserved_3

    fields = {
        'ProtocolVersion': protocol_version,
        'DeviceID': device_id,
        'MajorSerialNumber': major_serial_number,
        'MinorSerialNumber': minor_serial_number,
        'PingNumber': ping_number,
        'MultiPingSeq': multi_ping_seq,
        'Frequency_Hz': frequency_raw / 1.0e3,
        'SampleRate_Hz': sample_rate_raw / 1.0e4,
        'ReceiverBandwidth_Hz': receiver_bandwidth_raw / 1.0e4,
        'TxPulseWidth_s': tx_pulse_width_raw / 1.0e7,
        'TxPulseTypeID': tx_pulse_type_id,
        'TxPulseEnvelopeID': tx_pulse_envlp_id,
        'TxPulseEnvelopeParam': tx_pulse_envlp_param_raw / 1.0e2,
        'TxPulseReserved': tx_pulse_reserved,
        'MaxPingRate_pps': max_ping_rate_raw / 1.0e6,
        'PingPeriod_s': ping_period_raw / 1.0e6,
        'Range_m': range_raw / 1.0e2,
        'Power_dB': power_raw / 1.0e2,
        'Gain_dB': gain_raw / 1.0e2,
        'ControlFlags': control_flags,
        'ProjectorID': projector_id,
        'ProjectorSteerAnglVert_deg': projector_steer_vert_raw / 1.0e3,
        'ProjectorSteerAnglHoriz_deg': projector_steer_horz_raw / 1.0e3,
        'ProjectorBeamWidthVert_deg': projector_bw_vert_raw / 1.0e2,
        'ProjectorBeamWidthHoriz_deg': projector_bw_horz_raw / 1.0e2,
        'ProjectorBeamFocalPt_m': projector_focal_pt_raw / 1.0e2,
        'ProjectorBeamWeightingWindowType': projector_weight_window_type,
        'ProjectorBeamWeightingWindowParam': projector_weight_window_param,
        'TransmitFlags': transmit_flags,
        'HydrophoneID': hydrophone_id,
        'ReceivingBeamWeightingWindowType': rx_weight_window_type,
        'ReceivingBeamWeightingWindowParam': rx_weight_window_param,
        'ReceiveFlags': receive_flags,
        'ReceiveBeamWidth_deg': rx_beam_width_raw / 1.0e2,
        'RangeFiltMin_m': range_filt_min_raw / 1.0e1,
        'RangeFiltMax_m': range_filt_max_raw / 1.0e1,
        'DepthFiltMin_m': depth_filt_min_raw / 1.0e1,
        'DepthFiltMax_m': depth_filt_max_raw / 1.0e1,
        'Absorption_dBkm': absorption_raw / 1.0e3,
        'SoundVelocity_mps': sound_velocity_raw / 1.0e1,
        'Spreading_dB': spreading_raw / 1.0e3,
        'RawDataFrom7027': raw_data_from_7027,
        'SvSource': sv_source,
        'LayerCompFlag': layer_comp_flag,
    }
    return fields, {}, pos - start


def _encode_reson7125_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_RESON_7125_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_reson7125_specific(). Ported from gsf_enc.c's
    EncodeReson7100Specific(). Several fields (Frequency_Hz,
    SampleRate_Hz, ReceiverBandwidth_Hz, TxPulseWidth_s,
    TxPulseEnvelopeParam, MaxPingRate_pps, PingPeriod_s, Range_m, and the
    beam-width/filter/absorption/sound-velocity/spreading fields) round
    with the reference encoder's unconditional `+0.501` (they're always
    non-negative in practice); Power_dB, Gain_dB, and the projector
    steering angles use a sign-aware +/-0.501 branch. Both are equivalent
    to this module's standard _gsf_round() convention, used uniformly
    here for every scaled field.
    """
    g = fields.get
    body = struct.pack('>H', int(g('ProtocolVersion', 0)))
    body += struct.pack('>I', int(g('DeviceID', 0)))
    body += b'\x00' * 16  # reserved_1
    body += struct.pack('>I', int(g('MajorSerialNumber', 0)))
    body += struct.pack('>I', int(g('MinorSerialNumber', 0)))
    body += struct.pack('>I', int(g('PingNumber', 0)))
    body += struct.pack('>H', int(g('MultiPingSeq', 0)))
    body += struct.pack('>I', _gsf_round(g('Frequency_Hz', 0.0) * 1.0e3))
    body += struct.pack('>I', _gsf_round(g('SampleRate_Hz', 0.0) * 1.0e4))
    body += struct.pack('>I', _gsf_round(g('ReceiverBandwidth_Hz', 0.0) * 1.0e4))
    body += struct.pack('>I', _gsf_round(g('TxPulseWidth_s', 0.0) * 1.0e7))
    body += struct.pack('>I', int(g('TxPulseTypeID', 0)))
    body += struct.pack('>I', int(g('TxPulseEnvelopeID', 0)))
    body += struct.pack('>I', _gsf_round(g('TxPulseEnvelopeParam', 0.0) * 1.0e2))
    body += struct.pack('>I', int(g('TxPulseReserved', 0)))
    body += struct.pack('>I', _gsf_round(g('MaxPingRate_pps', 0.0) * 1.0e6))
    body += struct.pack('>I', _gsf_round(g('PingPeriod_s', 0.0) * 1.0e6))
    body += struct.pack('>I', _gsf_round(g('Range_m', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('Power_dB', 0.0) * 1.0e2))
    body += struct.pack('>i', _gsf_round(g('Gain_dB', 0.0) * 1.0e2))
    body += struct.pack('>I', int(g('ControlFlags', 0)))
    body += struct.pack('>I', int(g('ProjectorID', 0)))
    body += struct.pack('>i', _gsf_round(g('ProjectorSteerAnglVert_deg', 0.0) * 1.0e3))
    body += struct.pack('>i', _gsf_round(g('ProjectorSteerAnglHoriz_deg', 0.0) * 1.0e3))
    body += struct.pack('>H', _gsf_round(g('ProjectorBeamWidthVert_deg', 0.0) * 1.0e2))
    body += struct.pack('>H', _gsf_round(g('ProjectorBeamWidthHoriz_deg', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('ProjectorBeamFocalPt_m', 0.0) * 1.0e2))
    body += struct.pack('>I', int(g('ProjectorBeamWeightingWindowType', 0)))
    body += struct.pack('>I', int(g('ProjectorBeamWeightingWindowParam', 0)))
    body += struct.pack('>I', int(g('TransmitFlags', 0)))
    body += struct.pack('>I', int(g('HydrophoneID', 0)))
    body += struct.pack('>I', int(g('ReceivingBeamWeightingWindowType', 0)))
    body += struct.pack('>I', int(g('ReceivingBeamWeightingWindowParam', 0)))
    body += struct.pack('>I', int(g('ReceiveFlags', 0)))
    body += struct.pack('>H', _gsf_round(g('ReceiveBeamWidth_deg', 0.0) * 1.0e2))
    body += struct.pack('>H', _gsf_round(g('RangeFiltMin_m', 0.0) * 1.0e1))
    body += struct.pack('>H', _gsf_round(g('RangeFiltMax_m', 0.0) * 1.0e1))
    body += struct.pack('>H', _gsf_round(g('DepthFiltMin_m', 0.0) * 1.0e1))
    body += struct.pack('>H', _gsf_round(g('DepthFiltMax_m', 0.0) * 1.0e1))
    body += struct.pack('>I', _gsf_round(g('Absorption_dBkm', 0.0) * 1.0e3))
    body += struct.pack('>H', _gsf_round(g('SoundVelocity_mps', 0.0) * 1.0e1))
    body += struct.pack('>I', _gsf_round(g('Spreading_dB', 0.0) * 1.0e3))
    body += struct.pack('>B', int(g('RawDataFrom7027', 0)) & 0xFF)
    body += b'\x00' * 15  # reserved_2
    body += struct.pack('>B', int(g('SvSource', 0)) & 0xFF)
    body += struct.pack('>B', int(g('LayerCompFlag', 0)) & 0xFF)
    body += b'\x00' * 8  # reserved_3
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[138] = ("Reson7125", _decode_reson7125_specific, _encode_reson7125_specific)


def _decode_reson_tseries_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_RESON_TSERIES_SPECIFIC subrecord (id
    155, Reson T20/T50 series). Ported from gsf_dec.c's
    DecodeResonTSeriesSpecific() -- the largest sensor-specific struct in
    gsf.h (~90 named fields plus large spare ranges, 715 bytes on the
    wire). No nested arrays or bitmask-gated sub-blocks, unlike EM3/EM4 --
    a single large flat struct, read and written straight through.

    Bitmask fields (ControlFlags, TransmitFlags, ReceiveFlags,
    DetectionFlags) are kept as a single raw int, not split into booleans,
    matching this codebase's convention elsewhere.

    Quirk (present in the real wire format, not a bug): SoundVelocity_mps
    is stored TWICE -- once as a 2-byte low-precision value (*10) right
    after Absorption_dBkm, and again as a 4-byte high-precision value
    (*1.0e6) right after DeviceDescription. gsf_dec.c only keeps the
    high-precision value if it's nonzero, overwriting the low-precision
    read; this port does the same and exposes only the single resolved
    'SoundVelocity_mps' field. _encode_reson_tseries_specific() writes
    BOTH wire copies from that one field, mirroring gsf_enc.c exactly (no
    information is lost either way).

    Checked for the same tx_pulse_reserved width bug found in the related
    Reson7125 decoder (DecodeReson7100Specific reads it from a stale
    2-byte local instead of the 4-byte one just loaded) -- this decoder
    does NOT have that bug: TxPulseReserved is correctly read as its own
    fresh 2-byte value here.

    A few of gsf.h's own field-comments understate the on-disk width
    (e.g. "two byte" for what DecodeResonTSeriesSpecific/
    EncodeResonTSeriesSpecific actually read/write as 4 bytes, or "four
    byte" for MatchFilterShadingValue which is actually 2) -- this port
    follows the C code's actual memcpy/htons/htonl widths, not the
    comments, throughout.

    Untested against a verified GSF file: no sample data containing a
    RESON_TSERIES_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (no nested arrays), bytes_consumed).
    """
    start = pos
    (protocol_version,) = struct.unpack_from('>H', payload, pos); pos += 2
    (device_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (number_devices,) = struct.unpack_from('>I', payload, pos); pos += 4
    (system_enumerator,) = struct.unpack_from('>H', payload, pos); pos += 2
    pos += 10  # reserved_1
    (major_serial_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (minor_serial_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (multi_ping_seq,) = struct.unpack_from('>H', payload, pos); pos += 2
    (frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sample_rate_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (receiver_bandwidth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_width_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_type_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_envlp_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_envlp_param_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tx_pulse_reserved,) = struct.unpack_from('>H', payload, pos); pos += 2
    (max_ping_rate_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_period_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (range_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (power_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (gain_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (control_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_steer_vert_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (projector_steer_horz_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (projector_bw_vert_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (projector_bw_horz_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (projector_focal_pt_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_weight_window_type,) = struct.unpack_from('>I', payload, pos); pos += 4
    (projector_weight_window_param,) = struct.unpack_from('>I', payload, pos); pos += 4
    (transmit_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    (hydrophone_id,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_weight_window_type,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_weight_window_param,) = struct.unpack_from('>I', payload, pos); pos += 4
    (receive_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_beam_width_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_filt_min_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (range_filt_max_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (depth_filt_min_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (depth_filt_max_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (absorption_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sound_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    sv_source = payload[pos]; pos += 1
    (spreading_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (beam_spacing_mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sonar_source_mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    coverage_mode = payload[pos]; pos += 1
    (coverage_angle_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (horiz_rx_steer_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    pos += 3  # reserved_2
    (uncertainty_type,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_steering_angle_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (applied_roll_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (detection_algorithm,) = struct.unpack_from('>H', payload, pos); pos += 2
    (detection_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    device_description = payload[pos:pos + 60].split(b'\x00', 1)[0].decode('ascii', 'replace'); pos += 60
    (sound_velocity_hp_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    pos += 60  # reserved_7027
    match_filter_control = payload[pos]; pos += 1
    (match_filter_start_freq_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (match_filter_end_freq_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    match_filter_window_type = payload[pos]; pos += 1
    (match_filter_shading_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (match_filter_pulse_width_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    pos += 52  # reserved_7002
    pos += 32  # reserved_3
    pos += 288  # reserved_4

    sound_velocity_mps = sound_velocity_raw / 1.0e1
    if sound_velocity_hp_raw > 0:
        sound_velocity_mps = sound_velocity_hp_raw / 1.0e6

    fields = {
        'ProtocolVersion': protocol_version,
        'DeviceID': device_id,
        'NumberDevices': number_devices,
        'SystemEnumerator': system_enumerator,
        'MajorSerialNumber': major_serial_number,
        'MinorSerialNumber': minor_serial_number,
        'PingNumber': ping_number,
        'MultiPingSeq': multi_ping_seq,
        'Frequency_Hz': frequency_raw / 1.0e3,
        'SampleRate_Hz': sample_rate_raw / 1.0e4,
        'ReceiverBandwidth_Hz': receiver_bandwidth_raw / 1.0e4,
        'TxPulseWidth_s': tx_pulse_width_raw / 1.0e7,
        'TxPulseTypeID': tx_pulse_type_id,
        'TxPulseEnvelopeID': tx_pulse_envlp_id,
        'TxPulseEnvelopeParam': tx_pulse_envlp_param_raw / 1.0e2,
        'TxPulseMode': tx_pulse_mode,
        'TxPulseReserved': tx_pulse_reserved,
        'MaxPingRate_pps': max_ping_rate_raw / 1.0e6,
        'PingPeriod_s': ping_period_raw / 1.0e6,
        'Range_m': range_raw / 1.0e2,
        'Power_dB': power_raw / 1.0e2,
        'Gain_dB': gain_raw / 1.0e2,
        'ControlFlags': control_flags,
        'ProjectorID': projector_id,
        'ProjectorSteerAnglVert_deg': projector_steer_vert_raw / 1.0e3,
        'ProjectorSteerAnglHoriz_deg': projector_steer_horz_raw / 1.0e3,
        'ProjectorBeamWidthVert_deg': projector_bw_vert_raw / 1.0e2,
        'ProjectorBeamWidthHoriz_deg': projector_bw_horz_raw / 1.0e2,
        'ProjectorBeamFocalPt_m': projector_focal_pt_raw / 1.0e2,
        'ProjectorBeamWeightingWindowType': projector_weight_window_type,
        'ProjectorBeamWeightingWindowParam': projector_weight_window_param,
        'TransmitFlags': transmit_flags,
        'HydrophoneID': hydrophone_id,
        'ReceivingBeamWeightingWindowType': rx_weight_window_type,
        'ReceivingBeamWeightingWindowParam': rx_weight_window_param,
        'ReceiveFlags': receive_flags,
        'ReceiveBeamWidth_deg': rx_beam_width_raw / 1.0e2,
        'RangeFiltMin_m': range_filt_min_raw / 1.0e1,
        'RangeFiltMax_m': range_filt_max_raw / 1.0e1,
        'DepthFiltMin_m': depth_filt_min_raw / 1.0e1,
        'DepthFiltMax_m': depth_filt_max_raw / 1.0e1,
        'Absorption_dBkm': absorption_raw / 1.0e3,
        'SoundVelocity_mps': sound_velocity_mps,
        'SvSource': sv_source,
        'Spreading_dB': spreading_raw / 1.0e3,
        'BeamSpacingMode': beam_spacing_mode,
        'SonarSourceMode': sonar_source_mode,
        'CoverageMode': coverage_mode,
        'CoverageAngle_deg': coverage_angle_raw / 1.0e2,
        'HorizontalReceiverSteeringAngle_deg': horiz_rx_steer_raw / 1.0e2,
        'UncertaintyType': uncertainty_type,
        'TransmitterSteeringAngle_rad': tx_steering_angle_raw / 1.0e5,
        'AppliedRoll_rad': applied_roll_raw / 1.0e5,
        'DetectionAlgorithm': detection_algorithm,
        'DetectionFlags': detection_flags,
        'DeviceDescription': device_description,
        'MatchFilterControl': match_filter_control,
        'MatchFilterStartFreq_Hz': match_filter_start_freq_raw / 1.0e2,
        'MatchFilterEndFreq_Hz': match_filter_end_freq_raw / 1.0e2,
        'MatchFilterWindowType': match_filter_window_type,
        'MatchFilterShadingValue': match_filter_shading_raw / 1.0e4,
        'MatchFilterEffectivePulseWidth_s': match_filter_pulse_width_raw / 1.0e11,
    }
    return fields, {}, pos - start


def _encode_reson_tseries_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_RESON_TSERIES_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_reson_tseries_specific(). Ported from gsf_enc.c's
    EncodeResonTSeriesSpecific(). Writes SoundVelocity_mps out twice (a
    2-byte low-precision copy and a 4-byte high-precision copy), matching
    the reference encoder exactly -- see _decode_reson_tseries_specific()'s
    docstring. Every scaled field uses this module's standard, sign-correct
    _gsf_round() convention, equivalent to the reference encoder's mix of
    unconditional and sign-branched +/-0.501 rounding.
    """
    g = fields.get
    body = struct.pack('>H', int(g('ProtocolVersion', 0)))
    body += struct.pack('>I', int(g('DeviceID', 0)))
    body += struct.pack('>I', int(g('NumberDevices', 0)))
    body += struct.pack('>H', int(g('SystemEnumerator', 0)))
    body += b'\x00' * 10  # reserved_1
    body += struct.pack('>I', int(g('MajorSerialNumber', 0)))
    body += struct.pack('>I', int(g('MinorSerialNumber', 0)))
    body += struct.pack('>I', int(g('PingNumber', 0)))
    body += struct.pack('>H', int(g('MultiPingSeq', 0)))
    body += struct.pack('>I', _gsf_round(g('Frequency_Hz', 0.0) * 1.0e3))
    body += struct.pack('>I', _gsf_round(g('SampleRate_Hz', 0.0) * 1.0e4))
    body += struct.pack('>I', _gsf_round(g('ReceiverBandwidth_Hz', 0.0) * 1.0e4))
    body += struct.pack('>I', _gsf_round(g('TxPulseWidth_s', 0.0) * 1.0e7))
    body += struct.pack('>I', int(g('TxPulseTypeID', 0)))
    body += struct.pack('>I', int(g('TxPulseEnvelopeID', 0)))
    body += struct.pack('>I', _gsf_round(g('TxPulseEnvelopeParam', 0.0) * 1.0e2))
    body += struct.pack('>H', int(g('TxPulseMode', 0)))
    body += struct.pack('>H', int(g('TxPulseReserved', 0)))
    body += struct.pack('>I', _gsf_round(g('MaxPingRate_pps', 0.0) * 1.0e6))
    body += struct.pack('>I', _gsf_round(g('PingPeriod_s', 0.0) * 1.0e6))
    body += struct.pack('>I', _gsf_round(g('Range_m', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('Power_dB', 0.0) * 1.0e2))
    body += struct.pack('>i', _gsf_round(g('Gain_dB', 0.0) * 1.0e2))
    body += struct.pack('>I', int(g('ControlFlags', 0)))
    body += struct.pack('>I', int(g('ProjectorID', 0)))
    body += struct.pack('>i', _gsf_round(g('ProjectorSteerAnglVert_deg', 0.0) * 1.0e3))
    body += struct.pack('>i', _gsf_round(g('ProjectorSteerAnglHoriz_deg', 0.0) * 1.0e3))
    body += struct.pack('>H', _gsf_round(g('ProjectorBeamWidthVert_deg', 0.0) * 1.0e2))
    body += struct.pack('>H', _gsf_round(g('ProjectorBeamWidthHoriz_deg', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('ProjectorBeamFocalPt_m', 0.0) * 1.0e2))
    body += struct.pack('>I', int(g('ProjectorBeamWeightingWindowType', 0)))
    body += struct.pack('>I', int(g('ProjectorBeamWeightingWindowParam', 0)))
    body += struct.pack('>I', int(g('TransmitFlags', 0)))
    body += struct.pack('>I', int(g('HydrophoneID', 0)))
    body += struct.pack('>I', int(g('ReceivingBeamWeightingWindowType', 0)))
    body += struct.pack('>I', int(g('ReceivingBeamWeightingWindowParam', 0)))
    body += struct.pack('>I', int(g('ReceiveFlags', 0)))
    body += struct.pack('>H', _gsf_round(g('ReceiveBeamWidth_deg', 0.0) * 1.0e2))
    body += struct.pack('>i', _gsf_round(g('RangeFiltMin_m', 0.0) * 1.0e1))
    body += struct.pack('>i', _gsf_round(g('RangeFiltMax_m', 0.0) * 1.0e1))
    body += struct.pack('>i', _gsf_round(g('DepthFiltMin_m', 0.0) * 1.0e1))
    body += struct.pack('>i', _gsf_round(g('DepthFiltMax_m', 0.0) * 1.0e1))
    body += struct.pack('>I', _gsf_round(g('Absorption_dBkm', 0.0) * 1.0e3))
    sound_velocity_mps = g('SoundVelocity_mps', 0.0)
    body += struct.pack('>H', _gsf_round(sound_velocity_mps * 1.0e1))
    body += struct.pack('>B', int(g('SvSource', 0)) & 0xFF)
    body += struct.pack('>I', _gsf_round(g('Spreading_dB', 0.0) * 1.0e3))
    body += struct.pack('>H', int(g('BeamSpacingMode', 0)))
    body += struct.pack('>H', int(g('SonarSourceMode', 0)))
    body += struct.pack('>B', int(g('CoverageMode', 0)) & 0xFF)
    body += struct.pack('>I', _gsf_round(g('CoverageAngle_deg', 0.0) * 1.0e2))
    body += struct.pack('>i', _gsf_round(g('HorizontalReceiverSteeringAngle_deg', 0.0) * 1.0e2))
    body += b'\x00' * 3  # reserved_2
    body += struct.pack('>I', int(g('UncertaintyType', 0)))
    body += struct.pack('>i', _gsf_round(g('TransmitterSteeringAngle_rad', 0.0) * 1.0e5))
    body += struct.pack('>i', _gsf_round(g('AppliedRoll_rad', 0.0) * 1.0e5))
    body += struct.pack('>H', int(g('DetectionAlgorithm', 0)))
    body += struct.pack('>I', int(g('DetectionFlags', 0)))
    body += g('DeviceDescription', "").encode('ascii')[:60].ljust(60, b'\x00')
    body += struct.pack('>I', _gsf_round(sound_velocity_mps * 1.0e6))
    body += b'\x00' * 60  # reserved_7027
    body += struct.pack('>B', int(g('MatchFilterControl', 0)) & 0xFF)
    body += struct.pack('>I', _gsf_round(g('MatchFilterStartFreq_Hz', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('MatchFilterEndFreq_Hz', 0.0) * 1.0e2))
    body += struct.pack('>B', int(g('MatchFilterWindowType', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('MatchFilterShadingValue', 0.0) * 1.0e4))
    body += struct.pack('>I', _gsf_round(g('MatchFilterEffectivePulseWidth_s', 0.0) * 1.0e11))
    body += b'\x00' * 52  # reserved_7002
    body += b'\x00' * 32  # reserved_3
    body += b'\x00' * 288  # reserved_4
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[155] = ("ResonTSeries", _decode_reson_tseries_specific, _encode_reson_tseries_specific)


def _decode_geoswath_plus_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_GEOSWATH_PLUS_SPECIFIC subrecord
    (id 136). Ported from gsf_dec.c's DecodeGeoSwathPlusSpecific().

    Untested against a verified GSF file: no sample data containing a
    GEOSWATH_PLUS_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (data_source,) = struct.unpack_from('>H', payload, pos); pos += 2
    (side,) = struct.unpack_from('>H', payload, pos); pos += 2
    (model_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (frequency_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (echosounder_type,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (num_nav_samples,) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_attitude_samples,) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_heading_samples,) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_minisvs_samples,) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_echosounder_samples,) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_raa_samples,) = struct.unpack_from('>H', payload, pos); pos += 2
    (mean_sv_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (valid_beams,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sample_rate_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (pulse_length,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_length,) = struct.unpack_from('>H', payload, pos); pos += 2
    (transmit_power,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sidescan_gain_channel,) = struct.unpack_from('>H', payload, pos); pos += 2
    (stabilization,) = struct.unpack_from('>H', payload, pos); pos += 2
    (gps_quality,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_uncertainty_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (angle_uncertainty_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    pos += 32  # spare

    fields = {
        'DataSource': data_source,
        'Side': side,
        'ModelNumber': model_number,
        'Frequency_Hz': frequency_raw * 10.0,
        'EchosounderType': echosounder_type,
        'PingNumber': ping_number,
        'NumNavSamples': num_nav_samples,
        'NumAttitudeSamples': num_attitude_samples,
        'NumHeadingSamples': num_heading_samples,
        'NumMiniSVSSamples': num_minisvs_samples,
        'NumEchosounderSamples': num_echosounder_samples,
        'NumRaaSamples': num_raa_samples,
        'MeanSV_mps': mean_sv_raw / 20.0,
        'SurfaceVelocity_mps': surface_velocity_raw / 20.0,
        'ValidBeams': valid_beams,
        'SampleRate_Hz': sample_rate_raw * 10.0,
        'PulseLength_us': float(pulse_length),
        'PingLength_m': ping_length,
        'TransmitPower': transmit_power,
        'SidescanGainChannel': sidescan_gain_channel,
        'Stabilization': stabilization,
        'GpsQuality': gps_quality,
        'RangeUncertainty_m': range_uncertainty_raw / 1000.0,
        'AngleUncertainty_deg': angle_uncertainty_raw / 100.0,
    }
    return fields, {}, pos - start


def _encode_geoswath_plus_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_GEOSWATH_PLUS_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_geoswath_plus_specific(). Ported from gsf_enc.c's
    EncodeGeoSwathPlusSpecific() (all fields here are non-negative in
    practice, so the reference encoder's unconditional `+ 0.501` and this
    module's sign-correct _gsf_round() agree).
    """
    g = fields.get
    body = struct.pack('>H', int(g('DataSource', 0)))
    body += struct.pack('>H', int(g('Side', 0)))
    body += struct.pack('>H', int(g('ModelNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('Frequency_Hz', 0.0) / 10.0))
    body += struct.pack('>H', int(g('EchosounderType', 0)))
    body += struct.pack('>I', int(g('PingNumber', 0)))
    body += struct.pack('>H', int(g('NumNavSamples', 0)))
    body += struct.pack('>H', int(g('NumAttitudeSamples', 0)))
    body += struct.pack('>H', int(g('NumHeadingSamples', 0)))
    body += struct.pack('>H', int(g('NumMiniSVSSamples', 0)))
    body += struct.pack('>H', int(g('NumEchosounderSamples', 0)))
    body += struct.pack('>H', int(g('NumRaaSamples', 0)))
    body += struct.pack('>H', _gsf_round(g('MeanSV_mps', 0.0) * 20.0))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 20.0))
    body += struct.pack('>H', int(g('ValidBeams', 0)))
    body += struct.pack('>H', _gsf_round(g('SampleRate_Hz', 0.0) / 10.0))
    body += struct.pack('>H', int(g('PulseLength_us', 0)))
    body += struct.pack('>H', int(g('PingLength_m', 0)))
    body += struct.pack('>H', int(g('TransmitPower', 0)))
    body += struct.pack('>H', int(g('SidescanGainChannel', 0)))
    body += struct.pack('>H', int(g('Stabilization', 0)))
    body += struct.pack('>H', int(g('GpsQuality', 0)))
    body += struct.pack('>H', _gsf_round(g('RangeUncertainty_m', 0.0) * 1000.0))
    body += struct.pack('>H', _gsf_round(g('AngleUncertainty_deg', 0.0) * 100.0))
    body += b'\x00' * 32  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[136] = (
    "GeoSwathPlus", _decode_geoswath_plus_specific, _encode_geoswath_plus_specific)


def _decode_klein5410bss_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_KLEIN_5410_BSS_SPECIFIC subrecord
    (id 137) -- the ping-level sensor-specific block, distinct from
    _decode_klein5410bss_imagery_specific() (the smaller preamble inside
    the BRB intensity subrecord, id 21). Ported from gsf_dec.c's
    DecodeKlein5410BssSpecific().

    Note: FishDepth_V/FishAltitude_m/SoundSpeed_mps are decoded from their
    raw 4-byte field as UNSIGNED (matching gsf_dec.c's own `(double)
    ntohl(ltemp)`, with no sign reinterpretation), even though
    EncodeKlein5410BssSpecific() allows negative values in for those same
    three fields via a signed rounding branch -- an asymmetry in the
    reference library itself (a file written with a negative value there
    would not decode back correctly even with the real gsflib). This
    encoder matches gsf_dec.c's decode side; see
    _encode_klein5410bss_specific()'s docstring for the encode-side
    consequence.

    Untested against a verified GSF file: no sample data containing a
    KLEIN_5410_BSS_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (data_source,) = struct.unpack_from('>H', payload, pos); pos += 2
    (side,) = struct.unpack_from('>H', payload, pos); pos += 2
    (model_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (acoustic_frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sampling_frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (num_samples,) = struct.unpack_from('>I', payload, pos); pos += 4
    (num_raa_samples,) = struct.unpack_from('>I', payload, pos); pos += 4
    (error_flags,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rng,) = struct.unpack_from('>I', payload, pos); pos += 4
    (fish_depth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (fish_altitude_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sound_speed_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_waveform,) = struct.unpack_from('>H', payload, pos); pos += 2
    (altimeter,) = struct.unpack_from('>H', payload, pos); pos += 2
    (raw_data_config,) = struct.unpack_from('>I', payload, pos); pos += 4
    pos += 32  # spare

    fields = {
        'DataSource': data_source,
        'Side': side,
        'ModelNumber': model_number,
        'AcousticFrequency_Hz': acoustic_frequency_raw / 1000.0,
        'SamplingFrequency_Hz': sampling_frequency_raw / 1000.0,
        'PingNumber': ping_number,
        'NumSamples': num_samples,
        'NumRaaSamples': num_raa_samples,
        'ErrorFlags': error_flags,
        'Range': rng,
        'FishDepth_V': fish_depth_raw / 1000.0,
        'FishAltitude_m': fish_altitude_raw / 1000.0,
        'SoundSpeed_mps': sound_speed_raw / 1000.0,
        'TxWaveform': tx_waveform,
        'Altimeter': altimeter,
        'RawDataConfig': raw_data_config,
    }
    return fields, {}, pos - start


def _encode_klein5410bss_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_KLEIN_5410_BSS_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_klein5410bss_specific(). Ported from gsf_enc.c's
    EncodeKlein5410BssSpecific().

    FishDepth_V/FishAltitude_m/SoundSpeed_mps are packed as unsigned
    32-bit fields (matching how _decode_klein5410bss_specific() reads
    them back) -- a negative value raises struct.error rather than
    silently producing bytes the decoder can't recover, which is the
    practical effect of the same asymmetry in gsf_enc.c/gsf_dec.c (see
    _decode_klein5410bss_specific()'s docstring).
    """
    g = fields.get
    body = struct.pack('>H', int(g('DataSource', 0)))
    body += struct.pack('>H', int(g('Side', 0)))
    body += struct.pack('>H', int(g('ModelNumber', 0)))
    body += struct.pack('>I', _gsf_round(g('AcousticFrequency_Hz', 0.0) * 1000.0))
    body += struct.pack('>I', _gsf_round(g('SamplingFrequency_Hz', 0.0) * 1000.0))
    body += struct.pack('>I', int(g('PingNumber', 0)))
    body += struct.pack('>I', int(g('NumSamples', 0)))
    body += struct.pack('>I', int(g('NumRaaSamples', 0)))
    body += struct.pack('>I', int(g('ErrorFlags', 0)))
    body += struct.pack('>I', int(g('Range', 0)))
    body += struct.pack('>I', _gsf_round(g('FishDepth_V', 0.0) * 1000.0))
    body += struct.pack('>I', _gsf_round(g('FishAltitude_m', 0.0) * 1000.0))
    body += struct.pack('>I', _gsf_round(g('SoundSpeed_mps', 0.0) * 1000.0))
    body += struct.pack('>H', int(g('TxWaveform', 0)))
    body += struct.pack('>H', int(g('Altimeter', 0)))
    body += struct.pack('>I', int(g('RawDataConfig', 0)))
    body += b'\x00' * 32  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[137] = (
    "Klein5410Bss", _decode_klein5410bss_specific, _encode_klein5410bss_specific)


def _decode_sass_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SASS_SPECIFIC (id 108) or
    _TYPEIII_SEABEAM_SPECIFIC (id 112) subrecord -- identical wire format,
    one struct (gsf.h's t_gsfTypeIIISpecific) and decoder shared by both in
    gsflib. Ported from gsf_dec.c's DecodeSASSSpecific() (byte-for-byte
    identical to DecodeTypeIIISeaBeamSpecific(), verified against both).
    Both record types are marked obsolete in gsf.h (replaced by
    CMP_SASS_SPECIFIC, see _decode_cmp_sass_specific()), but gsflib still
    ships a full decode/encode pair for them, so this does too.

    Untested against a verified GSF file: no sample data containing a
    SASS_SPECIFIC/TYPEIII_SEABEAM_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (leftmost_beam,) = struct.unpack_from('>H', payload, pos); pos += 2
    (rightmost_beam,) = struct.unpack_from('>H', payload, pos); pos += 2
    (total_beams,) = struct.unpack_from('>H', payload, pos); pos += 2
    (nav_mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (mission_number,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {
        'LeftmostBeam': leftmost_beam,
        'RightmostBeam': rightmost_beam,
        'TotalBeams': total_beams,
        'NavMode': nav_mode,
        'PingNumber': ping_number,
        'MissionNumber': mission_number,
    }
    return fields, {}, pos - start


def _encode_sass_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SASS_SPECIFIC or
    _TYPEIII_SEABEAM_SPECIFIC subrecord (subrecord_id selects which -- see
    _decode_sass_specific()), including its own 4-byte subrecord id+size
    word: the inverse of _decode_sass_specific(). Ported from gsf_enc.c's
    EncodeSASSSpecific() (byte-for-byte identical to
    EncodeTypeIIISeaBeamSpecific()).
    """
    g = fields.get
    body = struct.pack(
        '>6H',
        int(g('LeftmostBeam', 0)), int(g('RightmostBeam', 0)), int(g('TotalBeams', 0)),
        int(g('NavMode', 0)), int(g('PingNumber', 0)), int(g('MissionNumber', 0)))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[108] = ("SASS", _decode_sass_specific, _encode_sass_specific)
_PING_SENSOR_SPECIFIC_CODECS[112] = ("TypeIIISeaBeam", _decode_sass_specific, _encode_sass_specific)


def _decode_delta_t_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_DELTA_T_SPECIFIC subrecord (id 150):
    Imagenex Delta T multibeam sensor metadata. Ported from gsf_dec.c's
    DecodeDeltaTSpecific(). Field scaling follows the C source exactly,
    including its asymmetries between similar-looking fields -- e.g.
    start_angle is stored as (angle+180)*100 but profile_tilt_angle as
    plain (angle+180) with no *100, and sector_size/acoustic_range/
    acoustic_frequency/range_resolution/repetition_rate carry no scale
    factor at all despite being stored as doubles.

    Untested against a verified GSF file: no sample data containing a
    DELTA_T_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    decode_file_type = payload[pos:pos + 4].decode('ascii', 'replace').rstrip('\x00'); pos += 4
    version = payload[pos]; pos += 1
    (ping_byte_size,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sec, nsec) = struct.unpack_from('>2I', payload, pos); pos += 8
    (samples_per_beam,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sector_size_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (start_angle_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (angle_increment_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (acoustic_range,) = struct.unpack_from('>H', payload, pos); pos += 2
    (acoustic_frequency,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sound_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_resolution_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (profile_tilt_angle_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (repetition_rate_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    intensity_flag = payload[pos]; pos += 1
    (ping_latency_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (data_latency_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    sample_rate_flag = payload[pos]; pos += 1
    option_flags = payload[pos]; pos += 1
    num_pings_avg = payload[pos]; pos += 1
    (center_ping_time_offset_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    user_defined_byte = payload[pos]; pos += 1
    (altitude_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    external_sensor_flags = payload[pos]; pos += 1
    (pulse_length_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fore_aft_beamwidth_raw = payload[pos]; pos += 1
    athwartships_beamwidth_raw = payload[pos]; pos += 1
    pos += 32  # spare

    fields = {
        'DecodeFileType': decode_file_type,
        'Version': version,
        'PingByteSize': ping_byte_size,
        'InterrogationTime': _gsf_timestamp(sec, nsec),
        'SamplesPerBeam': samples_per_beam,
        'SectorSize_deg': float(sector_size_raw),
        'StartAngle_deg': start_angle_raw / 100.0 - 180.0,
        'AngleIncrement_deg': angle_increment_raw / 100.0,
        'AcousticRange_m': acoustic_range,
        'AcousticFrequency_kHz': acoustic_frequency,
        'SoundVelocity_mps': sound_velocity_raw / 10.0,
        'RangeResolution_cm': float(range_resolution_raw),
        'ProfileTiltAngle_deg': profile_tilt_angle_raw - 180.0,
        'RepetitionRate_ms': float(repetition_rate_raw),
        'PingNumber': ping_number,
        'IntensityFlag': intensity_flag,
        'PingLatency_s': ping_latency_raw / 10000.0,
        'DataLatency_s': data_latency_raw / 10000.0,
        'SampleRateFlag': sample_rate_flag,
        'OptionFlags': option_flags,
        'NumPingsAvg': num_pings_avg,
        'CenterPingTimeOffset_s': center_ping_time_offset_raw / 10000.0,
        'UserDefinedByte': user_defined_byte,
        'Altitude_m': altitude_raw / 100.0,
        'ExternalSensorFlags': external_sensor_flags,
        'PulseLength_s': pulse_length_raw / 1.0e6,
        'ForeAftBeamwidth_deg': fore_aft_beamwidth_raw / 10.0,
        'AthwartshipsBeamwidth_deg': athwartships_beamwidth_raw / 10.0,
    }
    return fields, {}, pos - start


def _encode_delta_t_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_DELTA_T_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_delta_t_specific(). Ported from gsf_enc.c's
    EncodeDeltaTSpecific() -- see that decoder's docstring for the
    per-field scaling asymmetries this preserves.
    """
    g = fields.get
    decode_file_type = str(g('DecodeFileType', '')).encode('ascii')[:4].ljust(4, b'\x00')
    body = decode_file_type
    body += struct.pack('>B', int(g('Version', 0)) & 0xFF)
    body += struct.pack('>H', int(g('PingByteSize', 0)))
    sec, nsec = _gsf_epoch(g('InterrogationTime', 0))
    body += struct.pack('>2I', sec, nsec)
    body += struct.pack('>H', int(g('SamplesPerBeam', 0)))
    body += struct.pack('>H', _gsf_round(g('SectorSize_deg', 0.0)))
    body += struct.pack('>H', _gsf_round((g('StartAngle_deg', -180.0) + 180.0) * 100.0))
    body += struct.pack('>H', _gsf_round(g('AngleIncrement_deg', 0.0) * 100.0))
    body += struct.pack('>H', int(g('AcousticRange_m', 0)))
    body += struct.pack('>H', int(g('AcousticFrequency_kHz', 0)))
    body += struct.pack('>H', _gsf_round(g('SoundVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>H', _gsf_round(g('RangeResolution_cm', 0.0)))
    body += struct.pack('>H', _gsf_round(g('ProfileTiltAngle_deg', -180.0) + 180.0))
    body += struct.pack('>H', _gsf_round(g('RepetitionRate_ms', 0.0)))
    body += struct.pack('>I', int(g('PingNumber', 0)))
    body += struct.pack('>B', int(g('IntensityFlag', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('PingLatency_s', 0.0) * 10000.0))
    body += struct.pack('>H', _gsf_round(g('DataLatency_s', 0.0) * 10000.0))
    body += struct.pack('>B', int(g('SampleRateFlag', 0)) & 0xFF)
    body += struct.pack('>B', int(g('OptionFlags', 0)) & 0xFF)
    body += struct.pack('>B', int(g('NumPingsAvg', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('CenterPingTimeOffset_s', 0.0) * 10000.0))
    body += struct.pack('>B', int(g('UserDefinedByte', 0)) & 0xFF)
    body += struct.pack('>I', _gsf_round(g('Altitude_m', 0.0) * 100.0))
    body += struct.pack('>B', int(g('ExternalSensorFlags', 0)) & 0xFF)
    body += struct.pack('>I', _gsf_round(g('PulseLength_s', 0.0) * 1.0e6))
    body += struct.pack('>B', _gsf_round(g('ForeAftBeamwidth_deg', 0.0) * 10.0) & 0xFF)
    body += struct.pack('>B', _gsf_round(g('AthwartshipsBeamwidth_deg', 0.0) * 10.0) & 0xFF)
    body += b'\x00' * 32  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[150] = ("DeltaT", _decode_delta_t_specific, _encode_delta_t_specific)


def _decode_r2sonic_specific(payload, pos):
    """
    Decode an R2SONIC_2020/2022/2024_SPECIFIC subrecord (ids 153/151/152 --
    one struct/codec shared by all three). Ported from gsf_dec.c's
    DecodeR2SonicSpecific(). Distinct from _decode_r2sonic_imagery_specific()
    (a different, smaller struct nested in the intensity-series subrecord,
    id 21) -- this is the ping-level "_SPECIFIC" subrecord.

    Untested against a verified GSF file: no sample data containing an
    R2SONIC_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    model_number = payload[pos:pos + 12].split(b'\x00', 1)[0].decode('ascii', 'replace'); pos += 12
    serial_number = payload[pos:pos + 12].split(b'\x00', 1)[0].decode('ascii', 'replace'); pos += 12
    (sec,) = struct.unpack_from('>I', payload, pos); pos += 4
    (nsec,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_period_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sound_speed_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_power_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_width_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_beamwidth_vert_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_beamwidth_horiz_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_steering_vert_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (tx_steering_horiz_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (tx_misc_info,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_bandwidth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_sample_rate_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_range_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_gain_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_spreading_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_absorption_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_mount_tilt_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (rx_misc_info,) = struct.unpack_from('>I', payload, pos); pos += 4
    (reserved,) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_beams,) = struct.unpack_from('>H', payload, pos); pos += 2
    a0_more_info = [v / 1.0e6 for v in struct.unpack_from('>6i', payload, pos)]; pos += 24
    a2_more_info = [v / 1.0e6 for v in struct.unpack_from('>6i', payload, pos)]; pos += 24
    (g0_depth_gate_min_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (g0_depth_gate_max_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (g0_depth_gate_slope_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    pos += 32  # spare

    fields = {
        'ModelNumber': model_number,
        'SerialNumber': serial_number,
        'PingTime': _gsf_timestamp(sec, nsec),
        'PingNumber': ping_number,
        'PingPeriod_s': ping_period_raw / 1.0e6,
        'SoundSpeed_mps': sound_speed_raw / 1.0e2,
        'Frequency_Hz': frequency_raw / 1.0e3,
        'TxPower_dB': tx_power_raw / 1.0e2,
        'TxPulseWidth_s': tx_pulse_width_raw / 1.0e7,
        'TxBeamwidthVert_deg': tx_beamwidth_vert_raw / 1.0e6,
        'TxBeamwidthHoriz_deg': tx_beamwidth_horiz_raw / 1.0e6,
        'TxSteeringVert_deg': tx_steering_vert_raw / 1.0e6,
        'TxSteeringHoriz_deg': tx_steering_horiz_raw / 1.0e6,
        'TxMiscInfo': tx_misc_info,
        'RxBandwidth_Hz': rx_bandwidth_raw / 1.0e4,
        'RxSampleRate_Hz': rx_sample_rate_raw / 1.0e3,
        'RxRange_m': rx_range_raw / 1.0e5,
        'RxGain_dB': rx_gain_raw / 1.0e2,
        'RxSpreading': rx_spreading_raw / 1.0e3,
        'RxAbsorption_dBkm': rx_absorption_raw / 1.0e3,
        'RxMountTilt_deg': rx_mount_tilt_raw / 1.0e6,
        'RxMiscInfo': rx_misc_info,
        'Reserved': reserved,
        'NumBeams': num_beams,
        'A0MoreInfo': a0_more_info,
        'A2MoreInfo': a2_more_info,
        'G0DepthGateMin_s': g0_depth_gate_min_raw / 1.0e6,
        'G0DepthGateMax_s': g0_depth_gate_max_raw / 1.0e6,
        'G0DepthGateSlope_deg': g0_depth_gate_slope_raw / 1.0e6,
    }
    return fields, {}, pos - start


def _encode_r2sonic_specific(subrecord_id, fields, tables=None):
    """
    Encode an R2SONIC_2020/2022/2024_SPECIFIC subrecord, including its own
    4-byte subrecord id+size word: the inverse of _decode_r2sonic_specific().
    Ported from gsf_enc.c's EncodeR2SonicSpecific(). The reference encoder
    rounds most fields with an unconditional `+0.501` (fields that are
    physically non-negative) and a few with a sign-aware +/-0.501 branch
    (tx_steering_vert/horiz, rx_mount_tilt, A0/A2_more_info,
    G0_depth_gate_slope); this uses the standard sign-correct _gsf_round()
    convention for every field, which is equivalent for the non-negative
    ones and matches the reference exactly for the signed ones.
    """
    g = fields.get

    def model_bytes(key):
        return g(key, "").encode('ascii')[:12].ljust(12, b'\x00')

    body = model_bytes('ModelNumber')
    body += model_bytes('SerialNumber')
    ping_time = g('PingTime')
    sec, nsec = _gsf_epoch(ping_time) if ping_time is not None else (0, 0)
    body += struct.pack('>2I', sec, nsec)
    body += struct.pack('>I', int(g('PingNumber', 0)))
    body += struct.pack('>I', _gsf_round(g('PingPeriod_s', 0.0) * 1.0e6))
    body += struct.pack('>I', _gsf_round(g('SoundSpeed_mps', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('Frequency_Hz', 0.0) * 1.0e3))
    body += struct.pack('>I', _gsf_round(g('TxPower_dB', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('TxPulseWidth_s', 0.0) * 1.0e7))
    body += struct.pack('>I', _gsf_round(g('TxBeamwidthVert_deg', 0.0) * 1.0e6))
    body += struct.pack('>I', _gsf_round(g('TxBeamwidthHoriz_deg', 0.0) * 1.0e6))
    body += struct.pack('>i', _gsf_round(g('TxSteeringVert_deg', 0.0) * 1.0e6))
    body += struct.pack('>i', _gsf_round(g('TxSteeringHoriz_deg', 0.0) * 1.0e6))
    body += struct.pack('>I', int(g('TxMiscInfo', 0)))
    body += struct.pack('>I', _gsf_round(g('RxBandwidth_Hz', 0.0) * 1.0e4))
    body += struct.pack('>I', _gsf_round(g('RxSampleRate_Hz', 0.0) * 1.0e3))
    body += struct.pack('>I', _gsf_round(g('RxRange_m', 0.0) * 1.0e5))
    body += struct.pack('>I', _gsf_round(g('RxGain_dB', 0.0) * 1.0e2))
    body += struct.pack('>I', _gsf_round(g('RxSpreading', 0.0) * 1.0e3))
    body += struct.pack('>I', _gsf_round(g('RxAbsorption_dBkm', 0.0) * 1.0e3))
    body += struct.pack('>i', _gsf_round(g('RxMountTilt_deg', 0.0) * 1.0e6))
    body += struct.pack('>I', int(g('RxMiscInfo', 0)))
    body += struct.pack('>H', int(g('Reserved', 0)))
    body += struct.pack('>H', int(g('NumBeams', 0)))
    a0_more_info = g('A0MoreInfo', [0.0] * 6)
    a2_more_info = g('A2MoreInfo', [0.0] * 6)
    body += struct.pack('>6i', *(_gsf_round(v * 1.0e6) for v in a0_more_info))
    body += struct.pack('>6i', *(_gsf_round(v * 1.0e6) for v in a2_more_info))
    body += struct.pack('>I', _gsf_round(g('G0DepthGateMin_s', 0.0) * 1.0e6))
    body += struct.pack('>I', _gsf_round(g('G0DepthGateMax_s', 0.0) * 1.0e6))
    body += struct.pack('>i', _gsf_round(g('G0DepthGateSlope_deg', 0.0) * 1.0e6))
    body += b'\x00' * 32  # spare
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[151] = ("R2Sonic", _decode_r2sonic_specific, _encode_r2sonic_specific)
_PING_SENSOR_SPECIFIC_CODECS[152] = ("R2Sonic", _decode_r2sonic_specific, _encode_r2sonic_specific)
_PING_SENSOR_SPECIFIC_CODECS[153] = ("R2Sonic", _decode_r2sonic_specific, _encode_r2sonic_specific)


def _decode_em_run_time(payload, pos):
    """
    Decode one t_gsfEMRunTime block (Kongsberg EM-series run-time
    parameters). This is an inline sub-block, not a standalone subrecord
    -- it's used unconditionally once inside EM4_SPECIFIC (see
    _decode_em4_specific()) and, identically, inside the EM3 "_RAW"-variant
    subrecords. Ported from the run-time-parameter field reads inlined in
    gsf_dec.c's DecodeEM4Specific() (byte-for-byte duplicated there and in
    DecodeEM3RawSpecific() -- factored into a shared helper here instead).

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 63.
    """
    start = pos
    (model_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sec,) = struct.unpack_from('>I', payload, pos); pos += 4
    (nsec,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_counter,) = struct.unpack_from('>H', payload, pos); pos += 2
    (serial_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    operator_station_status = payload[pos]; pos += 1
    processing_unit_status = payload[pos]; pos += 1
    bsp_status = payload[pos]; pos += 1
    head_transceiver_status = payload[pos]; pos += 1
    mode = payload[pos]; pos += 1
    filter_id = payload[pos]; pos += 1
    (min_depth,) = struct.unpack_from('>H', payload, pos); pos += 2
    (max_depth,) = struct.unpack_from('>H', payload, pos); pos += 2
    (absorption_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tx_pulse_length,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tx_beam_width_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tx_power_re_max,) = struct.unpack_from('>b', payload, pos); pos += 1
    rx_beam_width_raw = payload[pos]; pos += 1
    rx_bandwidth_raw = payload[pos]; pos += 1
    rx_fixed_gain = payload[pos]; pos += 1
    tvg_cross_over_angle = payload[pos]; pos += 1
    ssv_source = payload[pos]; pos += 1
    (max_port_swath_width,) = struct.unpack_from('>H', payload, pos); pos += 2
    beam_spacing = payload[pos]; pos += 1
    max_port_coverage = payload[pos]; pos += 1
    stabilization = payload[pos]; pos += 1
    max_stbd_coverage = payload[pos]; pos += 1
    (max_stbd_swath_width,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tx_along_tilt_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    filter_id_2 = payload[pos]; pos += 1
    pos += 16  # spare

    fields = {
        'ModelNumber': model_number,
        'PingTime': _gsf_timestamp(sec, nsec),
        'PingCounter': ping_counter,
        'SerialNumber': serial_number,
        'OperatorStationStatus': operator_station_status,
        'ProcessingUnitStatus': processing_unit_status,
        'BspStatus': bsp_status,
        'HeadTransceiverStatus': head_transceiver_status,
        'Mode': mode,
        'FilterID': filter_id,
        'MinDepth_m': float(min_depth),
        'MaxDepth_m': float(max_depth),
        'Absorption_dBkm': absorption_raw / 100.0,
        'TxPulseLength_us': float(tx_pulse_length),
        'TxBeamWidth_deg': tx_beam_width_raw / 10.0,
        'TxPowerReMax_dB': float(tx_power_re_max),
        'RxBeamWidth_deg': rx_beam_width_raw / 10.0,
        'RxBandwidth_Hz': rx_bandwidth_raw * 50.0,
        'RxFixedGain_dB': float(rx_fixed_gain),
        'TvgCrossOverAngle_deg': float(tvg_cross_over_angle),
        'SsvSource': ssv_source,
        'MaxPortSwathWidth_m': max_port_swath_width,
        'BeamSpacing': beam_spacing,
        'MaxPortCoverage_deg': max_port_coverage,
        'Stabilization': stabilization,
        'MaxStbdCoverage_deg': max_stbd_coverage,
        'MaxStbdSwathWidth_m': max_stbd_swath_width,
        'TxAlongTilt_deg': tx_along_tilt_raw / 100.0,
        'FilterID2': filter_id_2,
    }
    return fields, pos - start


def _encode_em_run_time(fields):
    """
    Encode one t_gsfEMRunTime block: raw bytes only, no subrecord header
    (an inline sub-block of EM4_SPECIFIC/EM3 "_RAW"-variant subrecords, not
    its own subrecord). The inverse of _decode_em_run_time(). Ported from
    the run-time-parameter field writes inlined in gsf_enc.c's
    EncodeEM4Specific() (duplicated there and in EncodeEM3RawSpecific()).

    Deliberate deviation: the reference encoder writes min_depth,
    max_depth, tx_pulse_length, tx_power_re_max, and rx_fixed_gain/
    tvg_cross_over_angle via a direct truncating cast with no +/-0.501
    rounding offset (unlike every other scaled field here, which does
    round). This uses the standard _gsf_round() convention for all of
    them, consistent with every other encoder in this module.
    """
    g = fields.get
    ping_time = g('PingTime')
    sec, nsec = _gsf_epoch(ping_time) if ping_time is not None else (0, 0)

    out = struct.pack('>H', int(g('ModelNumber', 0)))
    out += struct.pack('>2I', sec, nsec)
    out += struct.pack('>H', int(g('PingCounter', 0)))
    out += struct.pack('>H', int(g('SerialNumber', 0)))
    out += struct.pack('>B', int(g('OperatorStationStatus', 0)) & 0xFF)
    out += struct.pack('>B', int(g('ProcessingUnitStatus', 0)) & 0xFF)
    out += struct.pack('>B', int(g('BspStatus', 0)) & 0xFF)
    out += struct.pack('>B', int(g('HeadTransceiverStatus', 0)) & 0xFF)
    out += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    out += struct.pack('>B', int(g('FilterID', 0)) & 0xFF)
    out += struct.pack('>H', _gsf_round(g('MinDepth_m', 0.0)))
    out += struct.pack('>H', _gsf_round(g('MaxDepth_m', 0.0)))
    out += struct.pack('>H', _gsf_round(g('Absorption_dBkm', 0.0) * 100.0))
    out += struct.pack('>H', _gsf_round(g('TxPulseLength_us', 0.0)))
    out += struct.pack('>H', _gsf_round(g('TxBeamWidth_deg', 0.0) * 10.0))
    out += struct.pack('>b', _gsf_round(g('TxPowerReMax_dB', 0.0)))
    out += struct.pack('>B', _gsf_round(g('RxBeamWidth_deg', 0.0) * 10.0) & 0xFF)
    out += struct.pack('>B', _gsf_round(g('RxBandwidth_Hz', 0.0) / 50.0) & 0xFF)
    out += struct.pack('>B', _gsf_round(g('RxFixedGain_dB', 0.0)) & 0xFF)
    out += struct.pack('>B', _gsf_round(g('TvgCrossOverAngle_deg', 0.0)) & 0xFF)
    out += struct.pack('>B', int(g('SsvSource', 0)) & 0xFF)
    out += struct.pack('>H', int(g('MaxPortSwathWidth_m', 0)))
    out += struct.pack('>B', int(g('BeamSpacing', 0)) & 0xFF)
    out += struct.pack('>B', int(g('MaxPortCoverage_deg', 0)) & 0xFF)
    out += struct.pack('>B', int(g('Stabilization', 0)) & 0xFF)
    out += struct.pack('>B', int(g('MaxStbdCoverage_deg', 0)) & 0xFF)
    out += struct.pack('>H', int(g('MaxStbdSwathWidth_m', 0)))
    out += struct.pack('>h', _gsf_round(g('TxAlongTilt_deg', 0.0) * 100.0))
    out += struct.pack('>B', int(g('FilterID2', 0)) & 0xFF)
    out += b'\x00' * 16
    return out


def _decode_em_pu_status(payload, pos):
    """
    Decode one t_gsfEMPUStatus block (Kongsberg EM-series processing-unit
    status). Like _decode_em_run_time(), this is an inline sub-block used
    unconditionally once inside EM4_SPECIFIC and the EM3 "_RAW"-variant
    subrecords, ported from field reads duplicated identically in
    gsf_dec.c's DecodeEM4Specific() and DecodeEM3RawSpecific().

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 23.
    """
    start = pos
    pu_cpu_load = payload[pos]; pos += 1
    (sensor_status,) = struct.unpack_from('>H', payload, pos); pos += 2
    (achieved_port_coverage,) = struct.unpack_from('>b', payload, pos); pos += 1
    (achieved_stbd_coverage,) = struct.unpack_from('>b', payload, pos); pos += 1
    (yaw_stabilization_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    pos += 16  # spare

    fields = {
        'PuCpuLoad_pct': float(pu_cpu_load),
        'SensorStatus': sensor_status,
        'AchievedPortCoverage_deg': achieved_port_coverage,
        'AchievedStbdCoverage_deg': achieved_stbd_coverage,
        'YawStabilization_deg': yaw_stabilization_raw / 100.0,
    }
    return fields, pos - start


def _encode_em_pu_status(fields):
    """
    Encode one t_gsfEMPUStatus block: raw bytes only, no subrecord header.
    The inverse of _decode_em_pu_status(). Ported from field writes
    duplicated identically in gsf_enc.c's EncodeEM4Specific() and
    EncodeEM3RawSpecific().

    Deliberate deviation: like _encode_em_run_time(), pu_cpu_load is
    written via a direct truncating cast with no rounding offset in the
    reference encoder -- this uses _gsf_round() instead, for consistency.
    """
    g = fields.get
    out = struct.pack('>B', _gsf_round(g('PuCpuLoad_pct', 0.0)) & 0xFF)
    out += struct.pack('>H', int(g('SensorStatus', 0)))
    out += struct.pack('>b', int(g('AchievedPortCoverage_deg', 0)))
    out += struct.pack('>b', int(g('AchievedStbdCoverage_deg', 0)))
    out += struct.pack('>h', _gsf_round(g('YawStabilization_deg', 0.0) * 100.0))
    out += b'\x00' * 16
    return out


def _decode_em4_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM4_SPECIFIC subrecord (ids 133
    EM710, 134 EM302, 135 EM122, 149 EM2040, 157 ME70BO): Kongsberg
    EM4-series per-ping sensor metadata, plus its per-transmit-sector
    array and inline run-time-parameters/PU-status blocks. Ported from
    gsf_dec.c's DecodeEM4Specific().

    Untested against a verified GSF file: no sample data containing an
    EM4_SPECIFIC subrecord is available.

    :return: (fields: dict -- the 9 top-level scalars plus every
        _decode_em_run_time()/_decode_em_pu_status() field merged in with
        a 'RunTime.'/'PuStatus.' prefix; tables: {'TxSectors':
        pandas.DataFrame}; bytes_consumed).
    """
    start = pos
    (model_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_counter,) = struct.unpack_from('>H', payload, pos); pos += 2
    (serial_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (transducer_depth_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (valid_detections,) = struct.unpack_from('>H', payload, pos); pos += 2
    (freq_int,) = struct.unpack_from('>I', payload, pos); pos += 4
    (freq_frac,) = struct.unpack_from('>I', payload, pos); pos += 4
    sampling_frequency = freq_int + freq_frac / 4.0e9
    (doppler_corr_scale,) = struct.unpack_from('>I', payload, pos); pos += 4
    (vehicle_depth_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    pos += 16  # spare_1

    (transmit_sectors,) = struct.unpack_from('>H', payload, pos); pos += 2
    transmit_sectors = min(transmit_sectors, 9)  # gsf.h: GSF_MAX_EM4_SECTORS

    sector_rows = []
    for _ in range(transmit_sectors):
        row = {}
        (tilt_angle_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
        row['TiltAngle_deg'] = tilt_angle_raw / 100.0
        (focus_range_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
        row['FocusRange_m'] = focus_range_raw / 10.0
        (signal_length_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['SignalLength_sec'] = signal_length_raw / 1.0e6
        (transmit_delay_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['TransmitDelay_sec'] = transmit_delay_raw / 1.0e6
        (center_frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['CenterFrequency_Hz'] = center_frequency_raw / 1.0e3
        (mean_absorption_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
        row['MeanAbsorption_dBkm'] = mean_absorption_raw / 100.0
        row['WaveformID'] = payload[pos]; pos += 1
        row['SectorNumber'] = payload[pos]; pos += 1
        (signal_bandwidth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['SignalBandwidth_Hz'] = signal_bandwidth_raw / 1.0e3
        pos += 16  # spare
        sector_rows.append(row)

    pos += 16  # spare_2

    run_time_fields, consumed = _decode_em_run_time(payload, pos)
    pos += consumed
    pu_status_fields, consumed = _decode_em_pu_status(payload, pos)
    pos += consumed

    fields = {
        'ModelNumber': model_number,
        'PingCounter': ping_counter,
        'SerialNumber': serial_number,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
        'TransducerDepth_m': transducer_depth_raw / 20000.0,
        'ValidDetections': valid_detections,
        'SamplingFrequency_Hz': sampling_frequency,
        'DopplerCorrScale': doppler_corr_scale,
        'VehicleDepth_m': vehicle_depth_raw / 1000.0,
    }
    fields.update({'RunTime.' + k: v for k, v in run_time_fields.items()})
    fields.update({'PuStatus.' + k: v for k, v in pu_status_fields.items()})

    sectors = pd.DataFrame(sector_rows)
    sectors.index.name = 'TxSectors'
    return fields, {'TxSectors': sectors}, pos - start


def _encode_em4_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM4_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_em4_specific(). Ported from gsf_enc.c's EncodeEM4Specific().

    :param tables: optional {'TxSectors': pandas.DataFrame}; transmit_sectors
        on the wire is len() of that table -- the exact DataFrame
        _decode_em4_specific() returns can be passed straight back in, no
        conversion required.
    """
    g = fields.get
    tables = tables or {}
    sectors = tables.get('TxSectors')
    if sectors is None:
        sectors = pd.DataFrame()

    body = struct.pack('>H', int(g('ModelNumber', 0)))
    body += struct.pack('>H', int(g('PingCounter', 0)))
    body += struct.pack('>H', int(g('SerialNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>i', _gsf_round(g('TransducerDepth_m', 0.0) * 20000.0))
    body += struct.pack('>H', int(g('ValidDetections', 0)))

    sampling_frequency = g('SamplingFrequency_Hz', 0.0)
    freq_int = int(sampling_frequency)
    freq_frac = _gsf_round((sampling_frequency - freq_int) * 4.0e9)
    body += struct.pack('>I', freq_int)
    body += struct.pack('>I', freq_frac)

    body += struct.pack('>I', int(g('DopplerCorrScale', 0)))
    body += struct.pack('>i', _gsf_round(g('VehicleDepth_m', 0.0) * 1000.0))
    body += b'\x00' * 16  # spare_1

    body += struct.pack('>H', len(sectors))
    for _, row in sectors.iterrows():
        r = row.get
        body += struct.pack('>h', _gsf_round(r('TiltAngle_deg', 0.0) * 100.0))
        body += struct.pack('>H', _gsf_round(r('FocusRange_m', 0.0) * 10.0))
        body += struct.pack('>I', _gsf_round(r('SignalLength_sec', 0.0) * 1.0e6))
        body += struct.pack('>I', _gsf_round(r('TransmitDelay_sec', 0.0) * 1.0e6))
        body += struct.pack('>I', _gsf_round(r('CenterFrequency_Hz', 0.0) * 1.0e3))
        body += struct.pack('>H', _gsf_round(r('MeanAbsorption_dBkm', 0.0) * 100.0))
        body += struct.pack('>B', int(r('WaveformID', 0)) & 0xFF)
        body += struct.pack('>B', int(r('SectorNumber', 0)) & 0xFF)
        body += struct.pack('>I', _gsf_round(r('SignalBandwidth_Hz', 0.0) * 1.0e3))
        body += b'\x00' * 16  # spare

    body += b'\x00' * 16  # spare_2

    run_time_fields = {k[len('RunTime.'):]: v for k, v in fields.items() if k.startswith('RunTime.')}
    pu_status_fields = {k[len('PuStatus.'):]: v for k, v in fields.items() if k.startswith('PuStatus.')}
    body += _encode_em_run_time(run_time_fields)
    body += _encode_em_pu_status(pu_status_fields)

    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[133] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[134] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[135] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[149] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[157] = ("EM4", _decode_em4_specific, _encode_em4_specific)


def _decode_em3raw_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_RAW_SPECIFIC subrecord (ids
    140-148: EM300/1002/2000/3000/120/3002/3000D/3002D/121A_SIS, each with
    raw range and beam angle data): Kongsberg EM3-series per-ping sensor
    metadata, plus its per-transmit-sector array and inline
    run-time-parameters/PU-status blocks (reusing _decode_em_run_time()/
    _decode_em_pu_status(), which EM4_SPECIFIC also uses -- gsf.h defines
    both subrecords using the same t_gsfEMRunTime/t_gsfEMPUStatus struct
    types by value). Ported from gsf_dec.c's DecodeEM3RawSpecific().

    Field order and scaling here differ from EM4_SPECIFIC in several
    places despite the superficially similar shape -- verified against
    DecodeEM3RawSpecific() directly rather than assumed from EM4: no
    doppler_corr_scale field; vehicle_depth precedes depth_difference
    (EM4 has no depth_difference at all); offset_multiplier is a signed
    byte; the TX sector struct has no mean_absorption field (8 fields,
    not EM4's 9); and the max sector count is 20 (GSF_MAX_EM3_SECTORS),
    not EM4's 9 (GSF_MAX_EM4_SECTORS).

    Untested against a verified GSF file: no sample data containing an
    EM3xxx_RAW_SPECIFIC subrecord is available.

    :return: (fields: dict -- the 10 top-level scalars plus every
        _decode_em_run_time()/_decode_em_pu_status() field merged in with
        a 'RunTime.'/'PuStatus.' prefix; tables: {'TxSectors':
        pandas.DataFrame}; bytes_consumed).
    """
    start = pos
    (model_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_counter,) = struct.unpack_from('>H', payload, pos); pos += 2
    (serial_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (transducer_depth_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (valid_detections,) = struct.unpack_from('>H', payload, pos); pos += 2
    (freq_int,) = struct.unpack_from('>I', payload, pos); pos += 4
    (freq_frac,) = struct.unpack_from('>I', payload, pos); pos += 4
    sampling_frequency = freq_int + freq_frac / 4.0e9
    (vehicle_depth_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (depth_difference_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (offset_multiplier,) = struct.unpack_from('>b', payload, pos); pos += 1
    pos += 16  # spare_1

    (transmit_sectors,) = struct.unpack_from('>H', payload, pos); pos += 2
    transmit_sectors = min(transmit_sectors, 20)  # gsf.h: GSF_MAX_EM3_SECTORS

    sector_rows = []
    for _ in range(transmit_sectors):
        row = {}
        (tilt_angle_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
        row['TiltAngle_deg'] = tilt_angle_raw / 100.0
        (focus_range_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
        row['FocusRange_m'] = focus_range_raw / 10.0
        (signal_length_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['SignalLength_sec'] = signal_length_raw / 1.0e6
        (transmit_delay_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['TransmitDelay_sec'] = transmit_delay_raw / 1.0e6
        (center_frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['CenterFrequency_Hz'] = center_frequency_raw / 1.0e3
        row['WaveformID'] = payload[pos]; pos += 1
        row['SectorNumber'] = payload[pos]; pos += 1
        (signal_bandwidth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
        row['SignalBandwidth_Hz'] = signal_bandwidth_raw / 1.0e3
        pos += 16  # spare
        sector_rows.append(row)

    pos += 16  # spare_2

    run_time_fields, consumed = _decode_em_run_time(payload, pos)
    pos += consumed
    pu_status_fields, consumed = _decode_em_pu_status(payload, pos)
    pos += consumed

    fields = {
        'ModelNumber': model_number,
        'PingCounter': ping_counter,
        'SerialNumber': serial_number,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
        'TransducerDepth_m': transducer_depth_raw / 20000.0,
        'ValidDetections': valid_detections,
        'SamplingFrequency_Hz': sampling_frequency,
        'VehicleDepth_m': vehicle_depth_raw / 1000.0,
        'DepthDifference_m': depth_difference_raw / 100.0,
        'OffsetMultiplier': offset_multiplier,
    }
    fields.update({'RunTime.' + k: v for k, v in run_time_fields.items()})
    fields.update({'PuStatus.' + k: v for k, v in pu_status_fields.items()})

    sectors = pd.DataFrame(sector_rows)
    sectors.index.name = 'TxSectors'
    return fields, {'TxSectors': sectors}, pos - start


def _encode_em3raw_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_RAW_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_em3raw_specific(). Ported from gsf_enc.c's
    EncodeEM3RawSpecific().

    :param tables: optional {'TxSectors': pandas.DataFrame}; transmit_sectors
        on the wire is len() of that table -- the exact DataFrame
        _decode_em3raw_specific() returns can be passed straight back in, no
        conversion required.
    """
    g = fields.get
    tables = tables or {}
    sectors = tables.get('TxSectors')
    if sectors is None:
        sectors = pd.DataFrame()

    body = struct.pack('>H', int(g('ModelNumber', 0)))
    body += struct.pack('>H', int(g('PingCounter', 0)))
    body += struct.pack('>H', int(g('SerialNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>i', _gsf_round(g('TransducerDepth_m', 0.0) * 20000.0))
    body += struct.pack('>H', int(g('ValidDetections', 0)))

    sampling_frequency = g('SamplingFrequency_Hz', 0.0)
    freq_int = int(sampling_frequency)
    freq_frac = _gsf_round((sampling_frequency - freq_int) * 4.0e9)
    body += struct.pack('>I', freq_int)
    body += struct.pack('>I', freq_frac)

    body += struct.pack('>i', _gsf_round(g('VehicleDepth_m', 0.0) * 1000.0))
    body += struct.pack('>h', _gsf_round(g('DepthDifference_m', 0.0) * 100.0))
    body += struct.pack('>b', int(g('OffsetMultiplier', 0)))
    body += b'\x00' * 16  # spare_1

    body += struct.pack('>H', len(sectors))
    for _, row in sectors.iterrows():
        r = row.get
        body += struct.pack('>h', _gsf_round(r('TiltAngle_deg', 0.0) * 100.0))
        body += struct.pack('>H', _gsf_round(r('FocusRange_m', 0.0) * 10.0))
        body += struct.pack('>I', _gsf_round(r('SignalLength_sec', 0.0) * 1.0e6))
        body += struct.pack('>I', _gsf_round(r('TransmitDelay_sec', 0.0) * 1.0e6))
        body += struct.pack('>I', _gsf_round(r('CenterFrequency_Hz', 0.0) * 1.0e3))
        body += struct.pack('>B', int(r('WaveformID', 0)) & 0xFF)
        body += struct.pack('>B', int(r('SectorNumber', 0)) & 0xFF)
        body += struct.pack('>I', _gsf_round(r('SignalBandwidth_Hz', 0.0) * 1.0e3))
        body += b'\x00' * 16  # spare

    body += b'\x00' * 16  # spare_2

    run_time_fields = {k[len('RunTime.'):]: v for k, v in fields.items() if k.startswith('RunTime.')}
    pu_status_fields = {k[len('PuStatus.'):]: v for k, v in fields.items() if k.startswith('PuStatus.')}
    body += _encode_em_run_time(run_time_fields)
    body += _encode_em_pu_status(pu_status_fields)

    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


for _em3raw_id in (140, 141, 142, 143, 144, 145, 146, 147, 148):
    _PING_SENSOR_SPECIFIC_CODECS[_em3raw_id] = ("EM3Raw", _decode_em3raw_specific, _encode_em3raw_specific)
del _em3raw_id


def _decode_em3_run_time(payload, pos):
    """
    Decode one gsfEM3RunTime block (the OLDER, plain-EM3-specific
    run-time-parameters struct -- distinct from t_gsfEMRunTime/
    _decode_em_run_time(), which is used by EM4_SPECIFIC and the EM3
    "_RAW"-variant subrecords instead). This is an inline sub-block, used
    0, 1, or 2 times depending on EM3_SPECIFIC's run_time_id bitmask (see
    _decode_em3_specific()), not a standalone subrecord. Ported from the
    run-time-parameter field reads inlined in gsf_dec.c's
    DecodeEM3Specific() (duplicated there once per head).

    Also computes SwathWidth_m/CoverageSector_deg, matching gsf_dec.c's
    own post-decode derivation: derived from port/stbd swath width (resp.
    coverage sector) alone if stbd is 0 (the total is then split evenly
    back into port/stbd), or their sum otherwise. These two fields exist
    for read-side convenience only -- there is no separate wire storage
    for them, and _encode_em3_run_time() ignores them entirely.

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 49.
    """
    start = pos
    (model_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sec,) = struct.unpack_from('>I', payload, pos); pos += 4
    (nsec,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (serial_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (system_status,) = struct.unpack_from('>I', payload, pos); pos += 4
    mode = payload[pos]; pos += 1
    filter_id = payload[pos]; pos += 1
    (min_depth,) = struct.unpack_from('>H', payload, pos); pos += 2
    (max_depth,) = struct.unpack_from('>H', payload, pos); pos += 2
    (absorption_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (pulse_length,) = struct.unpack_from('>H', payload, pos); pos += 2
    (transmit_beam_width_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    power_reduction = payload[pos]; pos += 1
    receive_beam_width_raw = payload[pos]; pos += 1
    receive_bandwidth_raw = payload[pos]; pos += 1
    receive_gain = payload[pos]; pos += 1
    cross_over_angle = payload[pos]; pos += 1
    ssv_source = payload[pos]; pos += 1
    (port_swath_width,) = struct.unpack_from('>H', payload, pos); pos += 2
    beam_spacing = payload[pos]; pos += 1
    port_coverage_sector = payload[pos]; pos += 1
    stabilization = payload[pos]; pos += 1
    stbd_coverage_sector = payload[pos]; pos += 1
    (stbd_swath_width,) = struct.unpack_from('>H', payload, pos); pos += 2
    hilo_freq_absorp_ratio = payload[pos]; pos += 1
    pos += 4  # spare1

    if stbd_swath_width:
        swath_width = port_swath_width + stbd_swath_width
    else:
        swath_width = port_swath_width
        port_swath_width = swath_width // 2
        stbd_swath_width = swath_width // 2

    if stbd_coverage_sector:
        coverage_sector = port_coverage_sector + stbd_coverage_sector
    else:
        coverage_sector = port_coverage_sector
        port_coverage_sector = coverage_sector // 2
        stbd_coverage_sector = coverage_sector // 2

    fields = {
        'ModelNumber': model_number,
        'PingTime': _gsf_timestamp(sec, nsec),
        'PingNumber': ping_number,
        'SerialNumber': serial_number,
        'SystemStatus': system_status,
        'Mode': mode,
        'FilterID': filter_id,
        'MinDepth_m': float(min_depth),
        'MaxDepth_m': float(max_depth),
        'Absorption_dBkm': absorption_raw / 100.0,
        'PulseLength_us': float(pulse_length),
        'TransmitBeamWidth_deg': transmit_beam_width_raw / 10.0,
        'PowerReduction_dB': power_reduction,
        'ReceiveBeamWidth_deg': receive_beam_width_raw / 10.0,
        'ReceiveBandwidth_Hz': receive_bandwidth_raw * 50,
        'ReceiveGain_dB': receive_gain,
        'CrossOverAngle_deg': cross_over_angle,
        'SsvSource': ssv_source,
        'PortSwathWidth_m': port_swath_width,
        'BeamSpacing': beam_spacing,
        'PortCoverageSector_deg': port_coverage_sector,
        'Stabilization': stabilization,
        'StbdCoverageSector_deg': stbd_coverage_sector,
        'StbdSwathWidth_m': stbd_swath_width,
        'HiloFreqAbsorpRatio': hilo_freq_absorp_ratio,
        'SwathWidth_m': swath_width,
        'CoverageSector_deg': coverage_sector,
    }
    return fields, pos - start


def _encode_em3_run_time(fields):
    """
    Encode one gsfEM3RunTime block: raw bytes only, no subrecord header
    (an inline sub-block of EM3_SPECIFIC, not its own subrecord). The
    inverse of _decode_em3_run_time() -- except SwathWidth_m/
    CoverageSector_deg, which are derived read-only fields with no wire
    storage of their own (see _decode_em3_run_time()'s docstring) and are
    ignored here; only PortSwathWidth_m/StbdSwathWidth_m and
    PortCoverageSector_deg/StbdCoverageSector_deg are written. Ported from
    the run-time-parameter field writes inlined in gsf_enc.c's
    EncodeEM3Specific() (duplicated there once per head).

    Deliberate deviation: the reference encoder writes min_depth,
    max_depth, and pulse_length via a direct truncating cast with no
    +/-0.501 rounding offset (unlike every other scaled field here, which
    does round). This uses the standard _gsf_round() convention for all
    of them, consistent with every other encoder in this module.
    """
    g = fields.get
    ping_time = g('PingTime')
    sec, nsec = _gsf_epoch(ping_time) if ping_time is not None else (0, 0)

    out = struct.pack('>H', int(g('ModelNumber', 0)))
    out += struct.pack('>2I', sec, nsec)
    out += struct.pack('>H', int(g('PingNumber', 0)))
    out += struct.pack('>H', int(g('SerialNumber', 0)))
    out += struct.pack('>I', int(g('SystemStatus', 0)))
    out += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    out += struct.pack('>B', int(g('FilterID', 0)) & 0xFF)
    out += struct.pack('>H', _gsf_round(g('MinDepth_m', 0.0)))
    out += struct.pack('>H', _gsf_round(g('MaxDepth_m', 0.0)))
    out += struct.pack('>H', _gsf_round(g('Absorption_dBkm', 0.0) * 100.0))
    out += struct.pack('>H', _gsf_round(g('PulseLength_us', 0.0)))
    out += struct.pack('>H', _gsf_round(g('TransmitBeamWidth_deg', 0.0) * 10.0))
    out += struct.pack('>B', int(g('PowerReduction_dB', 0)) & 0xFF)
    out += struct.pack('>B', _gsf_round(g('ReceiveBeamWidth_deg', 0.0) * 10.0) & 0xFF)
    out += struct.pack('>B', _gsf_round(g('ReceiveBandwidth_Hz', 0.0) / 50.0) & 0xFF)
    out += struct.pack('>B', int(g('ReceiveGain_dB', 0)) & 0xFF)
    out += struct.pack('>B', int(g('CrossOverAngle_deg', 0)) & 0xFF)
    out += struct.pack('>B', int(g('SsvSource', 0)) & 0xFF)
    out += struct.pack('>H', int(g('PortSwathWidth_m', 0)))
    out += struct.pack('>B', int(g('BeamSpacing', 0)) & 0xFF)
    out += struct.pack('>B', int(g('PortCoverageSector_deg', 0)) & 0xFF)
    out += struct.pack('>B', int(g('Stabilization', 0)) & 0xFF)
    out += struct.pack('>B', int(g('StbdCoverageSector_deg', 0)) & 0xFF)
    out += struct.pack('>H', int(g('StbdSwathWidth_m', 0)))
    out += struct.pack('>B', int(g('HiloFreqAbsorpRatio', 0)) & 0xFF)
    out += b'\x00' * 4
    return out


def _decode_em3_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_SPECIFIC subrecord (the
    plain, non-"_RAW" EM3-series family: EM3000/EM1002/EM300/EM120/
    EM3002/EM3000D/EM3002D/EM121A_SIS/EM2000). Ported from gsf_dec.c's
    DecodeEM3Specific().

    Unlike every other ping-level sensor-specific subrecord in this
    module, this one is genuinely variable-length: after the 17 fixed
    bytes + a 4-byte run_time_id bitmask, bit 0 (0x1) gates whether a
    run-time-parameters block for head 0 follows. Bit 1 (0x2) gates a
    SECOND block for head 1 (EM3000D dual-head), but -- matching
    DecodeEM3Specific()'s own nesting exactly -- bit 1 is only even
    inspected, and a head-1 block only ever decoded, when bit 0 is also
    set; a file with bit 1 set but bit 0 clear has no run-time blocks at
    all on the wire (there is nothing there to read a head-1 block from).
    Each present block is decoded by _decode_em3_run_time() and returned
    as a row (with a 'Head' column, 0 or 1) in tables['RunTime'] -- an
    empty DataFrame if bit 0 is clear.

    Untested against a verified GSF file: no sample data containing an
    EM3_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {'RunTime': pandas.DataFrame}, bytes_consumed).
    """
    start = pos
    (model_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (serial_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (transducer_depth_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (valid_beams,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sample_rate,) = struct.unpack_from('>H', payload, pos); pos += 2
    (depth_difference_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (offset_multiplier,) = struct.unpack_from('>b', payload, pos); pos += 1
    (run_time_id,) = struct.unpack_from('>I', payload, pos); pos += 4

    fields = {
        'ModelNumber': model_number,
        'PingNumber': ping_number,
        'SerialNumber': serial_number,
        'SurfaceVelocity_mps': surface_velocity_raw / 10.0,
        'TransducerDepth_m': transducer_depth_raw / 100.0,
        'ValidBeams': valid_beams,
        'SampleRate_Hz': sample_rate,
        'DepthDifference_m': depth_difference_raw / 100.0,
        'OffsetMultiplier': offset_multiplier,
    }

    run_time_rows = []
    if run_time_id & 0x1:
        head0_fields, consumed = _decode_em3_run_time(payload, pos)
        pos += consumed
        run_time_rows.append({'Head': 0, **head0_fields})

        if run_time_id & 0x2:
            head1_fields, consumed = _decode_em3_run_time(payload, pos)
            pos += consumed
            run_time_rows.append({'Head': 1, **head1_fields})

    run_time = pd.DataFrame(run_time_rows)
    run_time.index.name = 'RunTime'
    return fields, {'RunTime': run_time}, pos - start


def _encode_em3_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_SPECIFIC subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_em3_specific(). Ported from gsf_enc.c's EncodeEM3Specific(),
    with one deliberate improvement: the reference encoder as currently
    shipped hardcodes run_time_id = 1 unconditionally (the code path that
    would set bit 1 for an EM3000D dual-head run-time update is entirely
    commented out / dead in gsf_enc.c -- a real limitation of gsflib
    itself). Since the wire format and DecodeEM3Specific() both fully
    support 0 or 1 head-0-only or both-heads run-time blocks, this
    encoder writes exactly the blocks the caller supplies via
    tables['RunTime'] instead of always forcing exactly one.

    :param tables: optional {'RunTime': pandas.DataFrame}; each row needs a
        'Head' column (0 or 1) selecting which position it's written at --
        the exact DataFrame _decode_em3_specific() returns can be passed
        straight back in, no conversion required. Bit 0 of the on-disk
        run_time_id is set iff a Head==0 row is present, bit 1 iff a
        Head==1 row is present. A Head==1 row with no matching Head==0 row
        is rejected (see below) -- matching DecodeEM3Specific()'s nesting
        (bit 1 only means anything, and a head-1 block is only ever
        written, alongside a head-0 block; an unpaired head-1-only row
        could not be read back by this decoder, or by real gsflib). An
        empty/absent table writes run_time_id = 0 (no run-time blocks at
        all).
    :raises ValueError: a row's 'Head' isn't 0 or 1; the same Head
        appears more than once; or a Head==1 row is present without a
        matching Head==0 row.
    """
    g = fields.get
    tables = tables or {}
    run_time_df = tables.get('RunTime')
    if run_time_df is None:
        run_time_df = pd.DataFrame()
    rows_by_head = {}
    for _, row in run_time_df.iterrows():
        head = row.get('Head')
        if head not in (0, 1):
            raise ValueError("EM3 RunTime row 'Head' must be 0 or 1, got %r" % (head,))
        if head in rows_by_head:
            raise ValueError("EM3 RunTime: more than one row with Head=%d" % head)
        rows_by_head[head] = row
    if 1 in rows_by_head and 0 not in rows_by_head:
        raise ValueError(
            "EM3 RunTime: a Head=1 row requires a matching Head=0 row "
            "(the wire format can't represent head 1 alone)")

    body = struct.pack('>H', int(g('ModelNumber', 0)))
    body += struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', int(g('SerialNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>H', _gsf_round(g('TransducerDepth_m', 0.0) * 100.0))
    body += struct.pack('>H', int(g('ValidBeams', 0)))
    body += struct.pack('>H', int(g('SampleRate_Hz', 0)))
    body += struct.pack('>h', _gsf_round(g('DepthDifference_m', 0.0) * 100.0))
    body += struct.pack('>b', int(g('OffsetMultiplier', 0)))

    run_time_id = (0x1 if 0 in rows_by_head else 0) | (0x2 if 1 in rows_by_head else 0)
    body += struct.pack('>I', run_time_id)
    if 0 in rows_by_head:
        body += _encode_em3_run_time(rows_by_head[0])
    if 1 in rows_by_head:
        body += _encode_em3_run_time(rows_by_head[1])

    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


for _em3_id in (118, 119, 120, 128, 129, 130, 131, 132, 139):
    _PING_SENSOR_SPECIFIC_CODECS[_em3_id] = ("EM3", _decode_em3_specific, _encode_em3_specific)
del _em3_id


#: Registry of GSF_RECORD_SINGLE_BEAM_PING sensor-specific subrecord codecs,
#: keyed by gsf.h's GSF_SINGLE_BEAM_SUBRECORD_* id (a separate id
#: namespace, 201-205, from the swath-ping GSF_SWATH_BATHY_SUBRECORD_* ids
#: above -- no overlap). Same {subrecord_id: (family label, decode_fn,
#: encode_fn)} shape as _PING_SENSOR_SPECIFIC_CODECS; see that constant's
#: comment for the decode_fn/encode_fn contract.
_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS = {}

#: gsf.h GSF_SINGLE_BEAM_SUBRECORD_* id -> name, for the "not decoded yet"
#: fallback note (mirrors _SENSOR_SPECIFIC_SUBRECORD_NAMES for swath pings).
#: gsf.h also defines t_gsfSBPDDSpecific/t_gsfSBNavisoundSpecific structs,
#: but gsf_dec.c's gsfDecodeSinglebeam() switch has no case that ever
#: reaches them (no GSF_SINGLE_BEAM_SUBRECORD_PDD_SPECIFIC/_NAVISOUND_SPECIFIC
#: id is defined), so they're omitted here -- they cannot occur in a real
#: GSF file.
_SINGLE_BEAM_SENSOR_SPECIFIC_NAMES = {
    201: "ECHOTRAC_SPECIFIC",
    202: "BATHY2000_SPECIFIC",
    203: "MGD77_SPECIFIC",
    204: "BDB_SPECIFIC",
    205: "NOSHDB_SPECIFIC",
}


def _decode_echotrac_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_ECHOTRAC_SPECIFIC or
    _BATHY2000_SPECIFIC subrecord (ids 201/202 -- identical wire format,
    one struct/decoder shared by both in gsflib). Ported from gsf_dec.c's
    DecodeEchotracSpecific().

    Untested against a verified GSF file: no sample data containing an
    ECHOTRAC_SPECIFIC/BATHY2000_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (navigation_error,) = struct.unpack_from('>h', payload, pos); pos += 2
    mpp_source = payload[pos]; pos += 1
    tide_source = payload[pos]; pos += 1

    fields = {
        'NavigationError': navigation_error,
        'MppSource': mpp_source,
        'TideSource': tide_source,
    }
    return fields, {}, pos - start


def _encode_echotrac_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_ECHOTRAC_SPECIFIC or
    _BATHY2000_SPECIFIC subrecord (subrecord_id selects which -- see
    _decode_echotrac_specific()), including its own 4-byte subrecord
    id+size word: the inverse of _decode_echotrac_specific(). Ported from
    gsf_enc.c's EncodeEchotracSpecific().
    """
    g = fields.get
    body = struct.pack('>h', int(g('NavigationError', 0)))
    body += struct.pack('>B', int(g('MppSource', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TideSource', 0)) & 0xFF)
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[201] = ("Echotrac", _decode_echotrac_specific, _encode_echotrac_specific)
_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[202] = ("Bathy2000", _decode_echotrac_specific, _encode_echotrac_specific)


def _decode_mgd77_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_MGD77_SPECIFIC subrecord (id 203):
    MGD77 survey trackline data. Ported from gsf_dec.c's
    DecodeMGD77Specific().

    Untested against a verified GSF file: no sample data containing an
    MGD77_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (time_zone_corr,) = struct.unpack_from('>H', payload, pos); pos += 2
    (position_type_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (correction_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (bathy_type_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (quality_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (travel_time_raw,) = struct.unpack_from('>I', payload, pos); pos += 4

    fields = {
        'TimeZoneCorr': time_zone_corr,
        'PositionTypeCode': position_type_code,
        'CorrectionCode': correction_code,
        'BathyTypeCode': bathy_type_code,
        'QualityCode': quality_code,
        'TravelTime_sec': travel_time_raw / 10000.0,
    }
    return fields, {}, pos - start


def _encode_mgd77_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_MGD77_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_mgd77_specific(). Ported from gsf_enc.c's EncodeMGD77Specific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('TimeZoneCorr', 0)))
    body += struct.pack('>H', int(g('PositionTypeCode', 0)))
    body += struct.pack('>H', int(g('CorrectionCode', 0)))
    body += struct.pack('>H', int(g('BathyTypeCode', 0)))
    body += struct.pack('>H', int(g('QualityCode', 0)))
    body += struct.pack('>I', _gsf_round(float(g('TravelTime_sec', 0.0)) * 10000.0))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[203] = ("MGD77", _decode_mgd77_specific, _encode_mgd77_specific)


def _decode_bdb_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_BDB_SPECIFIC subrecord (id 204):
    BDB survey trackline data. Ported from gsf_dec.c's DecodeBDBSpecific().
    Every flag field here is a single ASCII character on disk (per gsf.h's
    field comments, e.g. eval is '1'-'4', datum_flag is 'W' or 'D'), so
    each is decoded to a one-character str rather than a raw int.

    Untested against a verified GSF file: no sample data containing a
    BDB_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (doc_no,) = struct.unpack_from('>i', payload, pos); pos += 4
    eval_flag = payload[pos:pos + 1].decode('ascii', 'replace'); pos += 1
    classification = payload[pos:pos + 1].decode('ascii', 'replace'); pos += 1
    track_adj_flag = payload[pos:pos + 1].decode('ascii', 'replace'); pos += 1
    source_flag = payload[pos:pos + 1].decode('ascii', 'replace'); pos += 1
    pt_or_track_ln = payload[pos:pos + 1].decode('ascii', 'replace'); pos += 1
    datum_flag = payload[pos:pos + 1].decode('ascii', 'replace'); pos += 1

    fields = {
        'DocNo': doc_no,
        'Eval': eval_flag,
        'Classification': classification,
        'TrackAdjFlag': track_adj_flag,
        'SourceFlag': source_flag,
        'PtOrTrackLn': pt_or_track_ln,
        'DatumFlag': datum_flag,
    }
    return fields, {}, pos - start


def _encode_bdb_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_BDB_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_bdb_specific(). Ported from gsf_enc.c's EncodeBDBSpecific().
    Each flag field is written as the first byte of the given string (or
    NUL if omitted/empty), matching the single-ASCII-character on-disk format.
    """
    def _char(value):
        text = str(value) if value else '\x00'
        return text.encode('ascii')[:1]

    g = fields.get
    body = struct.pack('>i', int(g('DocNo', 0)))
    body += _char(g('Eval', ''))
    body += _char(g('Classification', ''))
    body += _char(g('TrackAdjFlag', ''))
    body += _char(g('SourceFlag', ''))
    body += _char(g('PtOrTrackLn', ''))
    body += _char(g('DatumFlag', ''))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[204] = ("BDB", _decode_bdb_specific, _encode_bdb_specific)


def _decode_noshdb_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_NOSHDB_SPECIFIC subrecord (id 205):
    NOS HDB survey trackline data. Ported from gsf_dec.c's
    DecodeNOSHDBSpecific().

    Untested against a verified GSF file: no sample data containing a
    NOSHDB_SPECIFIC subrecord is available.

    :return: (fields: dict, tables: {} (none for this sensor), bytes_consumed).
    """
    start = pos
    (type_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (carto_code,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = {
        'TypeCode': type_code,
        'CartoCode': carto_code,
    }
    return fields, {}, pos - start


def _encode_noshdb_specific(subrecord_id, fields, tables=None):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_NOSHDB_SPECIFIC subrecord, including
    its own 4-byte subrecord id+size word: the inverse of
    _decode_noshdb_specific(). Ported from gsf_enc.c's EncodeNOSHDBSpecific().
    """
    g = fields.get
    body = struct.pack('>H', int(g('TypeCode', 0)))
    body += struct.pack('>H', int(g('CartoCode', 0)))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[205] = ("NOSHDB", _decode_noshdb_specific, _encode_noshdb_specific)


#: gsf.h GSF_SWATH_BATHY_SUBRECORD_*_SPECIFIC ids sharing
#: DecodeEM3ImagerySpecific()'s 18-byte sensor-imagery preamble: older
#: Simrad/Kongsberg EM3-series sonars, normal and "_RAW" range/angle variants.
_SUBRECORD_EM3_IMAGERY_IDS = {
    118, 119, 120, 128, 129, 130, 131, 132, 139,
    140, 141, 142, 143, 144, 145, 146, 147, 148,
}
#: ids sharing DecodeEM4ImagerySpecific()'s 50-byte preamble: EM122, EM302,
#: EM710, EM2040, ME70BO.
_SUBRECORD_EM4_IMAGERY_IDS = {133, 134, 135, 149, 157}
#: ids sharing the identical 66-byte "size + spare" preamble used by Reson
#: 7125 (DecodeReson7100ImagerySpecific()) and Reson T-series
#: (DecodeResonTSeriesImagerySpecific()).
_SUBRECORD_RESON_SIZE_SPARE_IMAGERY_IDS = {138, 155}
#: ids sharing DecodeReson8100ImagerySpecific()'s 8-byte, all-spare preamble.
_SUBRECORD_RESON_8100_IMAGERY_IDS = {122, 123, 124, 125, 126, 127}
#: ids sharing DecodeR2SonicImagerySpecific()'s 168-byte preamble.
_SUBRECORD_R2SONIC_IMAGERY_IDS = {151, 152, 153}
_SUBRECORD_KLEIN_5410_BSS_SPECIFIC = 137


def _decode_kmall_imagery_specific(payload, pos):
    """
    Decode the KMALL sensor-specific portion of a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord. Ported from
    gsf_dec.c's DecodeKMALLImagerySpecific(): entirely spare/reserved for
    the KMALL sensor, so this exists only to advance past its 64 bytes.

    :return: bytes_consumed (always 64).
    """
    return 64


def _decode_em3_imagery_specific(payload, pos):
    """
    Decode the EM3-series sensor-specific portion of a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (older
    Simrad/Kongsberg EM3000/EM1002/EM300/EM120/EM3002/EM3000D/EM3002D/
    EM121A-SIS/EM2000, and their "_RAW" range/angle variants). Ported from
    gsf_dec.c's DecodeEM3ImagerySpecific().

    Untested against a verified GSF file: no sample data containing an
    EM3-series intensity series subrecord is available.

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 18.
    """
    start = pos
    (range_norm,) = struct.unpack_from('>H', payload, pos); pos += 2
    (start_tvg_ramp,) = struct.unpack_from('>H', payload, pos); pos += 2
    (stop_tvg_ramp,) = struct.unpack_from('>H', payload, pos); pos += 2
    bsn = payload[pos]; pos += 1
    bso = payload[pos]; pos += 1
    (mean_absorption_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (offset,) = struct.unpack_from('>h', payload, pos); pos += 2
    (scale,) = struct.unpack_from('>h', payload, pos); pos += 2
    pos += 4  # spare

    fields = {
        'RangeNorm_samples': range_norm,
        'StartTvgRamp_samples': start_tvg_ramp,
        'StopTvgRamp_samples': stop_tvg_ramp,
        'BSNormal_dB': bsn,
        'BSOblique_dB': bso,
        'MeanAbsorption_dBkm': mean_absorption_raw / 100.0,
        'Offset': offset,
        'Scale': scale,
    }
    return fields, pos - start


def _decode_em4_imagery_specific(payload, pos):
    """
    Decode the EM4-series sensor-specific portion of a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (EM122,
    EM302, EM710, EM2040, ME70BO). Ported from gsf_dec.c's
    DecodeEM4ImagerySpecific().

    Untested against a verified GSF file: none of the checked-in sample
    files carry an intensity series subrecord for these sensors (the
    EM712 sample data is all KMALL_SPECIFIC, which uses the separate
    Kongsberg SIS 5 imagery format decoded by
    _decode_kmall_imagery_specific()).

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 50.
    """
    start = pos
    (freq_int,) = struct.unpack_from('>I', payload, pos); pos += 4
    (freq_frac,) = struct.unpack_from('>I', payload, pos); pos += 4
    sampling_frequency = freq_int + freq_frac / 4.0e9
    (mean_absorption_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tx_pulse_length,) = struct.unpack_from('>H', payload, pos); pos += 2
    (range_norm,) = struct.unpack_from('>H', payload, pos); pos += 2
    (start_tvg_ramp,) = struct.unpack_from('>H', payload, pos); pos += 2
    (stop_tvg_ramp,) = struct.unpack_from('>H', payload, pos); pos += 2
    (bsn_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (bso_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (tx_beam_width_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tvg_cross_over_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (offset,) = struct.unpack_from('>h', payload, pos); pos += 2
    (scale,) = struct.unpack_from('>h', payload, pos); pos += 2
    pos += 20  # spare

    fields = {
        'SamplingFrequency_Hz': sampling_frequency,
        'MeanAbsorption_dBkm': mean_absorption_raw / 100.0,
        'TxPulseLength_us': tx_pulse_length,
        'RangeNorm_samples': range_norm,
        'StartTvgRamp_samples': start_tvg_ramp,
        'StopTvgRamp_samples': stop_tvg_ramp,
        'BSNormal_dB': bsn_raw / 10.0,
        'BSOblique_dB': bso_raw / 10.0,
        'TxBeamWidth_deg': tx_beam_width_raw / 10.0,
        'TvgCrossOver_deg': tvg_cross_over_raw / 10.0,
        'Offset': offset,
        'Scale': scale,
    }
    return fields, pos - start


def _decode_reson_size_spare_imagery_specific(payload, pos):
    """
    Decode the 66-byte sensor-specific preamble shared by Reson 7125 and
    Reson T-series (a 2-byte record size followed by 64 spare bytes).
    Ported from gsf_dec.c's DecodeReson7100ImagerySpecific() /
    DecodeResonTSeriesImagerySpecific(), which are byte-for-byte identical
    apart from name.

    Untested against a verified GSF file: no sample data containing a
    Reson 7125 or Reson T-series intensity series subrecord is available.

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 66.
    """
    start = pos
    (size,) = struct.unpack_from('>H', payload, pos); pos += 2
    pos += 64  # spare
    return {'Size': size}, pos - start


def _decode_reson8100_imagery_specific(payload, pos):
    """
    Decode the Reson 8100-family sensor-specific portion of a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (8101/8111/
    8124/8125/8150/8160): entirely spare/reserved. Ported from gsf_dec.c's
    DecodeReson8100ImagerySpecific().

    Untested against a verified GSF file: no sample data containing a
    Reson 8100-family intensity series subrecord is available.

    :return: (fields: dict (always empty), bytes_consumed: int) -- bytes_consumed always 8.
    """
    return {}, 8


def _decode_klein5410bss_imagery_specific(payload, pos):
    """
    Decode the Klein 5410 BSS sensor-specific portion of a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord. Ported from
    gsf_dec.c's DecodeKlein5410BssImagerySpecific().

    Untested against a verified GSF file: no sample data containing a
    Klein 5410 BSS intensity series subrecord is available.

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 18.
    """
    start = pos
    (res_mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (tvg_page,) = struct.unpack_from('>H', payload, pos); pos += 2
    beam_id = list(struct.unpack_from('>5H', payload, pos)); pos += 10
    pos += 4  # spare

    fields = {'ResMode': res_mode, 'TvgPage': tvg_page, 'BeamID': beam_id}
    return fields, pos - start


def _decode_r2sonic_imagery_specific(payload, pos):
    """
    Decode the R2Sonic sensor-specific portion of a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (2020/2022/
    2024). Ported from gsf_dec.c's DecodeR2SonicImagerySpecific().

    Untested against a verified GSF file: no sample data containing an
    R2Sonic intensity series subrecord is available.

    :return: (fields: dict, bytes_consumed: int) -- bytes_consumed always 168.
    """
    start = pos
    model_number = payload[pos:pos + 12].split(b'\x00', 1)[0].decode('ascii', 'replace'); pos += 12
    serial_number = payload[pos:pos + 12].split(b'\x00', 1)[0].decode('ascii', 'replace'); pos += 12
    (sec,) = struct.unpack_from('>i', payload, pos); pos += 4
    (nsec,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_period_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (sound_speed_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_power_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_pulse_width_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_beamwidth_vert_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_beamwidth_horiz_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (tx_steering_vert_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (tx_steering_horiz_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (tx_misc_info,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_bandwidth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_sample_rate_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_range_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_gain_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_spreading_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_absorption_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_mount_tilt_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    (rx_misc_info,) = struct.unpack_from('>I', payload, pos); pos += 4
    (reserved,) = struct.unpack_from('>H', payload, pos); pos += 2
    (num_beams,) = struct.unpack_from('>H', payload, pos); pos += 2
    more_info = [v / 1.0e6 for v in struct.unpack_from('>6i', payload, pos)]; pos += 24
    pos += 32  # spare

    fields = {
        'ModelNumber': model_number,
        'SerialNumber': serial_number,
        'PingTime': _gsf_timestamp(sec, nsec),
        'PingNumber': ping_number,
        'PingPeriod_s': ping_period_raw / 1.0e6,
        'SoundSpeed_mps': sound_speed_raw / 1.0e2,
        'Frequency_Hz': frequency_raw / 1.0e3,
        'TxPower_dB': tx_power_raw / 1.0e2,
        'TxPulseWidth_s': tx_pulse_width_raw / 1.0e7,
        'TxBeamwidthVert_deg': tx_beamwidth_vert_raw / 1.0e6,
        'TxBeamwidthHoriz_deg': tx_beamwidth_horiz_raw / 1.0e6,
        'TxSteeringVert_deg': tx_steering_vert_raw / 1.0e6,
        'TxSteeringHoriz_deg': tx_steering_horiz_raw / 1.0e6,
        'TxMiscInfo': tx_misc_info,
        'RxBandwidth_Hz': rx_bandwidth_raw / 1.0e4,
        'RxSampleRate_Hz': rx_sample_rate_raw / 1.0e3,
        'RxRange_m': rx_range_raw / 1.0e5,
        'RxGain_dB': rx_gain_raw / 1.0e2,
        'RxSpreading': rx_spreading_raw / 1.0e3,
        'RxAbsorption_dBkm': rx_absorption_raw / 1.0e3,
        'RxMountTilt_deg': rx_mount_tilt_raw / 1.0e6,
        'RxMiscInfo': rx_misc_info,
        'Reserved': reserved,
        'NumBeams': num_beams,
        'MoreInfo': more_info,
    }
    return fields, pos - start


def _decode_kmall_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC subrecord (id 156):
    Kongsberg SIS 5 / .kmall-derived per-ping sensor metadata, carried over
    from the originating #MRZ datagram's header/cmnPart/pingInfo, plus its
    per-transmit-sector and per-extra-detection-class arrays. Ported from
    gsf_dec.c's DecodeKMALLSpecific().

    :return: (scalars: dict, sector_rows: pandas.DataFrame, class_rows:
        pandas.DataFrame, bytes_consumed: int)
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

    sectors = pd.DataFrame(sector_rows)
    sectors.index.name = 'TxSectors'
    classes = pd.DataFrame(class_rows)
    classes.index.name = 'ExtraDetectionClasses'
    return s, sectors, classes, pos - start


def _decode_brb_intensity(payload, pos, num_beams, sensor_id):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord: a
    per-beam receive-beam backscatter time series. Ported from gsf_dec.c's
    DecodeBRBIntensity(). The fixed header and per-beam sample loop are
    sensor-agnostic, but gsflib inserts an optional sensor-specific
    "imagery" preamble between them whose size depends on sensor_id --
    every vendor format gsf_dec.c special-cases (KMALL, EM3-series,
    EM4-series, Reson 7125/T-series/8100-family, Klein 5410 BSS, R2Sonic)
    is decoded here via the matching _decode_*_imagery_specific() helper;
    any other sensor_id (including sensors gsflib itself doesn't
    special-case, e.g. SeaBat, SeaBeam, EM12/100/950/1000/121, GeoSwath,
    DeltaT) has no preamble at all (gsf_dec.c's switch default,
    sensor_size=0) and decodes straight through to the per-beam samples.

    :param sensor_id: the vendor "_SPECIFIC" subrecord id most recently
        seen for this ping (identifies which sensor-imagery format, if
        any, precedes the per-beam samples).
    :return: (header: dict with 'BitsPerSample', 'AppliedCorrections', and
        any fields decoded from a sensor-imagery preamble; beam_rows:
        list[dict] with 'SampleCount', 'DetectSample', 'StartRangeSamples',
        'Samples' (a list of ints); bytes_consumed), or None if num_beams
        is non-positive or bits_per_sample doesn't resolve to a supported
        sample width.
    """
    if num_beams <= 0:
        return None

    start = pos
    bits_per_sample = payload[pos]; pos += 1
    (applied_corrections,) = struct.unpack_from('>I', payload, pos); pos += 4
    pos += 16  # spare

    sensor_fields = {}
    if sensor_id == _SUBRECORD_KMALL_SPECIFIC:
        pos += _decode_kmall_imagery_specific(payload, pos)
    elif sensor_id in _SUBRECORD_EM3_IMAGERY_IDS:
        sensor_fields, consumed = _decode_em3_imagery_specific(payload, pos)
        pos += consumed
    elif sensor_id in _SUBRECORD_EM4_IMAGERY_IDS:
        sensor_fields, consumed = _decode_em4_imagery_specific(payload, pos)
        pos += consumed
    elif sensor_id in _SUBRECORD_RESON_SIZE_SPARE_IMAGERY_IDS:
        sensor_fields, consumed = _decode_reson_size_spare_imagery_specific(payload, pos)
        pos += consumed
    elif sensor_id in _SUBRECORD_RESON_8100_IMAGERY_IDS:
        sensor_fields, consumed = _decode_reson8100_imagery_specific(payload, pos)
        pos += consumed
    elif sensor_id == _SUBRECORD_KLEIN_5410_BSS_SPECIFIC:
        sensor_fields, consumed = _decode_klein5410bss_imagery_specific(payload, pos)
        pos += consumed
    elif sensor_id in _SUBRECORD_R2SONIC_IMAGERY_IDS:
        sensor_fields, consumed = _decode_r2sonic_imagery_specific(payload, pos)
        pos += consumed
    # else: no sensor-imagery preamble precedes the per-beam samples for
    # this sensor_id (gsf_dec.c's switch default, sensor_size=0).

    bytes_per_sample = bits_per_sample // 8
    header = {'BitsPerSample': bits_per_sample, 'AppliedCorrections': applied_corrections}
    header.update(sensor_fields)

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


def _decode_quality_flags_array(payload, pos, num_beams, subrecord_size):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_QUALITY_FLAGS_ARRAY subrecord: one
    2-bit quality flag per beam, four beams packed per byte (bits 7-6 =
    first beam in the byte, down to bits 1-0 = fourth), MSB first. Ported
    from gsf_dec.c's DecodeQualityFlagsArray(). Vectorized with numpy.

    If subrecord_size is too small to cover every beam (gsf_dec.c reads
    only sr_size * 4 beams in that case, leaving the rest at their
    pre-allocated zero value), the remaining beams are returned as 0,
    matching that behavior.

    :return: numpy uint8 array of length num_beams, values 0-3.
    """
    count = min(subrecord_size * 4, num_beams)
    raw = np.frombuffer(payload, dtype=np.uint8, count=(count + 3) // 4, offset=pos)
    shifts = np.array([6, 4, 2, 0], dtype=np.uint8)
    values = ((raw[:, None] >> shifts[None, :]) & 0x03).reshape(-1)[:count]
    if count < num_beams:
        values = np.concatenate([values, np.zeros(num_beams - count, dtype=np.uint8)])
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
        backscatter time series) into record['IntensityTimeSeries'] --
        this can be tens of thousands of samples per ping, so it's opt-in
        and normally left False (used by gsf.print_intensity_series(), not
        the default gsf.print_records() path).
    :return: record: dict -- the fixed ping scalars (e.g. 'PingTime',
        'Longitude_deg') stay flat/unprefixed at the top level, plus
        'Beams' (pandas.DataFrame, if any beam arrays were decoded),
        'IntensityTimeSeries' (pandas.DataFrame, if decode_intensity and
        one was present), 'SensorSpecificID' (int)/'SensorSpecific' (dict:
        that one family's own fields merged with its own tables, as
        pandas.DataFrames) if a vendor sensor-specific subrecord was
        present (at most one per ping -- see gsf.h's union
        gsfSensorSpecific; the family name is derivable on demand via
        _SENSOR_SPECIFIC_SUBRECORD_NAMES[SensorSpecificID]), and 'Notes'
        (list[str], always present, diagnostic messages for anything not
        decoded).
    """
    record = {}
    notes = []

    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    record['PingTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (lon_raw, lat_raw) = struct.unpack_from('>2i', payload, 8)
    record['Longitude_deg'] = lon_raw / 1.0e7
    record['Latitude_deg'] = lat_raw / 1.0e7

    (number_beams, center_beam, ping_flags, reserved) = struct.unpack_from('>4H', payload, 16)
    record['NumberBeams'] = number_beams
    record['CenterBeam'] = center_beam
    record['PingFlags'] = ping_flags

    (tide_raw,) = struct.unpack_from('>h', payload, 24)
    record['TideCorrector_m'] = tide_raw / 100.0

    (depth_corr_raw,) = struct.unpack_from('>i', payload, 26)
    record['DepthCorrector_m'] = depth_corr_raw / 100.0

    (heading_raw,) = struct.unpack_from('>H', payload, 30)
    record['Heading_deg'] = heading_raw / 100.0

    (pitch_raw, roll_raw, heave_raw) = struct.unpack_from('>3h', payload, 32)
    record['Pitch_deg'] = pitch_raw / 100.0
    record['Roll_deg'] = roll_raw / 100.0
    record['Heave_m'] = heave_raw / 100.0

    (course_raw, speed_raw) = struct.unpack_from('>2H', payload, 38)
    record['Course_deg'] = course_raw / 100.0
    record['Speed_kn'] = speed_raw / 100.0

    pos = 42
    if major_version > 2:
        (height_raw, sep_raw, gps_tide_raw) = struct.unpack_from('>3i', payload, pos)
        record['Height_m'] = height_raw / 1000.0
        record['SEP_m'] = sep_raw / 1000.0
        record['GPSTideCorrector_m'] = gps_tide_raw / 1000.0
        pos += 14  # 3 x 4-byte fields, plus 2 spare bytes

    beam_columns = {}
    sensor_id = None
    sensor_specific_id = None
    sensor_specific_fields = None
    sensor_specific_tables = None

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

        elif subrecord_id == _SUBRECORD_QUALITY_FLAGS_ARRAY:
            beam_columns['QualityFlags'] = _decode_quality_flags_array(
                payload, pos, number_beams, subrecord_size)

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
                            "IntensityTimeSeries (21, %d bytes) not decoded: no beams, or an "
                            "unsupported bits-per-sample encoding (sensor_id=%s)" % (subrecord_size, sensor_id))
                    else:
                        _header, beam_rows, _consumed = decoded
                        series = pd.DataFrame(beam_rows)
                        series.index.name = 'Beam'
                        record['IntensityTimeSeries'] = series

        elif subrecord_id in _PING_SENSOR_SPECIFIC_CODECS:
            sensor_id = subrecord_id
            family_label, decode_fn, _encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
            try:
                fields, extra_tables, _consumed = decode_fn(payload, pos)
            except (struct.error, IndexError) as exc:
                notes.append("%s (%d, %d bytes) not decoded: %s" %
                             (_SENSOR_SPECIFIC_SUBRECORD_NAMES.get(subrecord_id, str(subrecord_id)),
                              subrecord_id, subrecord_size, exc))
            else:
                sensor_specific_id = subrecord_id
                sensor_specific_fields = fields
                sensor_specific_tables = extra_tables

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
        record['Beams'] = beams

    if sensor_specific_id is not None:
        record['SensorSpecificID'] = sensor_specific_id
        merged = dict(sensor_specific_fields)
        for table_name, df in sensor_specific_tables.items():
            if len(df):
                merged[table_name] = df
        record['SensorSpecific'] = merged

    record['Notes'] = notes
    return record


def _decode_single_beam_ping(payload):
    """
    Decode a GSF_RECORD_SINGLE_BEAM_PING payload: the fixed-format
    scalars, plus its one sensor-specific tail subrecord (if any) via
    _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS. Ported from gsf_dec.c's
    gsfDecodeSinglebeam(). Unlike the swath-ping sensor-specific subrecord
    stream, a single-beam ping carries at most one such subrecord, so this
    doesn't need a loop -- it reads the one 4-byte id+size word at offset
    38 (if present) and dispatches once. gsfDecodeSinglebeam()'s obscure
    "extract a trailing subrecord id when the declared size is exactly 0"
    fallback (see gsf_dec.c) isn't replicated -- it only matters for a
    subrecord that carries an id but no payload, which none of the
    registered families do.

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_SINGLE_BEAM_PING record is available.

    :return: record: dict -- the fixed ping scalars (e.g. 'PingTime',
        'Depth_m') stay flat/unprefixed at the top level, plus
        'SensorSpecificID' (int)/'SensorSpecific' (dict, tables as
        pandas.DataFrames) if a sensor-specific tail subrecord was
        present, and 'Notes' (list[str], always present).
    """
    record = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    record['PingTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (lon_raw, lat_raw) = struct.unpack_from('>2i', payload, 8)
    record['Longitude_deg'] = lon_raw / 1.0e7
    record['Latitude_deg'] = lat_raw / 1.0e7

    (tide_raw,) = struct.unpack_from('>h', payload, 16)
    record['TideCorrector_m'] = tide_raw / 100.0

    (depth_corr_raw,) = struct.unpack_from('>i', payload, 18)
    record['DepthCorrector_m'] = depth_corr_raw / 100.0

    (heading_raw,) = struct.unpack_from('>H', payload, 22)
    record['Heading_deg'] = heading_raw / 100.0

    (pitch_raw, roll_raw, heave_raw) = struct.unpack_from('>3h', payload, 24)
    record['Pitch_deg'] = pitch_raw / 100.0
    record['Roll_deg'] = roll_raw / 100.0
    record['Heave_m'] = heave_raw / 100.0

    (depth_raw,) = struct.unpack_from('>i', payload, 30)
    record['Depth_m'] = depth_raw / 100.0

    (ssc_raw,) = struct.unpack_from('>h', payload, 34)
    record['SoundSpeedCorrection_m'] = ssc_raw / 100.0

    (pos_type,) = struct.unpack_from('>H', payload, 36)
    record['PositioningSystemType'] = pos_type

    remaining = len(payload) - 38
    notes = []
    if remaining > 4:
        word, = struct.unpack_from('>I', payload, 38)
        subrecord_id = (word >> 24) & 0xFF
        subrecord_size = word & 0x00FFFFFF
        if subrecord_id in _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS:
            _family_label, decode_fn, _encode_fn = _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[subrecord_id]
            try:
                fields, extra_tables, _consumed = decode_fn(payload, 42)
            except (struct.error, IndexError) as exc:
                notes.append("%s (%d, %d bytes) not decoded: %s" %
                             (_SINGLE_BEAM_SENSOR_SPECIFIC_NAMES.get(subrecord_id, str(subrecord_id)),
                              subrecord_id, subrecord_size, exc))
            else:
                record['SensorSpecificID'] = subrecord_id
                merged = dict(fields)
                for table_name, df in extra_tables.items():
                    if len(df):
                        merged[table_name] = df
                record['SensorSpecific'] = merged
        elif subrecord_id in _SINGLE_BEAM_SENSOR_SPECIFIC_NAMES:
            notes.append("%s (%d, %d bytes) not decoded" %
                         (_SINGLE_BEAM_SENSOR_SPECIFIC_NAMES[subrecord_id], subrecord_id, subrecord_size))
        else:
            notes.append("subrecord id %d (%d bytes) not decoded" % (subrecord_id, subrecord_size))

    record['Notes'] = notes
    return record


def _decode_swath_bathy_summary(payload):
    """
    Ported from gsf_dec.c's gsfDecodeSwathBathySummary().

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_SWATH_BATHY_SUMMARY record is available.
    """
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

    Verified against real GSF files for GSF_RECORD_PROCESSING_PARAMETERS.
    Untested against a verified GSF file for GSF_RECORD_SENSOR_PARAMETERS:
    no sample data containing that record type is available, even though
    it decodes with this same function.
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
    """
    Ported from gsf_dec.c's gsfDecodeComment().

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_COMMENT record is available.
    """
    scalars = {}
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    scalars['CommentTime'] = _gsf_timestamp(sec, nsec).isoformat()

    (length,) = struct.unpack_from('>I', payload, 8)
    scalars['Comment'] = payload[12:12 + length].decode('ascii', 'replace')

    return scalars, {}, []


def _decode_history(payload):
    """
    Ported from gsf_dec.c's gsfDecodeHistory().

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_HISTORY record is available.
    """
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
    """
    Ported from gsf_dec.c's gsfDecodeNavigationError() (obsolete record).

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_NAVIGATION_ERROR record is available.
    """
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
    """
    Ported from gsf_dec.c's gsfDecodeHVNavigationError().

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_HV_NAVIGATION_ERROR record is available.
    """
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


def _decode_header(payload):
    """ Decode a GSF_RECORD_HEADER payload: just the version string. """
    version = payload[:GSF_VERSION_SIZE].split(b'\x00', 1)[0].decode('ascii', 'replace')
    return {'Version': version}, {}, []


###########################################################
# Field-level record encoding (subset)
#
# The inverse of "Field-level record decoding" above: ported from
# gsf_enc.c's gsfEncode* and Encode*Array functions. Confirmed field-for-
# field symmetric with the decoders above (same order, same scale/offset
# formulas solved for the raw integer) by reading gsf_enc.c directly
# rather than assuming symmetry -- with two real asymmetries worth noting:
#
#   * gsfEncodeHeader() always stamps the library's own current version
#     (GSF_VERSION), ignoring any caller-supplied header.version -- so
#     does _encode_header() below.
#   * EncodeKMALLSpecific() always writes 0 for gsfKMALLVersion regardless
#     of the caller's struct value -- so does _encode_kmall_specific().
#
# Every scaled float->int field uses gsf_enc.c's own rounding convention
# (add/subtract 0.501 then truncate toward zero, via _gsf_round()) rather
# than Python's round() (banker's rounding) or plain truncation, so a
# decode(encode(x)) round-trip reproduces x to the field's precision.
###########################################################

#: Current GSF version stamped into every GSF_RECORD_HEADER written here,
#: matching this codebase's verified-current gsf_enc.c/gsf_dec.c source.
GSF_VERSION = "GSF-v03.11"

###########################################################
# "Not available" sentinel values (ported from gsf.h)
#
# A field a caller omits from `scalars` when calling write_swath_
# bathymetry_ping() is NOT encoded as 0/0.0 -- gsf.h defines an explicit,
# out-of-valid-range sentinel for most ping-header scalar fields
# specifically so a real (in-range) zero (a genuine 0.0 knot speed, a
# perfectly level 0.0 degree roll) can never be confused with "this value
# was never measured/computed". write_swath_bathymetry_ping() uses these
# GSF_NULL_* constants as its defaults; pass them explicitly wherever you
# have a value you know rather than merely omitting the key, to make that
# distinction unambiguous in your own code too. See gsf.h's "Define null
# values to be used for missing data" block, and convert.md's "Marking a
# field as not available" section.
###########################################################

GSF_NULL_LATITUDE = 91.0
GSF_NULL_LONGITUDE = 181.0
GSF_NULL_HEADING = 361.0
GSF_NULL_COURSE = 361.0
GSF_NULL_SPEED = 99.0
GSF_NULL_PITCH = 99.0
GSF_NULL_ROLL = 99.0
GSF_NULL_HEAVE = 99.0
#: gsf.h defines this as 0.0 -- draft has no value distinguishable from a
#: genuine zero-draft measurement; there is no way to mark it unavailable.
GSF_NULL_DRAFT = 0.0
GSF_NULL_DEPTH_CORRECTOR = 99.99
GSF_NULL_TIDE_CORRECTOR = 99.99
GSF_NULL_SOUND_SPEED_CORRECTION = 99.99
GSF_NULL_HORIZONTAL_ERROR = -1.00
GSF_NULL_VERTICAL_ERROR = -1.00
GSF_NULL_HEIGHT = 9999.99
GSF_NULL_SEP = 9999.99
#: Also 0.0 in gsf.h -- see GSF_NULL_DRAFT.
GSF_NULL_SEP_UNCERTAINTY = 0.0

#: Null values for the per-beam array subrecords. Unlike the scalars
#: above, gsf.h defines every one of these as plain 0.0 -- i.e. a beam
#: array value of 0 does NOT by itself mean "not available". Use the
#: BEAM_FLAGS_ARRAY ('BeamFlags' in `beams`, GSF_IGNORE_BEAM bit below) to
#: mark individual beams unusable instead of relying on any array value.
GSF_NULL_DEPTH = 0.0
GSF_NULL_ACROSS_TRACK = 0.0
GSF_NULL_ALONG_TRACK = 0.0
GSF_NULL_TRAVEL_TIME = 0.0
GSF_NULL_BEAM_ANGLE = 0.0
GSF_NULL_MC_AMPLITUDE = 0.0
GSF_NULL_MR_AMPLITUDE = 0.0
GSF_NULL_ECHO_WIDTH = 0.0
GSF_NULL_QUALITY_FACTOR = 0.0
GSF_NULL_RECEIVE_HEAVE = 0.0
GSF_NULL_DEPTH_ERROR = 0.0
GSF_NULL_ACROSS_TRACK_ERROR = 0.0
GSF_NULL_ALONG_TRACK_ERROR = 0.0
GSF_NULL_NAV_POS_ERROR = 0.0

#: Used in some sensor-specific subrecords to mark an unknown beam width.
GSF_BEAM_WIDTH_UNKNOWN = -1.0

#: PingFlags bit: the whole ping is unusable (gsf.h: GSF_IGNORE_PING).
#: The remaining 15 bits are application-defined (GSF_PING_USER_FLAG_01-15).
GSF_IGNORE_PING = 0x0001
#: BeamFlags (per beam, in the 'BeamFlags' array) bit: this beam is
#: unusable (gsf.h: GSF_IGNORE_BEAM). The remaining 7 bits are
#: application-defined (GSF_BEAM_USER_FLAG_01-07).
GSF_IGNORE_BEAM = 0x01

_ENCODE_DTYPE = {
    (1, False): '>u1', (1, True): '>i1',
    (2, False): '>u2', (2, True): '>i2',
    (4, False): '>u4', (4, True): '>i4',
}
_ENCODE_DTYPE_RANGE = {
    (1, False): (0, 255), (1, True): (-128, 127),
    (2, False): (0, 65535), (2, True): (-32768, 32767),
    (4, False): (0, 4294967295), (4, True): (-2147483648, 2147483647),
}

#: Default (multiplier, offset, field width in bytes, signed) for every
#: scaled ping beam array, chosen to comfortably span depths from 1m to
#: 10,000m without needing gsflib's auto-offset heuristic (see the
#: DEFAULT_PING_SCALE_FACTORS docstring-equivalent discussion: offset=0
#: with a 4-byte depth field already covers 0-4,294,967m at 1mm
#: precision). Multipliers/offsets/field widths for depth, across/along
#: track, travel time, beam angle, amplitude, echo width, receive heave,
#: beam_angle_forward, and vertical/horizontal error match the convention
#: found in real EM712 GSF files (see the scale-factor survey in the
#: kmall2gsf design discussion); the remaining, rarer arrays use the
#: unconditional (non-field-size-switchable) width gsf_dec.c's decode
#: switch requires for that subrecord, with multiplier=1, offset=0.
#:
#: This table has a second role when a gsf instance is writing with
#: auto_scale=True (see write_swath_bathymetry_ping()/
#: _pick_ping_scale_factor()): its multiplier for a given subrecordID is
#: then read as a *ceiling* -- the finest precision to use if the ping's
#: actual data fits, not the precision that will always be used. The
#: offset and field width columns keep their usual meaning either way;
#: only offset=0.0 here is ever overridden (auto_scale computes its own
#: offset from the ping's actual min/max instead).
DEFAULT_PING_SCALE_FACTORS = {
    1: (1000.0, 0.0, 4, False),      # Depth_m
    14: (1000.0, 0.0, 4, False),     # NominalDepth_m
    2: (100.0, 0.0, 4, True),        # AcrossTrack_m
    3: (100.0, 0.0, 4, True),        # AlongTrack_m
    4: (100000.0, 0.0, 4, False),    # TravelTime_s
    5: (100.0, 0.0, 2, True),        # BeamAngle_deg
    6: (1.0, 0.0, 2, True),          # MeanCalAmplitude_dB
    7: (1.0, 0.0, 2, False),         # MeanRelAmplitude_dB
    8: (100.0, 0.0, 2, False),       # EchoWidth_s
    9: (1.0, 0.0, 1, False),         # QualityFactor
    10: (100.0, 0.0, 1, True),       # ReceiveHeave_m
    17: (1.0, 0.0, 1, True),         # SignalToNoise_dB
    18: (100.0, 90.0, 2, False),     # BeamAngleForward_deg
    19: (100.0, 0.0, 2, False),      # VerticalError_m
    20: (100.0, 0.0, 2, False),      # HorizontalError_m
    22: (1.0, 0.0, 1, False),        # SectorNumber
    23: (1.0, 0.0, 1, False),        # DetectionInfo
    24: (1.0, 0.0, 1, True),         # IncidentBeamAdj_deg
    25: (1.0, 0.0, 1, False),        # SystemCleaning
    26: (1.0, 0.0, 1, True),         # DopplerCorr
    27: (100.0, 0.0, 2, False),      # SonarVertUncert_m
    28: (100.0, 0.0, 2, False),      # SonarHorzUncert_m
    29: (100.0, 0.0, 2, False),      # DetectionWindow_s
    30: (100.0, 0.0, 2, False),      # MeanAbsCoeff
    31: (1.0, 0.0, 1, False),        # TVG_dB
}


def _gsf_round(x):
    """
    Round `x` to the nearest integer using gsf_enc.c's own convention
    (add/subtract 0.501, then truncate toward zero) rather than Python's
    round() (banker's rounding) -- used everywhere gsf_enc.c scales a
    float for storage.
    """
    return int(x + 0.501) if x >= 0.0 else int(x - 0.501)


def _gsf_epoch(time_value):
    """
    Split a time value into (sec, nsec) ints for GSF's on-disk timespec
    fields. Accepts a POSIX timestamp (int/float, seconds since epoch), a
    datetime.datetime (naive datetimes are assumed UTC), or an ISO8601
    string (as produced by _gsf_timestamp(...).isoformat(), so a decoded
    scalars dict's time fields can be passed back to an encoder unchanged).
    """
    if isinstance(time_value, str):
        time_value = datetime.datetime.fromisoformat(time_value)
    if isinstance(time_value, datetime.datetime):
        if time_value.tzinfo is None:
            time_value = time_value.replace(tzinfo=datetime.timezone.utc)
        time_value = time_value.timestamp()
    sec = int(time_value)
    nsec = _gsf_round((time_value - sec) * 1.0e9)
    if nsec < 0:
        sec -= 1
        nsec += 1_000_000_000
    return sec, nsec


def _encode_header(version=None):
    """
    Encode a GSF_RECORD_HEADER payload. Ported from gsf_enc.c's
    gsfEncodeHeader(), which always stamps the library's own current
    version string, ignoring any caller-supplied value -- mirrored here:
    `version` exists only for testing, and defaults to GSF_VERSION.
    """
    encoded = (version or GSF_VERSION).encode('ascii')
    return encoded[:GSF_VERSION_SIZE].ljust(GSF_VERSION_SIZE, b'\x00')


def _encode_name_value_parameters(param_time, params):
    """
    Encode the shared GSF_RECORD_PROCESSING_PARAMETERS /
    GSF_RECORD_SENSOR_PARAMETERS wire format from a {name: value} dict.
    Ported from gsf_enc.c's gsfEncodeProcessingParameters() /
    gsfEncodeSensorParameters(): each entry is written as a NUL-terminated
    "NAME=VALUE" string, with its 2-byte size field counting the NUL --
    matching what real encoders write (see _decode_name_value_parameters()).

    :param param_time: POSIX timestamp or datetime.
    :param params: dict of {name: value}; values are str()-ed.
    """
    sec, nsec = _gsf_epoch(param_time)
    out = struct.pack('>2IH', sec, nsec, len(params))
    for name, value in params.items():
        text = ("%s=%s" % (name, value)).encode('ascii') + b'\x00'
        out += struct.pack('>h', len(text)) + text
    return out


def _encode_sound_velocity_profile(observation_time, application_time,
                                    latitude_deg, longitude_deg,
                                    depth_m, sound_speed_mPerSec):
    """
    Encode a GSF_RECORD_SOUND_VELOCITY_PROFILE payload. Ported from
    gsf_enc.c's gsfEncodeSoundVelocityProfile().

    :param depth_m, sound_speed_mPerSec: equal-length array-likes,
        non-negative (both are stored as unsigned centimeters).
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    sound_speed_mPerSec = np.asarray(sound_speed_mPerSec, dtype=np.float64)
    if len(depth_m) != len(sound_speed_mPerSec):
        raise ValueError("depth_m and sound_speed_mPerSec must be the same length")

    obs_sec, obs_nsec = _gsf_epoch(observation_time)
    app_sec, app_nsec = _gsf_epoch(application_time)

    out = struct.pack('>4I', obs_sec, obs_nsec, app_sec, app_nsec)
    out += struct.pack('>i', _gsf_round(longitude_deg * 1.0e7))
    out += struct.pack('>i', _gsf_round(latitude_deg * 1.0e7))
    out += struct.pack('>I', len(depth_m))

    raw = np.empty(2 * len(depth_m), dtype='>u4')
    raw[0::2] = (depth_m * 100.0 + 0.501).astype('>u4')
    raw[1::2] = (sound_speed_mPerSec * 100.0 + 0.501).astype('>u4')
    out += raw.tobytes()
    return out


def _encode_attitude(attitude_time, pitch_deg, roll_deg, heave_m, heading_deg):
    """
    Encode a GSF_RECORD_ATTITUDE payload. Ported from gsf_enc.c's
    gsfEncodeAttitude(): the first entry of `attitude_time` becomes the
    record's base time, and every measurement (including the first) is
    stored as a millisecond offset from it -- so `attitude_time` must be
    non-decreasing (offsets are stored as an unsigned 16-bit field, and
    must span less than 65.536 seconds).

    :param attitude_time: array-like of POSIX timestamps or datetimes.
    :param pitch_deg, roll_deg, heave_m, heading_deg: equal-length array-likes.
    """
    n = len(attitude_time)
    if not (len(pitch_deg) == len(roll_deg) == len(heave_m) == len(heading_deg) == n):
        raise ValueError("attitude arrays must all be the same length")

    base_sec, base_nsec = _gsf_epoch(attitude_time[0])
    out = struct.pack('>2IH', base_sec, base_nsec, n)
    for i in range(n):
        t_sec, t_nsec = _gsf_epoch(attitude_time[i])
        offset_ms = _gsf_round((t_sec - base_sec) * 1000.0 + (t_nsec - base_nsec) / 1.0e6)
        out += struct.pack(
            '>H3hH', offset_ms,
            _gsf_round(pitch_deg[i] * 100.0),
            _gsf_round(roll_deg[i] * 100.0),
            _gsf_round(heave_m[i] * 100.0),
            _gsf_round(heading_deg[i] * 100.0))
    return out


def _encode_swath_bathy_summary(start_time, end_time,
                                 min_latitude_deg, min_longitude_deg,
                                 max_latitude_deg, max_longitude_deg,
                                 min_depth_m, max_depth_m):
    """
    Encode a GSF_RECORD_SWATH_BATHY_SUMMARY payload. Ported from
    gsf_enc.c's gsfEncodeSwathBathySummary().

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_SWATH_BATHY_SUMMARY record is available.
    """
    start_sec, start_nsec = _gsf_epoch(start_time)
    end_sec, end_nsec = _gsf_epoch(end_time)
    out = struct.pack('>4I', start_sec, start_nsec, end_sec, end_nsec)
    for value, scale in (
        (min_latitude_deg, 1.0e7), (min_longitude_deg, 1.0e7),
        (max_latitude_deg, 1.0e7), (max_longitude_deg, 1.0e7),
        (min_depth_m, 100.0), (max_depth_m, 100.0),
    ):
        out += struct.pack('>i', _gsf_round(value * scale))
    return out


def _encode_comment(comment_time, comment):
    """
    Encode a GSF_RECORD_COMMENT payload. Ported from gsf_enc.c's
    gsfEncodeComment().

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_COMMENT record is available.
    """
    sec, nsec = _gsf_epoch(comment_time)
    text = comment.encode('ascii')
    return struct.pack('>3I', sec, nsec, len(text)) + text


def _encode_history(history_time, host_name, operator_name, command_line, comment):
    """
    Encode a GSF_RECORD_HISTORY payload. Ported from gsf_enc.c's
    gsfEncodeHistory(). host_name, operator_name, and command_line are
    each written as a NUL-terminated string, with the 2-byte size field
    counting that NUL byte -- comment is the odd one out, written with no
    NUL terminator and a size field that's the plain string length. This
    matches gsf_enc.c exactly; _decode_history() doesn't strip the
    embedded NUL from the other three fields, so a round trip returns
    host_name/operator_name/command_line with a trailing '\\x00'.

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_HISTORY record is available.
    """
    sec, nsec = _gsf_epoch(history_time)
    out = struct.pack('>2I', sec, nsec)
    for value in (host_name, operator_name, command_line):
        text = value.encode('ascii') + b'\x00'
        out += struct.pack('>H', len(text)) + text
    text = comment.encode('ascii')
    out += struct.pack('>H', len(text)) + text
    return out


def _encode_navigation_error(nav_error_time, record_id, longitude_error_m, latitude_error_m):
    """
    Encode a GSF_RECORD_NAVIGATION_ERROR payload (obsolete record,
    superseded by GSF_RECORD_HV_NAVIGATION_ERROR -- see
    _encode_hv_navigation_error()). Ported from gsf_enc.c's
    gsfEncodeNavigationError(), with one deliberate deviation: the
    reference encoder rounds both fields with an unconditional `+ 0.501`
    (no sign check), which for negative error values is a rounding bug --
    e.g. a longitude_error of -1.29 m encodes (via the reference's own
    formula, truncating toward zero after the scale+offset) to -12
    (1/10 m units) instead of the correctly-rounded -13. This uses the
    standard, sign-correct _gsf_round() convention used by every other
    encoder in this module.

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_NAVIGATION_ERROR record is available.
    """
    sec, nsec = _gsf_epoch(nav_error_time)
    out = struct.pack('>3I', sec, nsec, record_id)
    out += struct.pack('>i', _gsf_round(longitude_error_m * 10.0))
    out += struct.pack('>i', _gsf_round(latitude_error_m * 10.0))
    return out


def _encode_hv_navigation_error(nav_error_time, record_id, horizontal_error_m,
                                 vertical_error_m, sep_uncertainty_m, position_type=""):
    """
    Encode a GSF_RECORD_HV_NAVIGATION_ERROR payload. Ported from
    gsf_enc.c's gsfEncodeHVNavigationError(). The reference encoder rounds
    vertical_error with +/-0.5 rather than the +/-0.501 used everywhere
    else (including horizontal_error here); functionally equivalent except
    exactly on a 0.5 fractional boundary, so this uses the standard
    _gsf_round() convention for both fields.

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_HV_NAVIGATION_ERROR record is available.
    """
    sec, nsec = _gsf_epoch(nav_error_time)
    out = struct.pack('>3I', sec, nsec, record_id)
    out += struct.pack('>i', _gsf_round(horizontal_error_m * 1000.0))
    out += struct.pack('>i', _gsf_round(vertical_error_m * 1000.0))
    out += struct.pack('>H', _gsf_round(sep_uncertainty_m * 100.0))
    out += b'\x00\x00'  # spare
    text = position_type.encode('ascii')
    out += struct.pack('>H', len(text)) + text
    return out


#: Fields _encode_single_beam_ping()/_encode_swath_bathymetry_ping() require
#: `record` to supply (see each function's docstring).
_REQUIRED_SINGLE_BEAM_PING_FIELDS = (
    'PingTime', 'Longitude_deg', 'Latitude_deg', 'TideCorrector_m',
    'DepthCorrector_m', 'Heading_deg', 'Pitch_deg', 'Roll_deg', 'Heave_m',
    'Depth_m', 'SoundSpeedCorrection_m')


def _split_sensor_specific(sensor_specific):
    """
    Split a merged SensorSpecific dict (as returned by
    _decode_swath_bathymetry_ping()/_decode_single_beam_ping() -- a flat
    dict of scalar fields plus zero or more pandas.DataFrame-valued table
    entries) back into the (fields: dict, tables: dict[str,
    pandas.DataFrame]) shape every registered encode_fn expects. This is
    pure repackaging, not a data-format conversion -- table values pass
    through untouched as DataFrames, exactly as decode produced them.
    """
    fields, tables = {}, {}
    for k, v in sensor_specific.items():
        if isinstance(v, pd.DataFrame):
            tables[k] = v
        else:
            fields[k] = v
    return fields, tables


def _encode_single_beam_ping(record):
    """
    Encode a GSF_RECORD_SINGLE_BEAM_PING payload: the fixed-format
    scalars, plus (if given) one sensor-specific tail subrecord via
    _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS. Ported from gsf_enc.c's
    gsfEncodeSinglebeam(). The exact inverse of _decode_single_beam_ping()
    -- a decoded record can be passed straight back in, unmodified.

    :param record: dict with the same shape _decode_single_beam_ping()
        returns: the 11 fixed scalars listed in
        _REQUIRED_SINGLE_BEAM_PING_FIELDS (all required), optional
        'PositioningSystemType' (defaults to 0), and optional
        'SensorSpecificID' (int)/'SensorSpecific' (dict, any table values
        as pandas.DataFrame) for the one sensor-specific tail subrecord.
        'Notes', if present, is ignored (there is no wire slot for it).
    :raises ValueError: a required field is missing from `record`.
    :raises KeyError: `record['SensorSpecificID']` has no encoder
        registered in _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS.

    Untested against a verified GSF file: no sample data containing a
    GSF_RECORD_SINGLE_BEAM_PING record is available.
    """
    missing = [k for k in _REQUIRED_SINGLE_BEAM_PING_FIELDS if record.get(k) is None]
    if missing:
        raise ValueError("record is missing required field(s): %s" % ', '.join(missing))

    g = record.get
    sec, nsec = _gsf_epoch(g('PingTime'))
    out = struct.pack('>2I', sec, nsec)
    out += struct.pack('>i', _gsf_round(g('Longitude_deg') * 1.0e7))
    out += struct.pack('>i', _gsf_round(g('Latitude_deg') * 1.0e7))
    out += struct.pack('>h', _gsf_round(g('TideCorrector_m') * 100.0))
    out += struct.pack('>i', _gsf_round(g('DepthCorrector_m') * 100.0))
    out += struct.pack('>H', _gsf_round(g('Heading_deg') * 100.0))
    out += struct.pack('>h', _gsf_round(g('Pitch_deg') * 100.0))
    out += struct.pack('>h', _gsf_round(g('Roll_deg') * 100.0))
    out += struct.pack('>h', _gsf_round(g('Heave_m') * 100.0))
    out += struct.pack('>i', _gsf_round(g('Depth_m') * 100.0))
    out += struct.pack('>h', _gsf_round(g('SoundSpeedCorrection_m') * 100.0))
    out += struct.pack('>H', int(g('PositioningSystemType', 0)))

    subrecord_id = record.get('SensorSpecificID')
    if subrecord_id is not None:
        if subrecord_id not in _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS:
            raise KeyError("no encoder registered for single-beam sensor-specific subrecord id %d" % subrecord_id)
        _family_label, _decode_fn, encode_fn = _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[subrecord_id]
        fields, tables = _split_sensor_specific(record.get('SensorSpecific') or {})
        out += encode_fn(subrecord_id, fields, tables)

    return out


def _encode_scale_factors(scale_factors):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS subrecord, including
    its own 4-byte subrecord id+size word. Ported from gsf_enc.c's
    EncodeScaleFactors(): entries are written in ascending subrecordID
    order regardless of the input dict's iteration order.

    :param scale_factors: dict mapping subrecordID -> (multiplier, offset,
        compressionFlag).
    """
    entries = sorted(scale_factors.items())
    body = struct.pack('>I', len(entries))
    for subrecord_id, (multiplier, offset, compression_flag) in entries:
        word = ((subrecord_id & 0xFF) << 24) | ((compression_flag & 0xFF) << 16)
        body += struct.pack('>I', word)
        body += struct.pack('>I', _gsf_round(multiplier))
        body += struct.pack('>i', _gsf_round(offset))
    header_word = (_SUBRECORD_SCALE_FACTORS << 24) | len(body)
    return struct.pack('>I', header_word) + body


def _encode_ping_array(subrecord_id, values, multiplier, offset, signed, width):
    """
    Encode one beam-array subrecord, including its own 4-byte subrecord
    id+size word, from engineering-unit values: raw = round((value +
    offset) * multiplier). Vectorized with numpy -- the inverse of
    _decode_ping_array(). Ported from gsf_enc.c's Encode*Array family.

    :raises ValueError: an encoded value doesn't fit in the requested
        integer width, rather than silently wrapping/truncating data.
    """
    values = np.asarray(values, dtype=np.float64)
    scaled = (values + offset) * multiplier
    raw = np.where(scaled >= 0, scaled + 0.501, scaled - 0.501).astype(np.int64)

    lo, hi = _ENCODE_DTYPE_RANGE[(width, signed)]
    if raw.size and (int(raw.min()) < lo or int(raw.max()) > hi):
        raise ValueError(
            "subrecord %d: encoded value out of range for a %d-byte %s field "
            "(multiplier=%s, offset=%s)" %
            (subrecord_id, width, "signed" if signed else "unsigned", multiplier, offset))

    body = raw.astype(_ENCODE_DTYPE[(width, signed)]).tobytes()
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


#: Auto-scale hysteresis constants used by _pick_ping_scale_factor() below.
#: When a ping's data no longer fits the currently active scale factor and
#: a new one has to be solved for, the target value range is padded
#: outward before solving, so that a slightly different next ping is
#: unlikely to immediately force yet another change. The padding is
#: whichever is larger of: a fraction of the observed value range for
#: this ping (_AUTO_SCALE_MARGIN_FRACTION), or a fixed number of steps at
#: the field's target precision (_AUTO_SCALE_MIN_MARGIN_STEPS) -- the
#: latter matters when the observed range is tiny or zero (e.g. a dead
#: flat seafloor, or calm heave), where a fraction of it would pad by
#: almost nothing and every ping's ordinary noise would trigger a new
#: solve.
_AUTO_SCALE_MARGIN_FRACTION = 0.25
_AUTO_SCALE_MIN_MARGIN_STEPS = 20.0


def _scale_factor_fits(min_v, max_v, multiplier, offset, width, signed):
    """
    Report whether a given (multiplier, offset) pair would let
    _encode_ping_array() represent both min_v and max_v -- the smallest
    and largest values in some beam array -- within the integer range of
    a field of the given width (in bytes) and signedness, without
    raising its own out-of-range ValueError.

    This reuses _gsf_round(), the exact rounding convention
    _encode_ping_array() itself applies to every scaled value, so a
    "yes" from this function is guaranteed to agree with what the real
    encoder would actually do -- there is no separate, approximate copy
    of the rounding rule here that could drift out of sync with it.

    :param min_v, max_v: the smallest and largest engineering-unit
        values that need to be representable.
    :param multiplier, offset: the candidate scale factor being tested;
        the same values _encode_ping_array() would use to compute
        raw = round((value + offset) * multiplier) for every beam.
    :param width: field width in bytes (1, 2, or 4).
    :param signed: whether the on-disk integer field is signed.
    :return: True if both min_v and max_v round to a value inside the
        field's representable integer range, False otherwise.
    """
    lo, hi = _ENCODE_DTYPE_RANGE[(width, signed)]
    return lo <= _gsf_round((min_v + offset) * multiplier) <= hi and \
        lo <= _gsf_round((max_v + offset) * multiplier) <= hi


def _solve_scale_factor(min_v, max_v, width, signed, target_multiplier, pad=0.0):
    """
    Compute a single (multiplier, offset) pair that represents the range
    [min_v - pad, max_v + pad] as precisely as possible within a field
    of the given width and signedness, preferring target_multiplier
    (the ideal/ceiling precision for this field, e.g. 1000.0 for 1mm
    depth resolution) and preferring offset=0.0 whenever the data
    doesn't actually require a shift.

    Both multiplier and offset are always returned as whole numbers
    (just represented as Python floats). This matters for a reason
    that isn't obvious from the arithmetic alone: _encode_ping_array()
    scales beam values using the *exact* multiplier/offset it is given,
    but _encode_scale_factors() -- which writes the on-disk
    SCALE_FACTORS subrecord a reader will use to undo that scaling --
    separately rounds multiplier/offset to whole numbers before writing
    them. If this function ever handed back a non-whole-number multiplier
    or offset, the beam values written to the file and the scale factor
    a reader later divides by would silently disagree. Rounding to whole
    numbers here, up front, is what keeps the two in agreement.

    :param min_v, max_v: the value range (already padded by the caller
        if any hysteresis headroom is wanted; this function does not
        apply any padding of its own beyond the `pad` argument).
    :param width: field width in bytes (1, 2, or 4).
    :param signed: whether the on-disk integer field is signed.
    :param target_multiplier: the ideal/ceiling precision to use if the
        padded range fits at that precision.
    :param pad: an amount to subtract from min_v and add to max_v before
        solving, giving the result some headroom beyond the exact
        [min_v, max_v] range. Pass 0.0 for an exact, unpadded fit.
    :return: (multiplier, offset), both whole numbers, guaranteed (by
        construction, not merely by luck) to satisfy
        _scale_factor_fits(min_v, max_v, multiplier, offset, width, signed)
        for the padded range -- see the step-by-step comments below for
        why that guarantee holds.
    """
    lo, hi = _ENCODE_DTYPE_RANGE[(width, signed)]
    pmin, pmax = min_v - pad, max_v + pad
    pspan = pmax - pmin

    # Step 1: choose the multiplier. Start from the ideal/target
    # precision and only reduce it if the padded value range would not
    # fit the field at that precision -- i.e. shrink precision only as
    # far as forced to, never further. (hi - lo) / pspan is the largest
    # multiplier that would make the padded span exactly reach from lo
    # to hi; capping target_multiplier at that value guarantees the
    # padded range fits. floor to a whole number via int() -- safe to
    # use plain truncation here (rather than a true floor function)
    # because multiplier is always positive at this point, so truncating
    # toward zero and flooring are the same operation. Clamp at 1, the
    # coarsest a GSF multiplier can ever be.
    if pspan <= 0:
        multiplier = target_multiplier
    else:
        multiplier = min(target_multiplier, (hi - lo) / pspan)
    multiplier = float(max(1, int(multiplier)))

    # Step 2: with that multiplier fixed, work out which offsets would
    # keep both padded endpoints inside [lo, hi]. off_lo is the smallest
    # offset that keeps pmin from rounding below lo; off_hi is the
    # largest offset that keeps pmax from rounding above hi. Because
    # multiplier was capped in step 1 specifically so that the padded
    # span fits inside (hi - lo), this interval [off_lo, off_hi] is
    # mathematically guaranteed to be non-empty (off_lo <= off_hi) --
    # there is always at least one valid offset to choose from.
    off_lo = lo / multiplier - pmin
    off_hi = hi / multiplier - pmax

    # Step 3: pick a whole-number offset from [off_lo, off_hi]. Prefer
    # 0.0 when it's already inside that interval -- this is what makes
    # the common case (data doesn't need any DC shift at all) come out
    # identical to today's static defaults, which always use offset=0.
    # Otherwise, take whichever end of the interval is closest to zero,
    # rounded *away from* the interval (ceiling the low end, flooring
    # the high end) rather than toward it, so the chosen whole number is
    # guaranteed to still land inside [off_lo, off_hi] and not just
    # near it.
    if off_lo <= 0.0 <= off_hi:
        offset = 0.0
    elif off_lo > 0.0:
        offset = float(np.ceil(off_lo))
    else:
        offset = float(np.floor(off_hi))
    return multiplier, offset


def _pick_ping_scale_factor(values, current, width, signed, target_multiplier,
                             margin_fraction=_AUTO_SCALE_MARGIN_FRACTION,
                             min_margin_steps=_AUTO_SCALE_MIN_MARGIN_STEPS):
    """
    Choose the (multiplier, offset) to use for one ping's beam array,
    reusing the scale factor already active for this subrecordID
    whenever it still works, and only solving for a new one when it
    doesn't. This is the entry point write_swath_bathymetry_ping() calls
    once per beam-array label when writing with auto_scale=True.

    The design goal is that this function is cheap to call on every
    single ping: it does exactly one vectorized pass over `values` (a
    numpy min/max reduction) to find this ping's value range, then does
    a handful of scalar comparisons -- there is no per-beam Python loop,
    and in the common case (the currently active scale factor already
    covers this ping) no further arithmetic happens at all.

    :param values: this ping's beam array, in engineering units (e.g.
        depth in meters). Non-finite values (NaN/Inf) are ignored when
        finding the value range, since GSF has no on-disk representation
        for "not a number" in a beam array; if every value is
        non-finite, the currently active scale factor (or, failing
        that, (target_multiplier, 0.0)) is returned unchanged, since
        there is nothing to fit.
    :param current: the (multiplier, offset) currently active for this
        subrecordID, as previously returned by this function, or None
        if this is the first ping this subrecordID has been seen in for
        this file.
    :param width: field width in bytes (1, 2, or 4).
    :param signed: whether the on-disk integer field is signed.
    :param target_multiplier: the ideal/ceiling precision for this
        field (see DEFAULT_PING_SCALE_FACTORS's docstring note on its
        dual role in auto_scale mode).
    :param margin_fraction, min_margin_steps: see the module-level
        _AUTO_SCALE_MARGIN_FRACTION/_AUTO_SCALE_MIN_MARGIN_STEPS
        constants' docstring for what these control.
    :return: (multiplier, offset) to use for this ping -- either `current`
        unchanged, or a freshly solved pair.
    """
    # Step 1: find this ping's actual value range. This is the only pass
    # over the per-beam data; everything below is scalar arithmetic on
    # just these two numbers, however many beams the ping has.
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return current if current is not None else (target_multiplier, 0.0)
    min_v, max_v = float(values.min()), float(values.max())

    # Step 2 -- the fast path, and the common case in practice: if the
    # scale factor already active for this subrecordID still covers
    # this ping's range, keep using it, unchanged. No new SCALE_FACTORS
    # values get computed, and (from the caller's side) nothing about
    # this subrecordID's on-disk scale factor changes between pings.
    # This is what "the scale factors don't need to change with every
    # ping" means in practice.
    if current is not None and _scale_factor_fits(min_v, max_v, current[0], current[1], width, signed):
        return current

    # Step 3: the current scale factor -- or the lack of one, on the
    # first ping this subrecordID has been seen in -- doesn't cover this
    # ping's range. A new one is genuinely needed. Rather than solving
    # for the tightest possible fit to just this ping's exact range
    # (which the very next, slightly different ping might again exceed,
    # forcing another change immediately), pad the target range outward
    # first and solve for that. This padding is the hysteresis: it
    # trades a small amount of precision headroom for scale-factor
    # stability across pings, the same trade gsflib's own depth-offset
    # auto function makes with its 100m "layers," but computed directly
    # from the observed data instead of a fixed, depth-specific size.
    span = max_v - min_v
    pad = max(margin_fraction * span, min_margin_steps / target_multiplier)
    multiplier, offset = _solve_scale_factor(min_v, max_v, width, signed, target_multiplier, pad)

    # Step 4: the padding above is a nice-to-have for stability, never
    # allowed to make representable data unrepresentable. In the rare
    # case where a field is already near its capacity, padding outward
    # can push the solve past what the field can hold even though the
    # ping's actual (unpadded) data would fit fine -- so if that
    # happens, retry with an exact, unpadded fit. If even that doesn't
    # fit, this field genuinely cannot represent this ping's data at
    # its declared width; rather than trying to paper over that here,
    # return the best attempt and let _encode_ping_array()'s own
    # ValueError be the one place that error is ever raised, exactly as
    # it already is today for the static, non-auto-scaled path.
    if not _scale_factor_fits(min_v, max_v, multiplier, offset, width, signed):
        multiplier, offset = _solve_scale_factor(min_v, max_v, width, signed, target_multiplier, pad=0.0)
    return multiplier, offset


def _encode_quality_flags_array(values):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_QUALITY_FLAGS_ARRAY subrecord,
    including its own 4-byte subrecord id+size word: the inverse of
    _decode_quality_flags_array(). Ported from gsf_enc.c's
    EncodeQualityFlagsArray(). Vectorized with numpy.

    :param values: array-like of per-beam quality flags, 0-3 (masked to
        2 bits, matching the C source's implicit truncation).
    """
    values = np.asarray(values, dtype=np.uint8) & 0x03
    pad = (-len(values)) % 4
    if pad:
        values = np.concatenate([values, np.zeros(pad, dtype=np.uint8)])
    shifts = np.array([6, 4, 2, 0], dtype=np.uint8)
    packed = (values.reshape(-1, 4) << shifts[None, :]).sum(axis=1, dtype=np.uint8)
    body = packed.astype('>u1').tobytes()
    header_word = (_SUBRECORD_QUALITY_FLAGS_ARRAY << 24) | len(body)
    return struct.pack('>I', header_word) + body


def new_kmall_specific():
    """
    Return a new dict with every GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC
    scalar field name present, pre-set to 0/0.0 (or, for
    NumBytesPerTxSector/NumBytesPerClass, the fixed byte counts
    _encode_kmall_specific() itself defaults to). gsf.h defines no
    GSF_NULL_* sentinel for these vendor-specific fields (unlike the ping
    scalars from new_swath_bathymetry_ping_scalars()), so 0 is the only
    "not specified" marker available -- if 0 is itself a plausible value
    for a field you care about, be sure to actually set it here rather
    than relying on the default. Field meanings are documented on
    _decode_kmall_specific(); convert.md points to real sample values.

    Note: GSFKMALLVersion is always forced to 0 on encode regardless of
    what you set here (a gsf_enc.c quirk, see _encode_kmall_specific());
    NumTxSectors/NumExtraDetectionClasses (also omitted from this dict)
    are likewise always derived from len(tx_sectors)/len(class_rows), not
    read from this dict, so there's no point setting them.
    """
    return {
        'DgmType': 0, 'DgmVersion': 0, 'SystemID': 0, 'EchoSounderID': 0,
        'NumBytesCmnPart': 0, 'PingCnt': 0, 'RxFansPerPing': 0, 'RxFanIndex': 0,
        'SwathsPerPing': 0, 'SwathAlongPosition': 0, 'TxTransducerInd': 0,
        'RxTransducerInd': 0, 'NumRxTransducers': 0, 'AlgorithmType': 0,
        'NumBytesInfoData': 0, 'PingRate_Hz': 0.0, 'BeamSpacing': 0,
        'DepthMode': 0, 'SubDepthMode': 0, 'DistanceBtwSwath': 0,
        'DetectionMode': 0, 'PulseForm': 0, 'FrequencyMode_Hz': 0.0,
        'FreqRangeLowLim_Hz': 0.0, 'FreqRangeHighLim_Hz': 0.0,
        'MaxTotalTxPulseLength_sec': 0.0, 'MaxEffTxPulseLength_sec': 0.0,
        'MaxEffTxBandWidth_Hz': 0.0, 'AbsCoeff_dBPerkm': 0.0,
        'PortSectorEdge_deg': 0.0, 'StarbSectorEdge_deg': 0.0,
        'PortMeanCov_deg': 0.0, 'StarbMeanCov_deg': 0.0,
        'PortMeanCov_m': 0.0, 'StarbMeanCov_m': 0.0,
        'ModeAndStabilisation': 0, 'RuntimeFilter1': 0, 'RuntimeFilter2': 0,
        'PipeTrackingStatus': 0, 'TransmitArraySizeUsed_deg': 0.0,
        'ReceiveArraySizeUsed_deg': 0.0, 'TransmitPower_dB': 0.0,
        'SLrampUpTimeRemaining': 0, 'YawAngle_deg': 0.0,
        'NumBytesPerTxSector': 53, 'HeadingVessel_deg': 0.0,
        'SoundSpeedAtTxDepth_mPerSec': 0.0, 'TxTransducerDepth_m': 0.0,
        'ZWaterLevelReRefPoint_m': 0.0, 'XKmallToAll_m': 0.0, 'YKmallToAll_m': 0.0,
        'LatLongInfo': 0, 'PosSensorStatus': 0, 'AttitudeSensorStatus': 0,
        'Latitude_deg': 0.0, 'Longitude_deg': 0.0, 'EllipsoidHeightReRefPoint_m': 0.0,
        'NumBytesRxInfo': 0, 'NumSoundingsMaxMain': 0, 'NumSoundingsValidMain': 0,
        'NumBytesPerSounding': 0, 'WCSampleRate': 0.0, 'SeabedImageSampleRate': 0.0,
        'BSnormal_dB': 0.0, 'BSoblique_dB': 0.0,
        'ExtraDetectionAlarmFlag': 0, 'NumExtraDetections': 0, 'NumBytesPerClass': 35,
    }


def new_kmall_tx_sector():
    """
    Return a new dict with every per-transmit-sector field name (one entry
    of the `tx_sectors` list passed to write_swath_bathymetry_ping()),
    pre-set to 0/0.0 -- see new_kmall_specific() for why 0 (not a
    GSF_NULL_* sentinel) is the best available default here.
    """
    return {
        'TxSectorNumb': 0, 'TxArrNumber': 0, 'TxSubArray': 0,
        'SectorTransmitDelay_sec': 0.0, 'TiltAngleReTx_deg': 0.0,
        'TxNominalSourceLevel_dB': 0.0, 'TxFocusRange_m': 0.0,
        'CentreFreq_Hz': 0.0, 'SignalBandWidth_Hz': 0.0,
        'TotalSignalLength_sec': 0.0, 'PulseShading': 0, 'SignalWaveForm': 0,
    }


def _encode_kmall_specific(s, sector_rows=None, class_rows=None):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC subrecord (id 156),
    including its own 4-byte subrecord id+size word. Ported from
    gsf_enc.c's EncodeKMALLSpecific(); the exact field-for-field inverse
    of _decode_kmall_specific() (same key names, so a decoded dict can be
    re-encoded directly).

    :param s: dict of scalar KMALL_SPECIFIC fields (see
        _decode_kmall_specific()'s return). Missing keys default to 0.
        NumTxSectors/NumExtraDetectionClasses are always derived from
        len(sector_rows)/len(class_rows), not read from `s`.
    :param sector_rows: pandas.DataFrame of per-transmit-sector rows (see
        _decode_kmall_specific()), at most 9 (GSF_MAX_KMALL_SECTORS) are
        written -- the exact DataFrame _decode_kmall_specific() returns
        can be passed straight back in, no conversion required.
    :param class_rows: pandas.DataFrame of per-extra-detection-class rows,
        at most 11 (GSF_MAX_KMALL_EXTRA_CLASSES) are written.
    """
    if sector_rows is None:
        sector_rows = pd.DataFrame()
    if class_rows is None:
        class_rows = pd.DataFrame()
    g = s.get

    out = bytearray()
    # gsf_enc.c always writes 0 for gsfKMALLVersion, regardless of `s`.
    out += struct.pack('>4B', 0, g('DgmType', 0), g('DgmVersion', 0), g('SystemID', 0))
    out += struct.pack('>H', g('EchoSounderID', 0))
    out += b'\x00' * 8

    out += struct.pack('>2H', g('NumBytesCmnPart', 0), g('PingCnt', 0))
    out += struct.pack(
        '>6B', g('RxFansPerPing', 0), g('RxFanIndex', 0), g('SwathsPerPing', 0),
        g('SwathAlongPosition', 0), g('TxTransducerInd', 0), g('RxTransducerInd', 0))
    out += struct.pack('>2B', g('NumRxTransducers', 0), g('AlgorithmType', 0))
    out += b'\x00' * 16

    out += struct.pack('>H', g('NumBytesInfoData', 0))
    out += struct.pack('>I', _gsf_round(g('PingRate_Hz', 0.0) * 1.0e5))
    out += struct.pack(
        '>6B', g('BeamSpacing', 0), g('DepthMode', 0), g('SubDepthMode', 0),
        g('DistanceBtwSwath', 0), g('DetectionMode', 0), g('PulseForm', 0))
    out += struct.pack('>i', _gsf_round(g('FrequencyMode_Hz', 0.0)))
    out += struct.pack('>i', _gsf_round(g('FreqRangeLowLim_Hz', 0.0) * 1.0e3))
    out += struct.pack('>i', _gsf_round(g('FreqRangeHighLim_Hz', 0.0) * 1.0e3))
    out += struct.pack('>i', _gsf_round(g('MaxTotalTxPulseLength_sec', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('MaxEffTxPulseLength_sec', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('MaxEffTxBandWidth_Hz', 0.0) * 1.0e3))
    out += struct.pack('>i', _gsf_round(g('AbsCoeff_dBPerkm', 0.0) * 1.0e3))
    out += struct.pack('>h', _gsf_round(g('PortSectorEdge_deg', 0.0) * 1.0e2))
    out += struct.pack('>h', _gsf_round(g('StarbSectorEdge_deg', 0.0) * 1.0e2))
    # Mirrors the duplicate port/starboard mean-coverage-in-degrees pair
    # gsf_dec.c reads and discards (see _decode_kmall_specific()); written
    # here as zero-filled padding to keep byte alignment identical.
    out += b'\x00' * 4
    out += struct.pack('>h', _gsf_round(g('PortMeanCov_deg', 0.0) * 1.0e2))
    out += struct.pack('>h', _gsf_round(g('StarbMeanCov_deg', 0.0) * 1.0e2))
    out += struct.pack('>h', _gsf_round(g('PortMeanCov_m', 0.0)))
    out += struct.pack('>h', _gsf_round(g('StarbMeanCov_m', 0.0)))
    out += struct.pack('>2B', g('ModeAndStabilisation', 0), g('RuntimeFilter1', 0))
    out += struct.pack('>H', g('RuntimeFilter2', 0))
    out += struct.pack('>i', g('PipeTrackingStatus', 0))
    out += struct.pack('>H', _gsf_round(g('TransmitArraySizeUsed_deg', 0.0) * 1.0e3))
    out += struct.pack('>H', _gsf_round(g('ReceiveArraySizeUsed_deg', 0.0) * 1.0e3))
    out += struct.pack('>h', _gsf_round(g('TransmitPower_dB', 0.0) * 1.0e2))
    out += struct.pack('>H', g('SLrampUpTimeRemaining', 0))
    out += struct.pack('>i', _gsf_round(g('YawAngle_deg', 0.0) * 1.0e6))
    out += struct.pack('>H', len(sector_rows))
    out += struct.pack('>H', g('NumBytesPerTxSector', 53))
    out += struct.pack('>i', _gsf_round(g('HeadingVessel_deg', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('SoundSpeedAtTxDepth_mPerSec', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('TxTransducerDepth_m', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('ZWaterLevelReRefPoint_m', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('XKmallToAll_m', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('YKmallToAll_m', 0.0) * 1.0e6))
    out += struct.pack('>3B', g('LatLongInfo', 0), g('PosSensorStatus', 0), g('AttitudeSensorStatus', 0))
    out += struct.pack('>i', _gsf_round(g('Latitude_deg', 0.0) * 1.0e7))
    out += struct.pack('>i', _gsf_round(g('Longitude_deg', 0.0) * 1.0e7))
    out += struct.pack('>i', _gsf_round(g('EllipsoidHeightReRefPoint_m', 0.0) * 1.0e3))
    out += b'\x00' * 32

    for _, row in sector_rows.iloc[:9].iterrows():  # gsf.h: GSF_MAX_KMALL_SECTORS
        r = row.get
        out += struct.pack('>3B', int(r('TxSectorNumb', 0)), int(r('TxArrNumber', 0)), int(r('TxSubArray', 0)))
        out += struct.pack('>i', _gsf_round(r('SectorTransmitDelay_sec', 0.0) * 1.0e6))
        out += struct.pack('>i', _gsf_round(r('TiltAngleReTx_deg', 0.0) * 1.0e6))
        out += struct.pack('>i', _gsf_round(r('TxNominalSourceLevel_dB', 0.0) * 1.0e6))
        out += struct.pack('>i', _gsf_round(r('TxFocusRange_m', 0.0) * 1.0e3))
        out += struct.pack('>i', _gsf_round(r('CentreFreq_Hz', 0.0) * 1.0e3))
        out += struct.pack('>i', _gsf_round(r('SignalBandWidth_Hz', 0.0) * 1.0e3))
        out += struct.pack('>i', _gsf_round(r('TotalSignalLength_sec', 0.0) * 1.0e6))
        out += struct.pack('>2B', int(r('PulseShading', 0)), int(r('SignalWaveForm', 0)))
        out += struct.pack('>i', _gsf_round(r('HighVoltageLevel_dB', 0.0) * 1.0e6))
        out += struct.pack('>i', _gsf_round(r('SectorTrackingCorr_dB', 0.0) * 1.0e6))
        out += struct.pack('>i', _gsf_round(r('EffectiveSignalLength_sec', 0.0) * 1.0e6))
        out += b'\x00' * 8

    out += struct.pack(
        '>4H', g('NumBytesRxInfo', 0), g('NumSoundingsMaxMain', 0),
        g('NumSoundingsValidMain', 0), g('NumBytesPerSounding', 0))
    wc = g('WCSampleRate', 0.0)
    wc_int = int(wc)
    wc_frac = _gsf_round((wc - wc_int) * 1.0e9)
    out += struct.pack('>iI', wc_int, wc_frac)
    sb = g('SeabedImageSampleRate', 0.0)
    sb_int = int(sb)
    sb_frac = _gsf_round((sb - sb_int) * 1.0e9)
    out += struct.pack('>iI', sb_int, sb_frac)
    out += struct.pack('>i', _gsf_round(g('BSnormal_dB', 0.0) * 1.0e6))
    out += struct.pack('>i', _gsf_round(g('BSoblique_dB', 0.0) * 1.0e6))
    out += struct.pack('>H', g('ExtraDetectionAlarmFlag', 0))
    out += struct.pack('>H', g('NumExtraDetections', 0))
    out += struct.pack('>H', len(class_rows))
    out += struct.pack('>H', g('NumBytesPerClass', 35))
    out += b'\x00' * 32

    for _, row in class_rows.iloc[:11].iterrows():  # gsf.h: GSF_MAX_KMALL_EXTRA_CLASSES
        out += struct.pack('>H', int(row.get('NumExtraDetInClass', 0)))
        out += struct.pack('>B', int(row.get('AlarmFlag', 0)))
        out += b'\x00' * 32

    out += b'\x00' * 32

    header_word = (_SUBRECORD_KMALL_SPECIFIC << 24) | len(out)
    return struct.pack('>I', header_word) + bytes(out)


def _decode_kmall_specific_adapter(payload, pos):
    """
    Adapter registering KMALL_SPECIFIC (id 156) in
    _PING_SENSOR_SPECIFIC_CODECS under the registry's generic
    decode_fn(payload, pos) -> (fields, tables, bytes_consumed) contract
    every other family follows. Delegates to _decode_kmall_specific(),
    whose own two-named-table return predates this registry and keeps its
    own dedicated tests -- this is pure repackaging, no data conversion,
    since both sides already speak pandas.DataFrame.
    """
    scalars, sector_df, class_df, consumed = _decode_kmall_specific(payload, pos)
    tables = {}
    if len(sector_df):
        tables['TxSectors'] = sector_df
    if len(class_df):
        tables['ExtraDetectionClasses'] = class_df
    return scalars, tables, consumed


def _encode_kmall_specific_adapter(subrecord_id, fields, tables=None):
    """
    Adapter registering KMALL_SPECIFIC (id 156) as an encode_fn under the
    registry's generic (subrecord_id, fields, tables) -> bytes contract.
    Splits the merged `tables` dict back into _encode_kmall_specific()'s
    own two positional DataFrame parameters -- pure repackaging, no data
    conversion. `subrecord_id` is accepted (matching the generic
    contract) but ignored -- _encode_kmall_specific() always stamps 156
    itself via _SUBRECORD_KMALL_SPECIFIC.
    """
    tables = tables or {}
    return _encode_kmall_specific(fields, tables.get('TxSectors'), tables.get('ExtraDetectionClasses'))


_PING_SENSOR_SPECIFIC_CODECS[_SUBRECORD_KMALL_SPECIFIC] = (
    "KMALL", _decode_kmall_specific_adapter, _encode_kmall_specific_adapter)


#: label (as used in tables['Beams']/_PING_ARRAY_SUBRECORDS) -> subrecordID.
_LABEL_TO_SUBRECORD_ID = {label: sid for sid, (_attr, label, _signed) in _PING_ARRAY_SUBRECORDS.items()}


def _beam_array_subrecord_id(label):
    """ Resolve a beams dict column label to its ping subrecord id. """
    if label == 'BeamFlags':
        return _SUBRECORD_BEAM_FLAGS_ARRAY
    if label == 'QualityFlags':
        return _SUBRECORD_QUALITY_FLAGS_ARRAY
    if label in _LABEL_TO_SUBRECORD_ID:
        return _LABEL_TO_SUBRECORD_ID[label]
    raise KeyError("no known ping array subrecord for beams column %r" % label)


def new_swath_bathymetry_ping_scalars():
    """
    Return a new dict with every GSF_RECORD_SWATH_BATHYMETRY_PING scalar
    field name present -- the four required fields (PingTime,
    Longitude_deg, Latitude_deg, NumberBeams) set to None as a placeholder
    you must overwrite, and every optional field pre-set to its
    GSF_NULL_* "not available" sentinel (or, for CenterBeam/PingFlags/
    GPSTideCorrector_m, to 0/0.0 -- gsf.h defines no sentinel for those
    three). Populate this dict with whatever you actually know, add
    'Beams' (a dict of {column label: array-like} or a pandas.DataFrame)
    and, optionally, 'SensorSpecificID'/'SensorSpecific' for a vendor
    sensor-specific subrecord, and pass the result straight to
    write_swath_bathymetry_ping(); every scalar field you don't touch is
    written as "not available", not as a misleading 0/0.0, and every
    valid key name is visible here in one place instead of having to be
    looked up. See convert.md's "Marking a field as not available"
    section.

    :raises ValueError: (from write_swath_bathymetry_ping()/
        _encode_swath_bathymetry_ping()) if PingTime, Longitude_deg,
        Latitude_deg, or NumberBeams is still None when you pass this
        dict to write_swath_bathymetry_ping().
    """
    return {
        'PingTime': None,
        'Longitude_deg': None,
        'Latitude_deg': None,
        'NumberBeams': None,
        'CenterBeam': 0,
        'PingFlags': 0,
        'TideCorrector_m': GSF_NULL_TIDE_CORRECTOR,
        'DepthCorrector_m': GSF_NULL_DEPTH_CORRECTOR,
        'Heading_deg': GSF_NULL_HEADING,
        'Pitch_deg': GSF_NULL_PITCH,
        'Roll_deg': GSF_NULL_ROLL,
        'Heave_m': GSF_NULL_HEAVE,
        'Course_deg': GSF_NULL_COURSE,
        'Speed_kn': GSF_NULL_SPEED,
        'Height_m': GSF_NULL_HEIGHT,
        'SEP_m': GSF_NULL_SEP,
        'GPSTideCorrector_m': 0.0,
    }


def _encode_swath_bathymetry_ping(record, scale_factors=None, major_version=3):
    """
    Encode a GSF_RECORD_SWATH_BATHYMETRY_PING payload: the fixed-format
    scalar fields, a GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS subrecord, one
    beam-array subrecord per entry in `record['Beams']`, and (if given) the
    vendor sensor-specific subrecord. The exact inverse of
    _decode_swath_bathymetry_ping(): `record` uses the same shape that
    function returns, so a decoded ping can be re-encoded directly,
    unmodified (PingTime accepts the ISO8601 string
    _decode_swath_bathymetry_ping() produces -- see _gsf_epoch()).

    :param record: dict; PingTime, Longitude_deg, Latitude_deg, and
        NumberBeams are required (must be present and not None). Building
        this dict with new_swath_bathymetry_ping_scalars() is recommended
        over hand-assembling it: it pre-fills every valid scalar key name,
        so there's nothing to look up and nothing to get wrong. CenterBeam,
        PingFlags, and (at major_version > 2) GPSTideCorrector_m default
        to 0/0.0 if absent (no GSF_NULL_* sentinel is defined for these).
        Every other scalar key (TideCorrector_m, DepthCorrector_m,
        Heading_deg, Pitch_deg, Roll_deg, Heave_m, Course_deg, Speed_kn,
        and, at major_version > 2, Height_m/SEP_m) defaults to its
        GSF_NULL_* sentinel (e.g. GSF_NULL_SPEED = 99.0 knots) if absent,
        per gsf.h's convention -- NOT 0/0.0, since 0 is itself a valid
        measured value for most of these fields. Pass the value explicitly
        (0.0 or otherwise) when you have it; omit the key only when it's
        genuinely unavailable.

        record['Beams']: dict of {column label: array-like} or a
        pandas.DataFrame, e.g. {'Depth_m': [...], 'AcrossTrack_m': [...]}.
        Every array must have length NumberBeams. Only labels resolvable
        by _beam_array_subrecord_id() (i.e. present in
        DEFAULT_PING_SCALE_FACTORS/`scale_factors`, or
        'BeamFlags'/'QualityFlags') can be encoded. Omit it (or pass an
        empty dict) for a ping with no beam arrays.

        record['SensorSpecificID']/record['SensorSpecific']: optional, for
        any vendor sensor-specific subrecord registered in
        _PING_SENSOR_SPECIFIC_CODECS (including KMALL_SPECIFIC, id 156).
        'SensorSpecific' is a flat dict of that family's scalar field
        names (as returned, unprefixed, by _decode_swath_bathymetry_ping())
        plus, for families with nested per-element arrays (EM3, EM3Raw,
        EM4, KMALL), any table entries as pandas.DataFrame values under
        their table name (e.g. 'TxSectors') -- the exact DataFrames
        _decode_swath_bathymetry_ping() returns can be passed straight
        back in, no conversion required.

        record['Notes'], if present, is ignored (there is no wire slot for
        record-level free-text notes in this subrecord).
    :param scale_factors: optional override of DEFAULT_PING_SCALE_FACTORS;
        same shape (subrecordID -> (multiplier, offset, field_width_bytes,
        signed)).
    :raises KeyError: `record['SensorSpecificID']` has no encoder
        registered in _PING_SENSOR_SPECIFIC_CODECS.
    :raises ValueError: PingTime, Longitude_deg, Latitude_deg, or
        NumberBeams is missing or None in `record`; or an encoded beam
        value doesn't fit its field width.
    :raises KeyError: `record['Beams']` has a column with no resolvable
        subrecordID.
    """
    missing = [k for k in ('PingTime', 'Longitude_deg', 'Latitude_deg', 'NumberBeams')
               if record.get(k) is None]
    if missing:
        raise ValueError(
            "record is missing required field(s): %s (see "
            "new_swath_bathymetry_ping_scalars())" % ', '.join(missing))

    g = record.get
    number_beams = int(record['NumberBeams'])
    beams = record.get('Beams')
    if beams is None:
        beams = {}

    out = struct.pack('>2I', *_gsf_epoch(record['PingTime']))
    out += struct.pack('>i', _gsf_round(record['Longitude_deg'] * 1.0e7))
    out += struct.pack('>i', _gsf_round(record['Latitude_deg'] * 1.0e7))
    out += struct.pack('>4H', number_beams, int(g('CenterBeam', 0)), int(g('PingFlags', 0)), 0)
    out += struct.pack('>h', _gsf_round(g('TideCorrector_m', GSF_NULL_TIDE_CORRECTOR) * 100.0))
    out += struct.pack('>i', _gsf_round(g('DepthCorrector_m', GSF_NULL_DEPTH_CORRECTOR) * 100.0))
    out += struct.pack('>H', _gsf_round(g('Heading_deg', GSF_NULL_HEADING) * 100.0))
    out += struct.pack(
        '>3h',
        _gsf_round(g('Pitch_deg', GSF_NULL_PITCH) * 100.0),
        _gsf_round(g('Roll_deg', GSF_NULL_ROLL) * 100.0),
        _gsf_round(g('Heave_m', GSF_NULL_HEAVE) * 100.0))
    out += struct.pack(
        '>2H',
        _gsf_round(g('Course_deg', GSF_NULL_COURSE) * 100.0),
        _gsf_round(g('Speed_kn', GSF_NULL_SPEED) * 100.0))

    if major_version > 2:
        out += struct.pack(
            '>3i',
            _gsf_round(g('Height_m', GSF_NULL_HEIGHT) * 1000.0),
            _gsf_round(g('SEP_m', GSF_NULL_SEP) * 1000.0),
            # gsf.h defines no GSF_NULL_GPS_TIDE_CORRECTOR -- 0.0 is the
            # best available default (also a real, achievable value).
            _gsf_round(g('GPSTideCorrector_m', 0.0) * 1000.0))
        out += b'\x00' * 2

    sf_table = scale_factors if scale_factors is not None else DEFAULT_PING_SCALE_FACTORS

    used_scale_factors = {}
    array_subrecords = b''
    for label in sorted(beams, key=_beam_array_subrecord_id):
        values = beams[label]
        if label == 'BeamFlags':
            body = np.asarray(values, dtype=np.uint8).astype('>u1').tobytes()
            header_word = (_SUBRECORD_BEAM_FLAGS_ARRAY << 24) | len(body)
            array_subrecords += struct.pack('>I', header_word) + body
            continue

        if label == 'QualityFlags':
            array_subrecords += _encode_quality_flags_array(values)
            continue

        subrecord_id = _LABEL_TO_SUBRECORD_ID[label]
        multiplier, offset, width, signed = sf_table[subrecord_id]
        used_scale_factors[subrecord_id] = (float(multiplier), float(offset), width << 4)
        array_subrecords += _encode_ping_array(subrecord_id, values, multiplier, offset, signed, width)

    out += _encode_scale_factors(used_scale_factors)
    out += array_subrecords

    subrecord_id = record.get('SensorSpecificID')
    if subrecord_id is not None:
        if subrecord_id not in _PING_SENSOR_SPECIFIC_CODECS:
            raise KeyError("no encoder registered for sensor-specific subrecord id %d" % subrecord_id)
        _family_label, _decode_fn, encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
        fields, tables = _split_sensor_specific(record.get('SensorSpecific') or {})
        out += encode_fn(subrecord_id, fields, tables)

    return out


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

    def __init__(self, filename=None, auto_scale=False):
        """
        Create a gsf object bound to `filename`, but don't open it yet --
        open_read()/OpenFiletoRead() or the various write_* methods do
        that on first use.

        :param filename: path to the GSF file this object will read from
            or write to.
        :param auto_scale: default value of the `auto_scale` parameter on
            write_swath_bathymetry_ping() for every call made through
            this instance, unless a given call overrides it explicitly.
            When True, write_swath_bathymetry_ping() picks each beam
            array's scale factor (multiplier, offset) automatically from
            that ping's actual data, reusing the previous ping's scale
            factor whenever it still fits rather than recomputing it
            every time -- see _pick_ping_scale_factor()'s docstring for
            the full algorithm, and the README's "Scale factors" section
            for a worked example. Defaults to False, i.e. writing keeps
            using DEFAULT_PING_SCALE_FACTORS statically unless a caller
            opts in.
        """
        self.verbose = 0
        self.filename = filename
        self.FID = None
        self.file_size = None
        self.Index = None

        #: GSF version string read from the file's GSF_RECORD_HEADER record
        #: (e.g. "GSF-v03.09"), set by index_file().
        self.gsfVersion = None

        #: Default for write_swath_bathymetry_ping()'s auto_scale parameter.
        self.auto_scale = auto_scale
        #: subrecordID -> the (multiplier, offset) currently active for
        #: that beam array, as chosen by _pick_ping_scale_factor(). Only
        #: populated/consulted when writing with auto_scale=True; carries
        #: forward from one write_swath_bathymetry_ping() call to the
        #: next on this same instance, which is what lets a scale factor
        #: stay stable across pings instead of being recomputed for each
        #: one.
        self._auto_scale_factors = {}

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

    @staticmethod
    def _print_ping_record(record):
        """
        print_records() helper for the two record types that return a
        merged dict (_decode_swath_bathymetry_ping()/
        _decode_single_beam_ping()) instead of the (scalars, tables,
        notes) 3-tuple every other decoded record type uses. Prints flat
        scalars, then 'Notes', then 'Beams'/'IntensityTimeSeries' as
        tables, then 'SensorSpecific' (its own scalar fields under a
        header naming the resolved family, followed by any of its own
        table entries).
        """
        flat = {k: v for k, v in record.items()
                if k not in ('Beams', 'IntensityTimeSeries', 'SensorSpecificID', 'SensorSpecific', 'Notes')}
        if flat:
            width = max(len(k) for k in flat)
            for k, v in flat.items():
                print("  %-*s : %s" % (width, k, v))
        for note in record.get('Notes', []):
            print("  # %s" % note)
        for label in ('Beams', 'IntensityTimeSeries'):
            table = record.get(label)
            if table is not None and len(table):
                print("-- %s --" % label)
                print(table.to_string())
        sensor_specific = record.get('SensorSpecific')
        if sensor_specific:
            sensor_id = record.get('SensorSpecificID')
            family_name = _SENSOR_SPECIFIC_SUBRECORD_NAMES.get(sensor_id, str(sensor_id))
            scalar_items = {k: v for k, v in sensor_specific.items() if not isinstance(v, pd.DataFrame)}
            if scalar_items:
                print("-- SensorSpecific (%s, id=%s) --" % (family_name, sensor_id))
                width = max(len(k) for k in scalar_items)
                for k, v in scalar_items.items():
                    print("  %-*s : %s" % (width, k, v))
            for k, v in sensor_specific.items():
                if isinstance(v, pd.DataFrame) and len(v):
                    print("-- %s --" % k)
                    print(v.to_string())

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

                    # Dispatch to the specific decoder for this record type,
                    # explicitly here rather than via a generic lookup, so the
                    # type -> decoder mapping is visible at the call site.
                    decoded = None
                    try:
                        rid = data_id.recordID
                        if rid == RecordType.GSF_RECORD_HEADER:
                            decoded = _decode_header(payload)
                        elif rid == RecordType.GSF_RECORD_SWATH_BATHY_SUMMARY:
                            decoded = _decode_swath_bathy_summary(payload)
                        elif rid == RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING:
                            decoded = _decode_swath_bathymetry_ping(payload, major_version, scale_factors)
                        elif rid == RecordType.GSF_RECORD_SOUND_VELOCITY_PROFILE:
                            decoded = _decode_sound_velocity_profile(payload)
                        elif rid in (RecordType.GSF_RECORD_PROCESSING_PARAMETERS, RecordType.GSF_RECORD_SENSOR_PARAMETERS):
                            # GSF_RECORD_SENSOR_PARAMETERS: untested against a verified
                            # GSF file -- no sample data containing that record type.
                            decoded = _decode_name_value_parameters(payload)
                        elif rid == RecordType.GSF_RECORD_COMMENT:
                            decoded = _decode_comment(payload)
                        elif rid == RecordType.GSF_RECORD_HISTORY:
                            decoded = _decode_history(payload)
                        elif rid == RecordType.GSF_RECORD_NAVIGATION_ERROR:
                            decoded = _decode_navigation_error(payload)
                        elif rid == RecordType.GSF_RECORD_HV_NAVIGATION_ERROR:
                            decoded = _decode_hv_navigation_error(payload)
                        elif rid == RecordType.GSF_RECORD_SINGLE_BEAM_PING:
                            decoded = _decode_single_beam_ping(payload)
                        elif rid == RecordType.GSF_RECORD_ATTITUDE:
                            decoded = _decode_attitude(payload)
                    except (struct.error, IndexError) as exc:
                        print("  # decode failed (%s); showing raw text" % exc)

                    if decoded is None:
                        text = ''.join(chr(b) if 32 <= b < 127 else '.' for b in payload)
                        print(text)
                    elif rid in (RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING,
                                 RecordType.GSF_RECORD_SINGLE_BEAM_PING):
                        self._print_ping_record(decoded)
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

        Decoded for every sensor gsflib defines an imagery-specific preamble
        for (KMALL, EM3-series, EM4-series, Reson 7125/T-series/8100-family,
        Klein 5410 BSS, R2Sonic) as well as every sensor that has no such
        preamble at all; pings that carry no intensity series subrecord, or
        whose bits-per-sample doesn't resolve to a supported sample width,
        are noted and skipped rather than guessed at.
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
                        record = _decode_swath_bathymetry_ping(
                            payload, major_version, scale_factors, decode_intensity=True)
                    except (struct.error, IndexError) as exc:
                        print("# ping offset=%d: decode failed (%s)" % (offset, exc))
                    else:
                        series = record.get('IntensityTimeSeries')
                        if series is None:
                            note = next((n for n in record.get('Notes', []) if 'IntensityTimeSeries' in n), None)
                            print("# ping offset=%d ping_time=%s: %s" %
                                  (offset, record.get('PingTime', '?'),
                                   note or "no intensity time series subrecord"))
                        else:
                            print("# ping offset=%d ping_time=%s" % (offset, record.get('PingTime', '?')))
                            print("# Beam,SampleCount,DetectSample,StartRangeSamples,Sample0,Sample1,...")
                            for beam, row in series.iterrows():
                                fields = [str(beam), str(row['SampleCount']), str(row['DetectSample']),
                                          str(row['StartRangeSamples'])]
                                fields.extend(str(v) for v in row['Samples'])
                                print(",".join(fields))

            self.FID.seek(offset + GSF_RECORD_FRAMING_SIZE + readSize, 0)

    ###########################################################
    # Writing
    ###########################################################

    def write_record(self, record_id, payload, checksum=False):
        """
        Write one GSF record: [4-byte size][4-byte data identifier
        (+4-byte checksum if requested)][payload]. Inverse of
        read_record_header() + the payload reads elsewhere in this class;
        mirrors gsf.c's gsfWrite()/gsfPackStream() framing, per the same
        encoding documented at gsfDataID above.

        :param record_id: a RecordType (or int recordID).
        :param payload: bytes -- the fully assembled record body, as
            returned by one of the _encode_* functions above.
        :param checksum: if True, prefix payload with gsf_checksum(payload)
            and set the checksum flag bit in the data identifier word.
        """
        if self.FID is None:
            self.OpenFiletoWrite()

        record_id = int(record_id)
        if checksum:
            body = struct.pack('>I', gsf_checksum(payload)) + payload
            did = 0x80000000 | (record_id & 0x003FFFFF)
        else:
            body = payload
            did = record_id & 0x003FFFFF

        self.FID.write(struct.pack('>II', len(payload), did) + body)

    def write_header(self, version=None):
        """
        Write the GSF_RECORD_HEADER record and remember its version (as
        self.gsfVersion) for subsequent write_swath_bathymetry_ping() calls.
        Must be the first record written to a new file. See _encode_header()
        -- note that the real GSF encoder always stamps its own current
        version, so `version` exists mainly for testing.
        """
        version = version or GSF_VERSION
        self.write_record(RecordType.GSF_RECORD_HEADER, _encode_header(version))
        self.gsfVersion = version

    def write_processing_parameters(self, params, param_time):
        """ Write a GSF_RECORD_PROCESSING_PARAMETERS record. See
        _encode_name_value_parameters(). """
        self.write_record(
            RecordType.GSF_RECORD_PROCESSING_PARAMETERS,
            _encode_name_value_parameters(param_time, params))

    def write_sensor_parameters(self, params, param_time):
        """
        Write a GSF_RECORD_SENSOR_PARAMETERS record. See
        _encode_name_value_parameters().

        Untested against a verified GSF file: no sample data containing a
        GSF_RECORD_SENSOR_PARAMETERS record is available, so the round-trip
        test for this method (unlike write_processing_parameters(), whose
        record type does appear in sample data) checks self-consistency
        only.
        """
        self.write_record(
            RecordType.GSF_RECORD_SENSOR_PARAMETERS,
            _encode_name_value_parameters(param_time, params))

    def write_sound_velocity_profile(self, observation_time, application_time,
                                      latitude_deg, longitude_deg,
                                      depth_m, sound_speed_mPerSec):
        """ Write a GSF_RECORD_SOUND_VELOCITY_PROFILE record. See
        _encode_sound_velocity_profile(). """
        self.write_record(
            RecordType.GSF_RECORD_SOUND_VELOCITY_PROFILE,
            _encode_sound_velocity_profile(
                observation_time, application_time, latitude_deg, longitude_deg,
                depth_m, sound_speed_mPerSec))

    def write_attitude(self, attitude_time, pitch_deg, roll_deg, heave_m, heading_deg):
        """ Write a GSF_RECORD_ATTITUDE record. See _encode_attitude(). """
        self.write_record(
            RecordType.GSF_RECORD_ATTITUDE,
            _encode_attitude(attitude_time, pitch_deg, roll_deg, heave_m, heading_deg))

    def write_swath_bathymetry_ping(self, record, scale_factors=None, auto_scale=None):
        """
        Write a GSF_RECORD_SWATH_BATHYMETRY_PING record. See
        _encode_swath_bathymetry_ping() for the expected shape of
        `record`/`scale_factors` -- the same dict shape
        _decode_swath_bathymetry_ping() returns, so a decoded ping can be
        passed straight back in. Uses self.gsfVersion (set by
        write_header(), which must be called first) to decide whether to
        include the height/SEP/GPS-tide-corrector fields (major_version >
        2 -- true for every GSF_VERSION this codebase writes).

        :param auto_scale: if True (or left as None with self.auto_scale
            True), `scale_factors` is computed automatically for every
            beam array in `record['Beams']`, instead of falling back to
            DEFAULT_PING_SCALE_FACTORS: each array's (multiplier, offset)
            is chosen by _pick_ping_scale_factor() from that array's
            actual values in this ping, reusing whatever scale factor was
            already active for that subrecordID on this gsf instance
            (self._auto_scale_factors) whenever it still fits, so a
            file's scale factors only change when the data actually
            requires it, not on every ping. See the README's "Scale
            factors" section for a worked example of when and why this
            differs from the static defaults, and _pick_ping_scale_factor()'s
            docstring for the full algorithm. Passing an explicit
            `scale_factors` together with `auto_scale=True` is not
            supported, since the two are two different answers to the
            same question ("what scale factors should this ping use").
        :raises ValueError: `auto_scale` resolves to True while
            `scale_factors` is also given explicitly.
        """
        if auto_scale is None:
            auto_scale = self.auto_scale
        if auto_scale and scale_factors is not None:
            raise ValueError(
                "auto_scale=True and an explicit scale_factors= override are mutually exclusive")

        beams = record.get('Beams')
        if beams is None:
            beams = {}

        if auto_scale:
            # Resolve one (multiplier, offset, width, signed) scale
            # factor per beam array actually present in this ping,
            # building the same shape of dict write_swath_bathymetry_ping()
            # would otherwise accept as an explicit scale_factors=
            # override -- BeamFlags/QualityFlags are excluded since they
            # aren't scaled at all (see _beam_array_subrecord_id()).
            scale_factors = {}
            for label in beams:
                if label in ('BeamFlags', 'QualityFlags'):
                    continue
                subrecord_id = _beam_array_subrecord_id(label)
                target_multiplier, _default_offset, width, signed = DEFAULT_PING_SCALE_FACTORS[subrecord_id]
                current = self._auto_scale_factors.get(subrecord_id)
                multiplier, offset = _pick_ping_scale_factor(
                    beams[label], current, width, signed, target_multiplier)
                self._auto_scale_factors[subrecord_id] = (multiplier, offset)
                scale_factors[subrecord_id] = (multiplier, offset, width, signed)

        major_version = _gsf_major_version(self.gsfVersion)
        self.write_record(
            RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING,
            _encode_swath_bathymetry_ping(record, scale_factors, major_version))

    def write_swath_bathy_summary(self, start_time, end_time,
                                   min_latitude_deg, min_longitude_deg,
                                   max_latitude_deg, max_longitude_deg,
                                   min_depth_m, max_depth_m):
        """
        Write a GSF_RECORD_SWATH_BATHY_SUMMARY record. See
        _encode_swath_bathy_summary().

        Untested against a verified GSF file: no sample data containing a
        GSF_RECORD_SWATH_BATHY_SUMMARY record is available.
        """
        self.write_record(
            RecordType.GSF_RECORD_SWATH_BATHY_SUMMARY,
            _encode_swath_bathy_summary(
                start_time, end_time, min_latitude_deg, min_longitude_deg,
                max_latitude_deg, max_longitude_deg, min_depth_m, max_depth_m))

    def write_comment(self, comment_time, comment):
        """
        Write a GSF_RECORD_COMMENT record. See _encode_comment().

        Untested against a verified GSF file: no sample data containing a
        GSF_RECORD_COMMENT record is available.
        """
        self.write_record(RecordType.GSF_RECORD_COMMENT, _encode_comment(comment_time, comment))

    def write_history(self, history_time, host_name, operator_name, command_line, comment):
        """
        Write a GSF_RECORD_HISTORY record. See _encode_history().

        Untested against a verified GSF file: no sample data containing a
        GSF_RECORD_HISTORY record is available.
        """
        self.write_record(
            RecordType.GSF_RECORD_HISTORY,
            _encode_history(history_time, host_name, operator_name, command_line, comment))

    def write_navigation_error(self, nav_error_time, record_id, longitude_error_m, latitude_error_m):
        """
        Write a GSF_RECORD_NAVIGATION_ERROR record (obsolete; prefer
        write_hv_navigation_error()). See _encode_navigation_error().

        Untested against a verified GSF file: no sample data containing a
        GSF_RECORD_NAVIGATION_ERROR record is available.
        """
        self.write_record(
            RecordType.GSF_RECORD_NAVIGATION_ERROR,
            _encode_navigation_error(nav_error_time, record_id, longitude_error_m, latitude_error_m))

    def write_hv_navigation_error(self, nav_error_time, record_id, horizontal_error_m,
                                   vertical_error_m, sep_uncertainty_m, position_type=""):
        """
        Write a GSF_RECORD_HV_NAVIGATION_ERROR record. See
        _encode_hv_navigation_error().

        Untested against a verified GSF file: no sample data containing a
        GSF_RECORD_HV_NAVIGATION_ERROR record is available.
        """
        self.write_record(
            RecordType.GSF_RECORD_HV_NAVIGATION_ERROR,
            _encode_hv_navigation_error(
                nav_error_time, record_id, horizontal_error_m, vertical_error_m,
                sep_uncertainty_m, position_type))

    def write_single_beam_ping(self, record):
        """
        Write a GSF_RECORD_SINGLE_BEAM_PING record. See
        _encode_single_beam_ping() for the expected shape of `record` --
        the same dict shape _decode_single_beam_ping() returns, so a
        decoded ping can be passed straight back in.

        Untested against a verified GSF file: no sample data containing a
        GSF_RECORD_SINGLE_BEAM_PING record is available.
        """
        self.write_record(
            RecordType.GSF_RECORD_SINGLE_BEAM_PING,
            _encode_single_beam_ping(record))


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
        description="A python script (and class) for indexing and reading "
                     "Generic Sensor Format (GSF) data files.",
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
                              "CSV, one row per beam. Decoded for KMALL, EM3-series, "
                              "EM4-series, Reson 7125/T-series/8100-family, Klein 5410 BSS, "
                              "R2Sonic, and any other sensor with no imagery-specific preamble.")
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
