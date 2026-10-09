#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A python class (and command line utility) to index, read, and write Generic
Sensor Format (GSF) sonar data files.

The physical, on-disk record encoding implemented here (the record size
field, the packed data identifier word, and the optional checksum) and the
record type constants/descriptions below are taken from the reference C
implementation of the GSF library ("gsflib"), version 3.11, distributed by
Leidos, Inc. under the LGPL 2.1 from the Leidos product page:

    https://www.leidos.com/products/ocean-marine

GSF v3.11 was released in 2025, but the copyright notices in its source
files were not updated and still read "Copyright 2019 Leidos, Inc.". That
distribution unpacks to a single GSF_03-11/ directory, and every gsflib
source file cited in this module (gsf.h, gsf.c, gsf_dec.c, gsf_enc.c, and
so on) is named relative to that directory's top level. It is the only
copy of gsflib this module was written against; other copies found online
(such as GitHub mirrors) may or may not be identical, and are not
authoritative.

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

The v3.11 distribution's own change summary
(gsf_version_03_11_change_summary.pdf) documents exactly one decode
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
import mmap
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

#: The record framing: the size field, then the packed data identifier word.
_RECORD_FRAMING = struct.Struct('>II')


class FileMode(IntEnum):
    """
    The file access modes a GSF file can be opened with, taken from gsf.h.
    """
    GSF_CREATE = 1
    GSF_READONLY = 2
    GSF_UPDATE = 3
    GSF_READONLY_INDEX = 4
    GSF_UPDATE_INDEX = 5
    GSF_APPEND = 6


class SeekOption(IntEnum):
    """
    The starting points a GSF file's read position can be seeked from,
    taken from gsf.h.
    """
    GSF_REWIND = 1
    GSF_END_OF_FILE = 2
    GSF_PREVIOUS_RECORD = 3


#: Specify a key to allow reading the next record, no matter what it is. (gsf.h: GSF_NEXT_RECORD)
GSF_NEXT_RECORD = 0


class RecordType(IntEnum):
    """
    The numeric type of every record a GSF file can contain, taken from
    gsf.h's GSF_RECORD_* defines. Each record in a file is tagged with one
    of these values.
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

#: RecordType name for each recordID, indexed by recordID, for naming every
#: record of a file in one vectorized lookup. recordID 0 is not a record type.
_RECORD_TYPE_NAMES = np.array(
    ["UNKNOWN(0)"] + [RecordType(i).name for i in range(1, NUM_REC_TYPES)], dtype=object)

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
    """
    The base class for every error raised while reading or writing a GSF
    file. The reference gsflib C library reports these as integer error
    codes through a global variable; this library raises them as
    exceptions instead.
    """


class GSFRecordSizeError(GSFError):
    """
    A record's declared size is not valid: it is eight bytes or smaller,
    or larger than GSF_MAX_RECORD_SIZE. This corresponds to gsf.h's
    GSF_RECORD_SIZE_ERROR.
    """


class GSFUnrecognizedRecordIDError(GSFError):
    """
    A record's type number is not a value between one and NUM_REC_TYPES
    minus one. This corresponds to gsf.h's GSF_UNRECOGNIZED_RECORD_ID.
    """


class GSFPartialRecordAtEndOfFileError(GSFError):
    """
    Fewer bytes remain in the file than the current record declares it
    needs. This corresponds to gsf.h's GSF_PARTIAL_RECORD_AT_END_OF_FILE.
    """


class GSFChecksumFailureError(GSFError):
    """
    A record's stored checksum does not match the checksum computed from
    its data. This corresponds to gsf.h's GSF_CHECKSUM_FAILURE.
    """


def gsf_checksum(data):
    """
    Compute the checksum of a GSF record's payload: the sum of its bytes,
    taken modulo 2**32. This is used to verify, or to generate, the
    optional four-byte checksum that precedes a record's payload when the
    data identifier word's checksum flag is set. This is ported from the
    reference gsflib C library's gsfChecksum() function in gsf.c.

    :param data: the record's payload, as bytes. This must not include
        the checksum field itself.

    :return: the computed checksum, as an integer.
    """
    return sum(data) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# gsfDataID -- gsf.h t_gsfDataID
# ---------------------------------------------------------------------------

@dataclass
class gsfDataID:
    """
    A GSF record's data identifier: the information that says which kind
    of record follows, and a few bits about how it is encoded. This
    mirrors gsf.h's t_gsfDataID structure.

    On disk, a data identifier is packed into a single four-byte,
    big-endian word immediately following a record's four-byte size
    field. The reference gsflib C library unpacks that word (in gsf.c's
    gsfUnpackStream() function) as follows:

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
# _PING_SENSOR_SPECIFIC_CODECS registry, including KMALL_SPECIFIC (id 156)
# -- see the registry's docstring below. Likewise every single-beam
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

#: Maps each column label used in a decoded swath bathymetry ping's 'Beams'
#: table (for example 'Depth_m' or 'BeamFlags') to the GSF subrecord id of
#: the beam array it holds (gsf.h: GSF_SWATH_BATHY_SUBRECORD_*_ARRAY), in
#: subrecord id order. It covers every scaled beam array in
#: _PING_ARRAY_SUBRECORDS, plus the beam flags and quality flags arrays,
#: which are stored without scale factors. Use it to find which subrecord a
#: column came from, or which column a subrecord id is decoded into.
BEAM_ARRAY_SUBRECORD_IDS = dict(sorted(
    [(label, sid) for sid, (_attr, label, _signed) in _PING_ARRAY_SUBRECORDS.items()]
    + [('QualityFlags', _SUBRECORD_QUALITY_FLAGS_ARRAY),
       ('BeamFlags', _SUBRECORD_BEAM_FLAGS_ARRAY)],
    key=lambda item: item[1]))

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
#: decode_fn(payload, pos) -> (record: dict, bytes_consumed). `record` is
#: one flat dict: scalar fields directly, plus, for a family with any
#: per-element arrays (EM3, EM3Raw, EM4, KMALL), those as tables (each a
#: dictionary of column name -> numpy array) under their own table name
#: (e.g. 'TxSectors') -- nested directly in the same dict, not returned
#: separately. encode_fn(record) -> bytes, including its own 4-byte
#: subrecord id+size word -- the exact dict decode_fn returns (or, for a
#: hand-built record, the same shape) can be passed straight back in, no
#: conversion or repackaging.
#:
#: A `decode_fn`/`encode_fn` pair registered under more than one id (e.g.
#: EM4 at 133/134/135/149/157) can't hardcode which one to stamp on
#: encode -- it reads `record['SubrecordID']` (defaulting to that family's
#: first/primary id if absent) instead. A pair registered under exactly
#: one id just hardcodes it and ignores 'SubrecordID' if present. Either
#: way, `_encode_swath_bathymetry_ping()` always sets 'SubrecordID' to
#: match `record['SensorSpecificID']` before calling encode_fn, so a
#: decoded ping's own `SensorSpecific` dict is already self-consistent.
_PING_SENSOR_SPECIFIC_CODECS = {}


def new_elac_mkii_specific():
    """
    Return a new dictionary with every ElacMkII sensor-specific subrecord
    field name present, in the same shape _decode_elac_mkii_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_elac_mkii_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (117).

    :return: a dictionary with every ElacMkII sensor-specific field name as
        a key, pre-set to its default.
    """
    return {
        'Mode': 0, 'PingNumber': 0, 'SoundVelocity_mps': 0, 'PulseLength_hundredth_ms': 0,
        'ReceiverGainStbd_dB': 0, 'ReceiverGainPort_dB': 0, 'Reserved': 0,
    }


def _decode_elac_mkii_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_ELAC_MKII_SPECIFIC subrecord (id
    117), which holds the per-ping metadata reported by an Elac MkII
    multibeam sonar. This is ported from the reference gsflib C library's
    DecodeElacMkIISpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an ELAC_MKII_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    mode = payload[pos]; pos += 1
    (ping_num,) = struct.unpack_from('>H', payload, pos); pos += 2
    (sound_vel,) = struct.unpack_from('>H', payload, pos); pos += 2
    (pulse_length,) = struct.unpack_from('>H', payload, pos); pos += 2
    receiver_gain_stbd = payload[pos]; pos += 1
    receiver_gain_port = payload[pos]; pos += 1
    (reserved,) = struct.unpack_from('>H', payload, pos); pos += 2

    record = new_elac_mkii_specific()
    record['Mode'] = mode
    record['PingNumber'] = ping_num
    record['SoundVelocity_mps'] = sound_vel
    record['PulseLength_hundredth_ms'] = pulse_length
    record['ReceiverGainStbd_dB'] = receiver_gain_stbd
    record['ReceiverGainPort_dB'] = receiver_gain_port
    record['Reserved'] = reserved
    return record, pos - start


def _encode_elac_mkii_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_ELAC_MKII_SPECIFIC subrecord (id
    117), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_elac_mkii_specific(), and is ported from the
    reference gsflib C library's EncodeElacMkIISpecific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_elac_mkii_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', int(g('SoundVelocity_mps', 0)))
    body += struct.pack('>H', int(g('PulseLength_hundredth_ms', 0)))
    body += struct.pack('>B', int(g('ReceiverGainStbd_dB', 0)) & 0xFF)
    body += struct.pack('>B', int(g('ReceiverGainPort_dB', 0)) & 0xFF)
    body += struct.pack('>H', int(g('Reserved', 0)))
    header_word = (117 << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[117] = ("ElacMkII", _decode_elac_mkii_specific, _encode_elac_mkii_specific)


def new_seabeam_specific():
    """
    Return a new dictionary with every SeaBeam sensor-specific subrecord
    field name present, in the same shape _decode_seabeam_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_seabeam_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (102).

    :return: a dictionary with every SeaBeam sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'EclipseTime_tenths_s': 0,
    }


def _decode_seabeam_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_SPECIFIC subrecord (id
    102), which holds the per-ping metadata reported by a 16-beam
    SeaBeam sensor. This is ported from the reference gsflib C library's
    DecodeSeabeamSpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SEABEAM_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (eclipse_time,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = new_seabeam_specific()
    fields['EclipseTime_tenths_s'] = eclipse_time
    return fields, pos - start


def _encode_seabeam_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_SPECIFIC subrecord (id
    102), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_seabeam_specific(), and is ported from the
    reference gsflib C library's EncodeSeabeamSpecific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_seabeam_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    body = struct.pack('>H', int(record.get('EclipseTime_tenths_s', 0)))
    header_word = ((102 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[102] = ("SeaBeam", _decode_seabeam_specific, _encode_seabeam_specific)


def new_em12_specific():
    """
    Return a new dictionary with every EM12 sensor-specific subrecord field
    name present, in the same shape _decode_em12_specific() returns, each
    pre-set to 0 or 0.0. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_em12_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord id (103).

    :return: a dictionary with every EM12 sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'PingNumber': 0, 'Resolution': 0, 'PingQuality': 0, 'SoundVelocity_mps': 0.0,
        'Mode': 0,
    }


def _decode_em12_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM12_SPECIFIC subrecord (id 103),
    which holds the per-ping metadata reported by a Simrad EM12
    multibeam sonar. This is ported from the reference gsflib C library's
    DecodeEM12Specific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM12_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    resolution = payload[pos]; pos += 1
    ping_quality = payload[pos]; pos += 1
    (sound_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    pos += 32  # spare

    fields = new_em12_specific()
    fields['PingNumber'] = ping_number
    fields['Resolution'] = resolution
    fields['PingQuality'] = ping_quality
    fields['SoundVelocity_mps'] = sound_velocity_raw / 10.0
    fields['Mode'] = mode
    return fields, pos - start


def _encode_em12_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM12_SPECIFIC subrecord (id 103),
    including its own four-byte subrecord ID and size word. This is the
    inverse of _decode_em12_specific(), and is ported from the reference
    gsflib C library's EncodeEM12Specific() function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_em12_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>B', int(g('Resolution', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PingQuality', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('SoundVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += b'\x00' * 32  # spare
    header_word = ((103 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[103] = ("EM12", _decode_em12_specific, _encode_em12_specific)


def new_em100_specific():
    """
    Return a new dictionary with every EM100 sensor-specific subrecord field
    name present, in the same shape _decode_em100_specific() returns, each
    pre-set to 0 or 0.0. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_em100_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord id (104).

    :return: a dictionary with every EM100 sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'ShipPitch_deg': 0.0, 'TransducerPitch_deg': 0.0, 'Mode': 0, 'Power': 0,
        'Attenuation': 0, 'TVG': 0, 'PulseLength': 0, 'Counter': 0,
    }


def _decode_em100_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM100_SPECIFIC subrecord (id 104),
    which holds the per-ping metadata reported by a Simrad EM100
    multibeam sonar. This is ported from the reference gsflib C library's
    DecodeEM100Specific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM100_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_em100_specific()
    fields['ShipPitch_deg'] = ship_pitch_raw / 100.0
    fields['TransducerPitch_deg'] = transducer_pitch_raw / 100.0
    fields['Mode'] = mode
    fields['Power'] = power
    fields['Attenuation'] = attenuation
    fields['TVG'] = tvg
    fields['PulseLength'] = pulse_length
    fields['Counter'] = counter
    return fields, pos - start


def _encode_em100_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM100_SPECIFIC subrecord (id 104),
    including its own four-byte subrecord ID and size word. This is the
    inverse of _decode_em100_specific(), and is ported from the reference
    gsflib C library's EncodeEM100Specific() function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_em100_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>h', _gsf_round(g('ShipPitch_deg', 0.0) * 100.0))
    body += struct.pack('>h', _gsf_round(g('TransducerPitch_deg', 0.0) * 100.0))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Power', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Attenuation', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TVG', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PulseLength', 0)) & 0xFF)
    body += struct.pack('>H', int(g('Counter', 0)))
    header_word = ((104 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[104] = ("EM100", _decode_em100_specific, _encode_em100_specific)


def new_cmp_sass_specific():
    """
    Return a new dictionary with every CmpSass sensor-specific subrecord
    field name present, in the same shape _decode_cmp_sass_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_cmp_sass_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (121).

    :return: a dictionary with every CmpSass sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'SurfaceSoundVelocity_ftps': 0.0, 'Heave_ftps': 0.0,
    }


def _decode_cmp_sass_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_CMP_SASS_SPECIFIC subrecord (id
    121), which holds the per-ping metadata reported by a Compressed
    SASS (BOSDAT) sensor. This is ported from the reference gsflib C
    library's DecodeCmpSassSpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a CMP_SASS_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (lfreq_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    (lntens_raw,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = new_cmp_sass_specific()
    fields['SurfaceSoundVelocity_ftps'] = lfreq_raw / 10.0
    fields['Heave_ftps'] = lntens_raw / 10.0
    return fields, pos - start


def _encode_cmp_sass_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_CMP_SASS_SPECIFIC subrecord (id
    121), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_cmp_sass_specific(), and is ported from the
    reference gsflib C library's EncodeCmpSassSpecific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_cmp_sass_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>H', _gsf_round(g('SurfaceSoundVelocity_ftps', 0.0) * 10.0))
    body += struct.pack('>H', _gsf_round(g('Heave_ftps', 0.0) * 10.0))
    header_word = ((121 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[121] = ("CmpSass", _decode_cmp_sass_specific, _encode_cmp_sass_specific)


def new_em950_specific():
    """
    Return a new dictionary with every EM950 sensor-specific subrecord field
    name present, in the same shape _decode_em950_specific() returns, each
    pre-set to 0 or 0.0. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_em950_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord ids (105, 111).

    :return: a dictionary with every EM950 sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'PingNumber': 0, 'Mode': 0, 'PingQuality': 0, 'ShipPitch_deg': 0.0,
        'TransducerPitch_deg': 0.0, 'SurfaceVelocity_mps': 0.0,
    }


def _decode_em950_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM950_SPECIFIC or
    GSF_SWATH_BATHY_SUBRECORD_EM1000_SPECIFIC subrecord (ids 105 and 111,
    for a Simrad EM950 or EM1000 multibeam sonar respectively). Both
    subrecords share an identical wire format, so gsflib uses one struct
    and one decoder for both, and this function does the same. This is
    ported from the reference gsflib C library's DecodeEM950Specific()
    function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM950_SPECIFIC or EM1000_SPECIFIC
    subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    (ping_quality,) = struct.unpack_from('>b', payload, pos); pos += 1
    (ship_pitch_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (transducer_pitch_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = new_em950_specific()
    fields['PingNumber'] = ping_number
    fields['Mode'] = mode
    fields['PingQuality'] = ping_quality
    fields['ShipPitch_deg'] = ship_pitch_raw / 100.0
    fields['TransducerPitch_deg'] = transducer_pitch_raw / 100.0
    fields['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    return fields, pos - start


def _encode_em950_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM950_SPECIFIC or
    GSF_SWATH_BATHY_SUBRECORD_EM1000_SPECIFIC subrecord, including its
    own four-byte subrecord ID and size word. The record's 'SubrecordID'
    field selects which of the two subrecord IDs (105 for EM950 or 111
    for EM1000) is written into the header; it defaults to 105 if not
    given. This is the inverse of _decode_em950_specific(), and is
    ported from the reference gsflib C library's EncodeEM950Specific()
    function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_em950_specific() returns, plus an optional
        'SubrecordID' entry. A field that is missing from the dictionary
        is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 105)
    g = record.get
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


def new_em121a_specific():
    """
    Return a new dictionary with every EM121A sensor-specific subrecord
    field name present, in the same shape _decode_em121a_specific() returns,
    each pre-set to 0 or 0.0. gsf.h defines no "not available" value for
    these vendor-specific fields, so these are the only defaults available.
    _decode_em121a_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord ids (106, 107).

    :return: a dictionary with every EM121A sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'PingNumber': 0, 'Mode': 0, 'ValidBeams': 0, 'PulseLength': 0, 'BeamWidth': 0,
        'TxPower': 0, 'TxStatus': 0, 'RxStatus': 0, 'SurfaceVelocity_mps': 0.0,
    }


def _decode_em121a_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM121A_SPECIFIC or
    GSF_SWATH_BATHY_SUBRECORD_EM121_SPECIFIC subrecord (ids 106 and 107,
    for a Simrad EM121A or EM121 multibeam sonar respectively). Both
    subrecords share an identical wire format, so gsflib uses one struct
    and one decoder for both, and this function does the same. This is
    ported from the reference gsflib C library's DecodeEM121ASpecific()
    function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM121A_SPECIFIC or EM121_SPECIFIC
    subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_em121a_specific()
    fields['PingNumber'] = ping_number
    fields['Mode'] = mode
    fields['ValidBeams'] = valid_beams
    fields['PulseLength'] = pulse_length
    fields['BeamWidth'] = beam_width
    fields['TxPower'] = tx_power
    fields['TxStatus'] = tx_status
    fields['RxStatus'] = rx_status
    fields['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    return fields, pos - start


def _encode_em121a_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM121A_SPECIFIC or
    GSF_SWATH_BATHY_SUBRECORD_EM121_SPECIFIC subrecord, including its
    own four-byte subrecord ID and size word. The record's 'SubrecordID'
    field selects which of the two subrecord IDs (106 for EM121A or 107
    for EM121) is written into the header; it defaults to 106 if not
    given. This is the inverse of _decode_em121a_specific(), and is
    ported from the reference gsflib C library's EncodeEM121ASpecific()
    function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_em121a_specific() returns, plus an optional
        'SubrecordID' entry. A field that is missing from the dictionary
        is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 106)
    g = record.get
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


def new_seamap_specific():
    """
    Return a new dictionary with every SeaMap sensor-specific subrecord
    field name present, in the same shape _decode_seamap_specific() returns,
    each pre-set to 0 or 0.0. gsf.h defines no "not available" value for
    these vendor-specific fields, so these are the only defaults available.
    _decode_seamap_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord id (109).

    :return: a dictionary with every SeaMap sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'PortTransmitter0': 0.0, 'PortTransmitter1': 0.0, 'StbdTransmitter0': 0.0,
        'StbdTransmitter1': 0.0, 'PortGain': 0.0, 'StbdGain': 0.0, 'PortPulseLength': 0.0,
        'StbdPulseLength': 0.0, 'PressureDepth': 0.0, 'Altitude': 0.0, 'Temperature': 0.0,
    }


def _decode_seamap_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEAMAP_SPECIFIC subrecord (id
    109), which holds the per-ping metadata reported by a SeaMap or
    SeaMap-II swath interferometric sonar. This is ported from the
    reference gsflib C library's DecodeSeaMapSpecific() function in
    gsf_dec.c.

    The reference gsflib function takes an extra GSF_FILE_TABLE pointer
    argument that exists only to gate one historical bug fix. In GSF
    files written by versions of gsflib older than v2.7, the code failed
    to advance its read position past the pressureDepth field (a
    documented bug, noted by the "JSB 11/08/2007" comment in gsf_dec.c
    and gsf_enc.c), so on those old files pressureDepth's raw bytes were
    immediately overwritten when altitude was decoded next. This Python
    library does not track per-file gsflib-version state the way
    gsflib's GSF_FILE_TABLE does, and it only ever writes modern files
    (at GSF_VERSION, which is newer than v2.7), so this function always
    takes the "fixed" code path and advances past pressureDepth
    normally. That is correct for any file this library writes itself,
    and for any real file produced by a non-ancient version of gsflib,
    but it is not bug-for-bug compatible with a SeaMap subrecord read
    from a GSF file older than v2.7.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SEAMAP_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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
    fields = new_seamap_specific()
    fields.update((key, value / 10.0) for key, value in zip(keys, values))
    return fields, pos - start


def _encode_seamap_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEAMAP_SPECIFIC subrecord (id
    109), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_seamap_specific(); see that function's
    docstring for the pre-v2.7 pressureDepth quirk in the reference
    gsflib C library that this encoder does not replicate. This is
    ported from the reference gsflib C library's EncodeSeaMapSpecific()
    function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_seamap_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    keys = (
        'PortTransmitter0', 'PortTransmitter1', 'StbdTransmitter0', 'StbdTransmitter1',
        'PortGain', 'StbdGain', 'PortPulseLength', 'StbdPulseLength',
        'PressureDepth', 'Altitude', 'Temperature',
    )
    body = b''.join(struct.pack('>H', _gsf_round(g(key, 0.0) * 10.0)) for key in keys)
    header_word = ((109 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[109] = ("SeaMap", _decode_seamap_specific, _encode_seamap_specific)


def new_seabat_specific():
    """
    Return a new dictionary with every SeaBat sensor-specific subrecord
    field name present, in the same shape _decode_seabat_specific() returns,
    each pre-set to 0 or 0.0. gsf.h defines no "not available" value for
    these vendor-specific fields, so these are the only defaults available.
    _decode_seabat_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord id (110).

    :return: a dictionary with every SeaBat sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'PingNumber': 0, 'SurfaceVelocity_mps': 0.0, 'Mode': 0, 'SonarRange_m': 0,
        'TransmitPower': 0, 'ReceiveGain': 0,
    }


def _decode_seabat_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_SPECIFIC subrecord (id
    110), which holds the per-ping metadata reported by a Reson SeaBat
    sensor. This is ported from the reference gsflib C library's
    DecodeSeaBatSpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SEABAT_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (surface_velocity_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    mode = payload[pos]; pos += 1
    sonar_range = payload[pos]; pos += 1
    transmit_power = payload[pos]; pos += 1
    receive_gain = payload[pos]; pos += 1

    fields = new_seabat_specific()
    fields['PingNumber'] = ping_number
    fields['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    fields['Mode'] = mode
    fields['SonarRange_m'] = sonar_range
    fields['TransmitPower'] = transmit_power
    fields['ReceiveGain'] = receive_gain
    return fields, pos - start


def _encode_seabat_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_SPECIFIC subrecord (id
    110), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_seabat_specific(), and is ported from the
    reference gsflib C library's EncodeSeaBatSpecific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_seabat_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>B', int(g('SonarRange_m', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TransmitPower', 0)) & 0xFF)
    body += struct.pack('>B', int(g('ReceiveGain', 0)) & 0xFF)
    header_word = ((110 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[110] = ("SeaBat", _decode_seabat_specific, _encode_seabat_specific)


def new_sb_amp_specific():
    """
    Return a new dictionary with every SBAmp sensor-specific subrecord field
    name present, in the same shape _decode_sb_amp_specific() returns, each
    pre-set to 0 or 0.0. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_sb_amp_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord id (113).

    :return: a dictionary with every SBAmp sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'Hour': 0, 'Minute': 0, 'Second': 0, 'Hundredths': 0, 'BlockNumber': 0,
        'AvgGateDepth': 0,
    }


def _decode_sb_amp_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SB_AMP_SPECIFIC subrecord (id
    113), which holds the per-ping metadata reported by a SeaBeam
    amplitude sensor. This is ported from the reference gsflib C
    library's DecodeSBAmpSpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SB_AMP_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    hour = payload[pos]; pos += 1
    minute = payload[pos]; pos += 1
    second = payload[pos]; pos += 1
    hundredths = payload[pos]; pos += 1
    (block_number,) = struct.unpack_from('>I', payload, pos); pos += 4
    (avg_gate_depth,) = struct.unpack_from('>h', payload, pos); pos += 2

    fields = new_sb_amp_specific()
    fields['Hour'] = hour
    fields['Minute'] = minute
    fields['Second'] = second
    fields['Hundredths'] = hundredths
    fields['BlockNumber'] = block_number
    fields['AvgGateDepth'] = avg_gate_depth
    return fields, pos - start


def _encode_sb_amp_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SB_AMP_SPECIFIC subrecord (id
    113), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_sb_amp_specific(), and is ported from the
    reference gsflib C library's EncodeSBAmpSpecific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_sb_amp_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>B', int(g('Hour', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Minute', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Second', 0)) & 0xFF)
    body += struct.pack('>B', int(g('Hundredths', 0)) & 0xFF)
    body += struct.pack('>I', int(g('BlockNumber', 0)))
    body += struct.pack('>h', int(g('AvgGateDepth', 0)))
    header_word = ((113 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[113] = ("SBAmp", _decode_sb_amp_specific, _encode_sb_amp_specific)


def new_seabat_ii_specific():
    """
    Return a new dictionary with every SeaBatII sensor-specific subrecord
    field name present, in the same shape _decode_seabat_ii_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_seabat_ii_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (114).

    :return: a dictionary with every SeaBatII sensor-specific field name as
        a key, pre-set to its default.
    """
    return {
        'PingNumber': 0, 'SurfaceVelocity_mps': 0.0, 'Mode': 0, 'SonarRange_m': 0,
        'TransmitPower': 0, 'ReceiveGain': 0, 'ForeAftBW_deg': 0.0, 'AthwartBW_deg': 0.0,
    }


def _decode_seabat_ii_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_II_SPECIFIC subrecord (id
    114), which holds the per-ping metadata reported by a Reson SeaBat
    II sensor. This subrecord replaced SEABAT_SPECIFIC as of GSF version
    1.04. This is ported from the reference gsflib C library's
    DecodeSeaBatIISpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SEABAT_II_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_seabat_ii_specific()
    fields['PingNumber'] = ping_number
    fields['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    fields['Mode'] = mode
    fields['SonarRange_m'] = sonar_range
    fields['TransmitPower'] = transmit_power
    fields['ReceiveGain'] = receive_gain
    fields['ForeAftBW_deg'] = fore_aft_bw / 10.0
    fields['AthwartBW_deg'] = athwart_bw / 10.0
    return fields, pos - start


def _encode_seabat_ii_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_II_SPECIFIC subrecord (id
    114), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_seabat_ii_specific(), and is ported from the
    reference gsflib C library's EncodeSeaBatIISpecific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_seabat_ii_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>H', int(g('PingNumber', 0)))
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 0.0) * 10.0))
    body += struct.pack('>H', int(g('Mode', 0)))
    body += struct.pack('>H', int(g('SonarRange_m', 0)))
    body += struct.pack('>H', int(g('TransmitPower', 0)))
    body += struct.pack('>H', int(g('ReceiveGain', 0)))
    body += struct.pack('>B', _gsf_round(g('ForeAftBW_deg', 0.0) * 10.0) & 0xFF)
    body += struct.pack('>B', _gsf_round(g('AthwartBW_deg', 0.0) * 10.0) & 0xFF)
    body += b'\x00' * 4  # spare
    header_word = ((114 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[114] = ("SeaBatII", _decode_seabat_ii_specific, _encode_seabat_ii_specific)


def new_seabeam_2112_specific():
    """
    Return a new dictionary with every SeaBeam2112 sensor-specific subrecord
    field name present, in the same shape _decode_seabeam_2112_specific()
    returns, each pre-set to 0 or 0.0, or an empty string. gsf.h defines no
    "not available" value for these vendor-specific fields, so these are the
    only defaults available. _decode_seabeam_2112_specific() builds its
    result starting from this same template, so the two can never define a
    different set of field names. To write one, fill it in and set it as a
    ping record's 'SensorSpecific', with 'SensorSpecificID' set to this
    family's subrecord id (116). SurfaceVelocity_mps is stored on the wire
    as an unsigned offset from 1300 m/s, so its default is 1300.0, the
    lowest value it can hold, rather than 0.0.

    :return: a dictionary with every SeaBeam2112 sensor-specific field name
        as a key, pre-set to its default.
    """
    return {
        'Mode': 0, 'SurfaceVelocity_mps': 1300.0, 'SsvSource': 0, 'PingGain_dB': 0,
        'PulseWidth_ms': 0, 'TransmitterAttenuation_dB': 0, 'NumberAlgorithms': 0,
        'AlgorithmOrder': '',
    }


def _decode_seabeam_2112_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_2112_SPECIFIC subrecord
    (id 116), which holds the per-ping metadata reported by a SeaBeam
    2112/36 sensor. This is ported from the reference gsflib C library's
    DecodeSeaBeam2112Specific() function in gsf_dec.c.

    The decoded 'Mode' field is left as the raw bitmask read from the
    file (see the GSF_2112_* bitmask macros defined in gsf.h just after
    the t_gsfSeaBeam2112Specific struct), rather than being split out
    into separate boolean fields. This matches how this library treats
    other bitmask fields elsewhere, such as KMALL's PingFlags.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SEABEAM_2112_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_seabeam_2112_specific()
    fields['Mode'] = mode
    fields['SurfaceVelocity_mps'] = (surface_velocity_raw + 130000) / 100.0
    fields['SsvSource'] = ssv_source
    fields['PingGain_dB'] = ping_gain
    fields['PulseWidth_ms'] = pulse_width
    fields['TransmitterAttenuation_dB'] = transmitter_attenuation
    fields['NumberAlgorithms'] = number_algorithms
    fields['AlgorithmOrder'] = algorithm_order
    return fields, pos - start


def _encode_seabeam_2112_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABEAM_2112_SPECIFIC subrecord
    (id 116), including its own four-byte subrecord ID and size word.
    This is the inverse of _decode_seabeam_2112_specific(), and is
    ported from the reference gsflib C library's
    EncodeSeaBeam2112Specific() function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_seabeam_2112_specific() returns. A field that
        is missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>B', int(g('Mode', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('SurfaceVelocity_mps', 1300.0) * 100.0 - 130000))
    body += struct.pack('>B', int(g('SsvSource', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PingGain_dB', 0)) & 0xFF)
    body += struct.pack('>B', int(g('PulseWidth_ms', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TransmitterAttenuation_dB', 0)) & 0xFF)
    body += struct.pack('>B', int(g('NumberAlgorithms', 0)) & 0xFF)
    body += g('AlgorithmOrder', '').encode('ascii')[:4].ljust(4, b'\x00')
    body += b'\x00' * 2  # spare
    header_word = ((116 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[116] = ("SeaBeam2112", _decode_seabeam_2112_specific, _encode_seabeam_2112_specific)


def new_seabat8101_specific():
    """
    Return a new dictionary with every SeaBat8101 sensor-specific subrecord
    field name present, in the same shape _decode_seabat8101_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_seabat8101_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (115).

    :return: a dictionary with every SeaBat8101 sensor-specific field name
        as a key, pre-set to its default.
    """
    return {
        'PingNumber': 0, 'SurfaceVelocity_mps': 0.0, 'Mode': 0, 'Range_m': 0, 'Power': 0,
        'Gain': 0, 'PulseWidth_us': 0, 'TvgSpreading': 0, 'TvgAbsorption': 0,
        'ForeAftBW_deg': 0.0, 'AthwartBW_deg': 0.0, 'RangeFiltMin': 0, 'RangeFiltMax': 0,
        'DepthFiltMin': 0, 'DepthFiltMax': 0, 'Projector': 0,
    }


def _decode_seabat8101_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_8101_SPECIFIC subrecord
    (id 115), which holds the per-ping metadata reported by a Reson
    SeaBat 8101 sensor. This is ported from the reference gsflib C
    library's DecodeSeaBat8101Specific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SEABAT_8101_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_seabat8101_specific()
    fields['PingNumber'] = ping_number
    fields['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    fields['Mode'] = mode
    fields['Range_m'] = rng
    fields['Power'] = power
    fields['Gain'] = gain
    fields['PulseWidth_us'] = pulse_width
    fields['TvgSpreading'] = tvg_spreading
    fields['TvgAbsorption'] = tvg_absorption
    fields['ForeAftBW_deg'] = fore_aft_bw_raw / 10.0
    fields['AthwartBW_deg'] = athwart_bw_raw / 10.0
    fields['RangeFiltMin'] = range_filt_min
    fields['RangeFiltMax'] = range_filt_max
    fields['DepthFiltMin'] = depth_filt_min
    fields['DepthFiltMax'] = depth_filt_max
    fields['Projector'] = projector
    return fields, pos - start


def _encode_seabat8101_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SEABAT_8101_SPECIFIC subrecord
    (id 115), including its own four-byte subrecord ID and size word.
    This is the inverse of _decode_seabat8101_specific(), and is ported
    from the reference gsflib C library's EncodeSeaBat8101Specific()
    function in gsf_enc.c.

    The reference C encoder rounds the fore-aft and athwart beam-width
    fields using a plain "add 0.5 and truncate" cast, rather than the
    +/-0.501 rounding convention this library uses everywhere else.
    Because these two fields are never negative, the two conventions are
    functionally equivalent for them, so this function uses this
    library's standard _gsf_round() convention for consistency with the
    rest of the module instead of replicating the reference cast.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_seabat8101_specific() returns. A field that
        is missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
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
    header_word = ((115 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[115] = ("SeaBat8101", _decode_seabat8101_specific, _encode_seabat8101_specific)


def new_reson8100_specific():
    """
    Return a new dictionary with every Reson8100 sensor-specific subrecord
    field name present, in the same shape _decode_reson8100_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_reson8100_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    ids (122, 123, 124, 125, 126, 127).

    :return: a dictionary with every Reson8100 sensor-specific field name as
        a key, pre-set to its default.
    """
    return {
        'Latency_ms': 0, 'PingNumber': 0, 'SonarID': 0, 'SonarModel': 0, 'Frequency_kHz': 0,
        'SurfaceVelocity_mps': 0.0, 'SampleRate_Hz': 0, 'PingRate_mHz': 0, 'Mode': 0,
        'Range_m': 0, 'Power': 0, 'Gain': 0, 'PulseWidth_us': 0, 'TvgSpreading': 0,
        'TvgAbsorption': 0, 'ForeAftBW_deg': 0.0, 'AthwartBW_deg': 0.0, 'ProjectorType': 0,
        'ProjectorAngle': 0, 'RangeFiltMin': 0, 'RangeFiltMax': 0, 'DepthFiltMin': 0,
        'DepthFiltMax': 0, 'FiltersActive': 0, 'Temperature_tenth_degC': 0,
        'BeamSpacing_deg': 0.0,
    }


def _decode_reson8100_specific(payload, pos):
    """
    Decode a Reson 8100-family sensor-specific subrecord: one of
    GSF_SWATH_BATHY_SUBRECORD_RESON_8101_SPECIFIC,
    _RESON_8111_SPECIFIC, _RESON_8124_SPECIFIC, _RESON_8125_SPECIFIC,
    _RESON_8150_SPECIFIC or _RESON_8160_SPECIFIC (ids 122 through 127).
    All six models share an identical wire format, so gsflib uses one
    struct and one decoder for the whole family, and this function does
    the same. This is ported from the reference gsflib C library's
    DecodeReson8100Specific() function in gsf_dec.c.

    The 'ProjectorAngle' field is kept as the raw on-disk value, an
    integer scaled by 100 (that is, hundredths of a degree), rather than
    being divided down to plain degrees. This matches the reference
    gsflib decoder itself, whose struct field is a plain C `int` for
    this value, unlike fields such as surface velocity or the fore-aft
    and athwart beam widths, which the reference decoder does convert
    into descaled `double` fields, despite a comment in the C source
    that describes ProjectorAngle as if it were also descaled. This
    function preserves that raw, undescaled representation for fidelity
    with the reference decoder's actual behavior.

    This function has not been tested against a verified GSF file, since
    no sample data containing a Reson 8100-family SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_reson8100_specific()
    fields['Latency_ms'] = latency
    fields['PingNumber'] = ping_number
    fields['SonarID'] = sonar_id
    fields['SonarModel'] = sonar_model
    fields['Frequency_kHz'] = frequency
    fields['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    fields['SampleRate_Hz'] = sample_rate
    fields['PingRate_mHz'] = ping_rate
    fields['Mode'] = mode
    fields['Range_m'] = rng
    fields['Power'] = power
    fields['Gain'] = gain
    fields['PulseWidth_us'] = pulse_width
    fields['TvgSpreading'] = tvg_spreading
    fields['TvgAbsorption'] = tvg_absorption
    fields['ForeAftBW_deg'] = fore_aft_bw_raw / 10.0
    fields['AthwartBW_deg'] = athwart_bw_raw / 10.0
    fields['ProjectorType'] = projector_type
    fields['ProjectorAngle'] = projector_angle
    fields['RangeFiltMin'] = range_filt_min
    fields['RangeFiltMax'] = range_filt_max
    fields['DepthFiltMin'] = depth_filt_min
    fields['DepthFiltMax'] = depth_filt_max
    fields['FiltersActive'] = filters_active
    fields['Temperature_tenth_degC'] = temperature
    fields['BeamSpacing_deg'] = beam_spacing_raw / 10000.0
    return fields, pos - start


def _encode_reson8100_specific(record):
    """
    Encode a Reson 8100-family sensor-specific subrecord: one of
    GSF_SWATH_BATHY_SUBRECORD_RESON_8101_SPECIFIC,
    _RESON_8111_SPECIFIC, _RESON_8124_SPECIFIC, _RESON_8125_SPECIFIC,
    _RESON_8150_SPECIFIC or _RESON_8160_SPECIFIC (ids 122 through 127),
    including its own four-byte subrecord ID and size word. The record's
    'SubrecordID' field selects which of the six family member IDs is
    written into the header; it defaults to 122 (RESON_8101) if not
    given. This is the inverse of _decode_reson8100_specific(), and is
    ported from the reference gsflib C library's
    EncodeReson8100Specific() function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_reson8100_specific() returns, plus an
        optional 'SubrecordID' entry. A field that is missing from the
        dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 122)
    g = record.get
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


def new_reson7125_specific():
    """
    Return a new dictionary with every Reson7125 sensor-specific subrecord
    field name present, in the same shape _decode_reson7125_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_reson7125_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (138).

    :return: a dictionary with every Reson7125 sensor-specific field name as
        a key, pre-set to its default.
    """
    return {
        'ProtocolVersion': 0, 'DeviceID': 0, 'MajorSerialNumber': 0, 'MinorSerialNumber': 0,
        'PingNumber': 0, 'MultiPingSeq': 0, 'Frequency_Hz': 0.0, 'SampleRate_Hz': 0.0,
        'ReceiverBandwidth_Hz': 0.0, 'TxPulseWidth_s': 0.0, 'TxPulseTypeID': 0,
        'TxPulseEnvelopeID': 0, 'TxPulseEnvelopeParam': 0.0, 'TxPulseReserved': 0,
        'MaxPingRate_pps': 0.0, 'PingPeriod_s': 0.0, 'Range_m': 0.0, 'Power_dB': 0.0,
        'Gain_dB': 0.0, 'ControlFlags': 0, 'ProjectorID': 0,
        'ProjectorSteerAnglVert_deg': 0.0, 'ProjectorSteerAnglHoriz_deg': 0.0,
        'ProjectorBeamWidthVert_deg': 0.0, 'ProjectorBeamWidthHoriz_deg': 0.0,
        'ProjectorBeamFocalPt_m': 0.0, 'ProjectorBeamWeightingWindowType': 0,
        'ProjectorBeamWeightingWindowParam': 0, 'TransmitFlags': 0, 'HydrophoneID': 0,
        'ReceivingBeamWeightingWindowType': 0, 'ReceivingBeamWeightingWindowParam': 0,
        'ReceiveFlags': 0, 'ReceiveBeamWidth_deg': 0.0, 'RangeFiltMin_m': 0.0,
        'RangeFiltMax_m': 0.0, 'DepthFiltMin_m': 0.0, 'DepthFiltMax_m': 0.0,
        'Absorption_dBkm': 0.0, 'SoundVelocity_mps': 0.0, 'Spreading_dB': 0.0,
        'RawDataFrom7027': 0, 'SvSource': 0, 'LayerCompFlag': 0,
    }


def _decode_reson7125_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_RESON_7125_SPECIFIC subrecord (id
    138), which holds the per-ping metadata reported by a Reson 7125
    multibeam sonar. This is ported from the reference gsflib C library's
    DecodeReson7100Specific() function in gsf_dec.c. The bitmask fields
    (ControlFlags, TransmitFlags, and ReceiveFlags) are kept as a single
    raw integer rather than being split into individual boolean fields,
    matching how this codebase handles similar bitmask fields elsewhere
    (for example, SeaBeam 2112's Mode field).

    The reference gsflib C library's own DecodeReson7100Specific()
    function has a latent bug in how it reads the tx_pulse_reserved
    field: it reads that field's value from a stale two-byte local
    variable left over from an earlier field, instead of from the
    four-byte value it just loaded for this field, so the real reference
    decoder misreads it. Because this field is documented as reserved and
    unused, this port reads the correct four bytes from their actual wire
    position instead of reproducing the bug, since there is no meaningful
    data to lose either way.

    This function has not been tested against a verified GSF file, since
    no sample data containing a RESON_7125_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_reson7125_specific()
    fields['ProtocolVersion'] = protocol_version
    fields['DeviceID'] = device_id
    fields['MajorSerialNumber'] = major_serial_number
    fields['MinorSerialNumber'] = minor_serial_number
    fields['PingNumber'] = ping_number
    fields['MultiPingSeq'] = multi_ping_seq
    fields['Frequency_Hz'] = frequency_raw / 1.0e3
    fields['SampleRate_Hz'] = sample_rate_raw / 1.0e4
    fields['ReceiverBandwidth_Hz'] = receiver_bandwidth_raw / 1.0e4
    fields['TxPulseWidth_s'] = tx_pulse_width_raw / 1.0e7
    fields['TxPulseTypeID'] = tx_pulse_type_id
    fields['TxPulseEnvelopeID'] = tx_pulse_envlp_id
    fields['TxPulseEnvelopeParam'] = tx_pulse_envlp_param_raw / 1.0e2
    fields['TxPulseReserved'] = tx_pulse_reserved
    fields['MaxPingRate_pps'] = max_ping_rate_raw / 1.0e6
    fields['PingPeriod_s'] = ping_period_raw / 1.0e6
    fields['Range_m'] = range_raw / 1.0e2
    fields['Power_dB'] = power_raw / 1.0e2
    fields['Gain_dB'] = gain_raw / 1.0e2
    fields['ControlFlags'] = control_flags
    fields['ProjectorID'] = projector_id
    fields['ProjectorSteerAnglVert_deg'] = projector_steer_vert_raw / 1.0e3
    fields['ProjectorSteerAnglHoriz_deg'] = projector_steer_horz_raw / 1.0e3
    fields['ProjectorBeamWidthVert_deg'] = projector_bw_vert_raw / 1.0e2
    fields['ProjectorBeamWidthHoriz_deg'] = projector_bw_horz_raw / 1.0e2
    fields['ProjectorBeamFocalPt_m'] = projector_focal_pt_raw / 1.0e2
    fields['ProjectorBeamWeightingWindowType'] = projector_weight_window_type
    fields['ProjectorBeamWeightingWindowParam'] = projector_weight_window_param
    fields['TransmitFlags'] = transmit_flags
    fields['HydrophoneID'] = hydrophone_id
    fields['ReceivingBeamWeightingWindowType'] = rx_weight_window_type
    fields['ReceivingBeamWeightingWindowParam'] = rx_weight_window_param
    fields['ReceiveFlags'] = receive_flags
    fields['ReceiveBeamWidth_deg'] = rx_beam_width_raw / 1.0e2
    fields['RangeFiltMin_m'] = range_filt_min_raw / 1.0e1
    fields['RangeFiltMax_m'] = range_filt_max_raw / 1.0e1
    fields['DepthFiltMin_m'] = depth_filt_min_raw / 1.0e1
    fields['DepthFiltMax_m'] = depth_filt_max_raw / 1.0e1
    fields['Absorption_dBkm'] = absorption_raw / 1.0e3
    fields['SoundVelocity_mps'] = sound_velocity_raw / 1.0e1
    fields['Spreading_dB'] = spreading_raw / 1.0e3
    fields['RawDataFrom7027'] = raw_data_from_7027
    fields['SvSource'] = sv_source
    fields['LayerCompFlag'] = layer_comp_flag
    return fields, pos - start


def _encode_reson7125_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_RESON_7125_SPECIFIC subrecord (id
    138), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_reson7125_specific(), and is ported from the
    reference gsflib C library's EncodeReson7100Specific() function in
    gsf_enc.c. The reference encoder rounds most scaled fields (including
    Frequency_Hz, SampleRate_Hz, ReceiverBandwidth_Hz, TxPulseWidth_s,
    TxPulseEnvelopeParam, MaxPingRate_pps, PingPeriod_s, Range_m, and the
    beam-width, filter, absorption, sound-velocity, and spreading fields)
    with an unconditional "+0.501", because those fields are always
    non-negative in practice, while it rounds Power_dB, Gain_dB, and the
    projector steering angles with a sign-aware "+/-0.501" branch,
    because those fields can be negative. Both approaches are equivalent
    to this module's standard _gsf_round() helper, which this function
    uses uniformly for every scaled field.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_reson7125_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
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
    header_word = ((138 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[138] = ("Reson7125", _decode_reson7125_specific, _encode_reson7125_specific)


def new_reson_tseries_specific():
    """
    Return a new dictionary with every ResonTSeries sensor-specific
    subrecord field name present, in the same shape
    _decode_reson_tseries_specific() returns, each pre-set to 0 or 0.0, or
    an empty string. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_reson_tseries_specific() builds its result starting from this
    same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (155).

    :return: a dictionary with every ResonTSeries sensor-specific field name
        as a key, pre-set to its default.
    """
    return {
        'ProtocolVersion': 0, 'DeviceID': 0, 'NumberDevices': 0, 'SystemEnumerator': 0,
        'MajorSerialNumber': 0, 'MinorSerialNumber': 0, 'PingNumber': 0, 'MultiPingSeq': 0,
        'Frequency_Hz': 0.0, 'SampleRate_Hz': 0.0, 'ReceiverBandwidth_Hz': 0.0,
        'TxPulseWidth_s': 0.0, 'TxPulseTypeID': 0, 'TxPulseEnvelopeID': 0,
        'TxPulseEnvelopeParam': 0.0, 'TxPulseMode': 0, 'TxPulseReserved': 0,
        'MaxPingRate_pps': 0.0, 'PingPeriod_s': 0.0, 'Range_m': 0.0, 'Power_dB': 0.0,
        'Gain_dB': 0.0, 'ControlFlags': 0, 'ProjectorID': 0,
        'ProjectorSteerAnglVert_deg': 0.0, 'ProjectorSteerAnglHoriz_deg': 0.0,
        'ProjectorBeamWidthVert_deg': 0.0, 'ProjectorBeamWidthHoriz_deg': 0.0,
        'ProjectorBeamFocalPt_m': 0.0, 'ProjectorBeamWeightingWindowType': 0,
        'ProjectorBeamWeightingWindowParam': 0, 'TransmitFlags': 0, 'HydrophoneID': 0,
        'ReceivingBeamWeightingWindowType': 0, 'ReceivingBeamWeightingWindowParam': 0,
        'ReceiveFlags': 0, 'ReceiveBeamWidth_deg': 0.0, 'RangeFiltMin_m': 0.0,
        'RangeFiltMax_m': 0.0, 'DepthFiltMin_m': 0.0, 'DepthFiltMax_m': 0.0,
        'Absorption_dBkm': 0.0, 'SoundVelocity_mps': 0.0, 'SvSource': 0, 'Spreading_dB': 0.0,
        'BeamSpacingMode': 0, 'SonarSourceMode': 0, 'CoverageMode': 0,
        'CoverageAngle_deg': 0.0, 'HorizontalReceiverSteeringAngle_deg': 0.0,
        'UncertaintyType': 0, 'TransmitterSteeringAngle_rad': 0.0, 'AppliedRoll_rad': 0.0,
        'DetectionAlgorithm': 0, 'DetectionFlags': 0, 'DeviceDescription': '',
        'MatchFilterControl': 0, 'MatchFilterStartFreq_Hz': 0.0, 'MatchFilterEndFreq_Hz': 0.0,
        'MatchFilterWindowType': 0, 'MatchFilterShadingValue': 0.0,
        'MatchFilterEffectivePulseWidth_s': 0.0,
    }


def _decode_reson_tseries_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_RESON_TSERIES_SPECIFIC subrecord
    (id 155), which holds the per-ping metadata reported by a Reson
    T20/T50-series multibeam sonar. This is ported from the reference
    gsflib C library's DecodeResonTSeriesSpecific() function in
    gsf_dec.c. This is the largest sensor-specific structure defined in
    gsf.h, with roughly ninety named fields plus large spare byte ranges,
    totaling 715 bytes on the wire; unlike the EM3/EM4 subrecords, it has
    no nested arrays or bitmask-gated sub-blocks, so it is simply one
    large flat structure that is read straight through. The bitmask
    fields (ControlFlags, TransmitFlags, ReceiveFlags, and
    DetectionFlags) are kept as a single raw integer rather than split
    into individual boolean fields, matching how this codebase handles
    similar bitmask fields elsewhere.

    The wire format genuinely stores the sound velocity twice, which is
    not a bug but an intentional quirk of the format: once as a two-byte
    low-precision value (scaled by 10) immediately after Absorption_dBkm,
    and again as a four-byte high-precision value (scaled by 1.0e6)
    immediately after DeviceDescription. The reference gsflib decoder
    keeps the high-precision value only when it is nonzero, overwriting
    the low-precision reading in that case, and this function does the
    same, exposing only the single resolved 'SoundVelocity_mps' field
    rather than both raw copies. _encode_reson_tseries_specific() writes
    both wire copies back out from that one field, mirroring the
    reference encoder exactly, so no information is lost in either
    direction.

    This decoder was checked for the same tx_pulse_reserved width bug
    that affects the related Reson 7125 decoder, where
    DecodeReson7100Specific() reads that field from a stale two-byte
    local variable instead of the four-byte one just loaded. This
    decoder does not have that bug: here, TxPulseReserved is correctly
    read as its own fresh two-byte value.

    A few of gsf.h's own field comments understate the true on-disk
    width of certain fields; for example, some comments say "two byte"
    for fields that DecodeResonTSeriesSpecific() and
    EncodeResonTSeriesSpecific() actually read and write as four bytes,
    and say "four byte" for MatchFilterShadingValue, which is actually
    two bytes. This port follows the C code's actual memcpy/htons/htonl
    widths rather than those comments, throughout.

    This function has not been tested against a verified GSF file, since
    no sample data containing a RESON_TSERIES_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_reson_tseries_specific()
    fields['ProtocolVersion'] = protocol_version
    fields['DeviceID'] = device_id
    fields['NumberDevices'] = number_devices
    fields['SystemEnumerator'] = system_enumerator
    fields['MajorSerialNumber'] = major_serial_number
    fields['MinorSerialNumber'] = minor_serial_number
    fields['PingNumber'] = ping_number
    fields['MultiPingSeq'] = multi_ping_seq
    fields['Frequency_Hz'] = frequency_raw / 1.0e3
    fields['SampleRate_Hz'] = sample_rate_raw / 1.0e4
    fields['ReceiverBandwidth_Hz'] = receiver_bandwidth_raw / 1.0e4
    fields['TxPulseWidth_s'] = tx_pulse_width_raw / 1.0e7
    fields['TxPulseTypeID'] = tx_pulse_type_id
    fields['TxPulseEnvelopeID'] = tx_pulse_envlp_id
    fields['TxPulseEnvelopeParam'] = tx_pulse_envlp_param_raw / 1.0e2
    fields['TxPulseMode'] = tx_pulse_mode
    fields['TxPulseReserved'] = tx_pulse_reserved
    fields['MaxPingRate_pps'] = max_ping_rate_raw / 1.0e6
    fields['PingPeriod_s'] = ping_period_raw / 1.0e6
    fields['Range_m'] = range_raw / 1.0e2
    fields['Power_dB'] = power_raw / 1.0e2
    fields['Gain_dB'] = gain_raw / 1.0e2
    fields['ControlFlags'] = control_flags
    fields['ProjectorID'] = projector_id
    fields['ProjectorSteerAnglVert_deg'] = projector_steer_vert_raw / 1.0e3
    fields['ProjectorSteerAnglHoriz_deg'] = projector_steer_horz_raw / 1.0e3
    fields['ProjectorBeamWidthVert_deg'] = projector_bw_vert_raw / 1.0e2
    fields['ProjectorBeamWidthHoriz_deg'] = projector_bw_horz_raw / 1.0e2
    fields['ProjectorBeamFocalPt_m'] = projector_focal_pt_raw / 1.0e2
    fields['ProjectorBeamWeightingWindowType'] = projector_weight_window_type
    fields['ProjectorBeamWeightingWindowParam'] = projector_weight_window_param
    fields['TransmitFlags'] = transmit_flags
    fields['HydrophoneID'] = hydrophone_id
    fields['ReceivingBeamWeightingWindowType'] = rx_weight_window_type
    fields['ReceivingBeamWeightingWindowParam'] = rx_weight_window_param
    fields['ReceiveFlags'] = receive_flags
    fields['ReceiveBeamWidth_deg'] = rx_beam_width_raw / 1.0e2
    fields['RangeFiltMin_m'] = range_filt_min_raw / 1.0e1
    fields['RangeFiltMax_m'] = range_filt_max_raw / 1.0e1
    fields['DepthFiltMin_m'] = depth_filt_min_raw / 1.0e1
    fields['DepthFiltMax_m'] = depth_filt_max_raw / 1.0e1
    fields['Absorption_dBkm'] = absorption_raw / 1.0e3
    fields['SoundVelocity_mps'] = sound_velocity_mps
    fields['SvSource'] = sv_source
    fields['Spreading_dB'] = spreading_raw / 1.0e3
    fields['BeamSpacingMode'] = beam_spacing_mode
    fields['SonarSourceMode'] = sonar_source_mode
    fields['CoverageMode'] = coverage_mode
    fields['CoverageAngle_deg'] = coverage_angle_raw / 1.0e2
    fields['HorizontalReceiverSteeringAngle_deg'] = horiz_rx_steer_raw / 1.0e2
    fields['UncertaintyType'] = uncertainty_type
    fields['TransmitterSteeringAngle_rad'] = tx_steering_angle_raw / 1.0e5
    fields['AppliedRoll_rad'] = applied_roll_raw / 1.0e5
    fields['DetectionAlgorithm'] = detection_algorithm
    fields['DetectionFlags'] = detection_flags
    fields['DeviceDescription'] = device_description
    fields['MatchFilterControl'] = match_filter_control
    fields['MatchFilterStartFreq_Hz'] = match_filter_start_freq_raw / 1.0e2
    fields['MatchFilterEndFreq_Hz'] = match_filter_end_freq_raw / 1.0e2
    fields['MatchFilterWindowType'] = match_filter_window_type
    fields['MatchFilterShadingValue'] = match_filter_shading_raw / 1.0e4
    fields['MatchFilterEffectivePulseWidth_s'] = match_filter_pulse_width_raw / 1.0e11
    return fields, pos - start


def _encode_reson_tseries_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_RESON_TSERIES_SPECIFIC subrecord
    (id 155), including its own four-byte subrecord ID and size word.
    This is the inverse of _decode_reson_tseries_specific(), and is
    ported from the reference gsflib C library's
    EncodeResonTSeriesSpecific() function in gsf_enc.c. It writes
    SoundVelocity_mps out twice, once as a two-byte low-precision copy
    and once as a four-byte high-precision copy, matching the reference
    encoder exactly; see _decode_reson_tseries_specific()'s docstring for
    why the wire format stores the value twice. Every scaled field uses
    this module's standard, sign-correct _gsf_round() convention, which
    is equivalent to the reference encoder's own mix of unconditional and
    sign-branched rounding.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_reson_tseries_specific() returns. A field that
        is missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
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
    header_word = ((155 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[155] = ("ResonTSeries", _decode_reson_tseries_specific, _encode_reson_tseries_specific)


def new_geoswath_plus_specific():
    """
    Return a new dictionary with every GeoSwathPlus sensor-specific
    subrecord field name present, in the same shape
    _decode_geoswath_plus_specific() returns, each pre-set to 0 or 0.0.
    gsf.h defines no "not available" value for these vendor-specific fields,
    so these are the only defaults available.
    _decode_geoswath_plus_specific() builds its result starting from this
    same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (136).

    :return: a dictionary with every GeoSwathPlus sensor-specific field name
        as a key, pre-set to its default.
    """
    return {
        'DataSource': 0, 'Side': 0, 'ModelNumber': 0, 'Frequency_Hz': 0.0,
        'EchosounderType': 0, 'PingNumber': 0, 'NumNavSamples': 0, 'NumAttitudeSamples': 0,
        'NumHeadingSamples': 0, 'NumMiniSVSSamples': 0, 'NumEchosounderSamples': 0,
        'NumRaaSamples': 0, 'MeanSV_mps': 0.0, 'SurfaceVelocity_mps': 0.0, 'ValidBeams': 0,
        'SampleRate_Hz': 0.0, 'PulseLength_us': 0.0, 'PingLength_m': 0, 'TransmitPower': 0,
        'SidescanGainChannel': 0, 'Stabilization': 0, 'GpsQuality': 0,
        'RangeUncertainty_m': 0.0, 'AngleUncertainty_deg': 0.0,
    }


def _decode_geoswath_plus_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_GEOSWATH_PLUS_SPECIFIC subrecord
    (id 136), which holds the per-ping metadata reported by a GeoAcoustics
    GeoSwath Plus interferometric sonar. This is ported from the
    reference gsflib C library's DecodeGeoSwathPlusSpecific() function in
    gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GEOSWATH_PLUS_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_geoswath_plus_specific()
    fields['DataSource'] = data_source
    fields['Side'] = side
    fields['ModelNumber'] = model_number
    fields['Frequency_Hz'] = frequency_raw * 10.0
    fields['EchosounderType'] = echosounder_type
    fields['PingNumber'] = ping_number
    fields['NumNavSamples'] = num_nav_samples
    fields['NumAttitudeSamples'] = num_attitude_samples
    fields['NumHeadingSamples'] = num_heading_samples
    fields['NumMiniSVSSamples'] = num_minisvs_samples
    fields['NumEchosounderSamples'] = num_echosounder_samples
    fields['NumRaaSamples'] = num_raa_samples
    fields['MeanSV_mps'] = mean_sv_raw / 20.0
    fields['SurfaceVelocity_mps'] = surface_velocity_raw / 20.0
    fields['ValidBeams'] = valid_beams
    fields['SampleRate_Hz'] = sample_rate_raw * 10.0
    fields['PulseLength_us'] = float(pulse_length)
    fields['PingLength_m'] = ping_length
    fields['TransmitPower'] = transmit_power
    fields['SidescanGainChannel'] = sidescan_gain_channel
    fields['Stabilization'] = stabilization
    fields['GpsQuality'] = gps_quality
    fields['RangeUncertainty_m'] = range_uncertainty_raw / 1000.0
    fields['AngleUncertainty_deg'] = angle_uncertainty_raw / 100.0
    return fields, pos - start


def _encode_geoswath_plus_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_GEOSWATH_PLUS_SPECIFIC subrecord
    (id 136), including its own four-byte subrecord ID and size word.
    This is the inverse of _decode_geoswath_plus_specific(), and is
    ported from the reference gsflib C library's
    EncodeGeoSwathPlusSpecific() function in gsf_enc.c. Every field here
    is non-negative in practice, so the reference encoder's unconditional
    rounding and this module's sign-correct _gsf_round() convention
    always agree.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_geoswath_plus_specific() returns. A field that
        is missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
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
    header_word = ((136 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[136] = (
    "GeoSwathPlus", _decode_geoswath_plus_specific, _encode_geoswath_plus_specific)


def new_klein5410bss_specific():
    """
    Return a new dictionary with every Klein5410Bss sensor-specific
    subrecord field name present, in the same shape
    _decode_klein5410bss_specific() returns, each pre-set to 0 or 0.0. gsf.h
    defines no "not available" value for these vendor-specific fields, so
    these are the only defaults available. _decode_klein5410bss_specific()
    builds its result starting from this same template, so the two can never
    define a different set of field names. To write one, fill it in and set
    it as a ping record's 'SensorSpecific', with 'SensorSpecificID' set to
    this family's subrecord id (137).

    :return: a dictionary with every Klein5410Bss sensor-specific field name
        as a key, pre-set to its default.
    """
    return {
        'DataSource': 0, 'Side': 0, 'ModelNumber': 0, 'AcousticFrequency_Hz': 0.0,
        'SamplingFrequency_Hz': 0.0, 'PingNumber': 0, 'NumSamples': 0, 'NumRaaSamples': 0,
        'ErrorFlags': 0, 'Range': 0, 'FishDepth_V': 0.0, 'FishAltitude_m': 0.0,
        'SoundSpeed_mps': 0.0, 'TxWaveform': 0, 'Altimeter': 0, 'RawDataConfig': 0,
    }


def _decode_klein5410bss_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_KLEIN_5410_BSS_SPECIFIC subrecord
    (id 137), which holds the per-ping metadata reported by a Klein 5410
    BSS side-scan sonar. This is the ping-level sensor-specific block,
    which is distinct from _decode_klein5410bss_imagery_specific() (the
    smaller preamble found inside the per-beam backscatter subrecord, id
    21). This is ported from the reference gsflib C library's
    DecodeKlein5410BssSpecific() function in gsf_dec.c.

    The reference gsflib C library has an asymmetry between how it reads
    and how it writes the FishDepth_V, FishAltitude_m, and SoundSpeed_mps
    fields. Its decoder reads each field's raw four-byte value as
    unsigned, with no sign reinterpretation, but its encoder allows a
    caller to write a negative value into those same three fields using a
    sign-aware rounding branch. This means a file written by the real
    gsflib with a negative value in one of those fields would not decode
    back correctly, even when read by the real gsflib itself. This
    function matches gsflib's decode side (reading the raw value as
    unsigned); see _encode_klein5410bss_specific()'s docstring for the
    resulting consequence on the encode side.

    This function has not been tested against a verified GSF file, since
    no sample data containing a KLEIN_5410_BSS_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_klein5410bss_specific()
    fields['DataSource'] = data_source
    fields['Side'] = side
    fields['ModelNumber'] = model_number
    fields['AcousticFrequency_Hz'] = acoustic_frequency_raw / 1000.0
    fields['SamplingFrequency_Hz'] = sampling_frequency_raw / 1000.0
    fields['PingNumber'] = ping_number
    fields['NumSamples'] = num_samples
    fields['NumRaaSamples'] = num_raa_samples
    fields['ErrorFlags'] = error_flags
    fields['Range'] = rng
    fields['FishDepth_V'] = fish_depth_raw / 1000.0
    fields['FishAltitude_m'] = fish_altitude_raw / 1000.0
    fields['SoundSpeed_mps'] = sound_speed_raw / 1000.0
    fields['TxWaveform'] = tx_waveform
    fields['Altimeter'] = altimeter
    fields['RawDataConfig'] = raw_data_config
    return fields, pos - start


def _encode_klein5410bss_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_KLEIN_5410_BSS_SPECIFIC subrecord
    (id 137), including its own four-byte subrecord ID and size word.
    This is the inverse of _decode_klein5410bss_specific(), and is ported
    from the reference gsflib C library's EncodeKlein5410BssSpecific()
    function in gsf_enc.c.

    Unlike the reference gsflib encoder, this function packs
    FishDepth_V, FishAltitude_m, and SoundSpeed_mps as unsigned
    thirty-two-bit fields, matching how _decode_klein5410bss_specific()
    reads them back rather than reproducing the reference encoder's own
    behavior of allowing a negative value there. As a result, passing a
    negative value for one of these three fields raises struct.error here,
    instead of silently producing bytes that no decoder, including the
    real gsflib, could correctly read back. See
    _decode_klein5410bss_specific()'s docstring for the underlying
    asymmetry between gsflib's own decoder and encoder that this avoids.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_klein5410bss_specific() returns. A field that
        is missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.

    :raises struct.error: FishDepth_V, FishAltitude_m, or SoundSpeed_mps
        is negative.
    """
    g = record.get
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
    header_word = ((137 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[137] = (
    "Klein5410Bss", _decode_klein5410bss_specific, _encode_klein5410bss_specific)


def new_sass_specific():
    """
    Return a new dictionary with every SASS sensor-specific subrecord field
    name present, in the same shape _decode_sass_specific() returns, each
    pre-set to 0 or 0.0. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_sass_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord ids (108, 112).

    :return: a dictionary with every SASS sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'LeftmostBeam': 0, 'RightmostBeam': 0, 'TotalBeams': 0, 'NavMode': 0, 'PingNumber': 0,
        'MissionNumber': 0,
    }


def _decode_sass_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SASS_SPECIFIC (id 108) or
    GSF_SWATH_BATHY_SUBRECORD_TYPEIII_SEABEAM_SPECIFIC (id 112)
    subrecord. Both ids share exactly the same wire format, and gsf.h
    defines a single struct (t_gsfTypeIIISpecific) and a single decoder
    for both, so this function does too. This is ported from the
    reference gsflib C library's DecodeSASSSpecific() function in
    gsf_dec.c, which was confirmed to be byte-for-byte identical to
    gsflib's separate DecodeTypeIIISeaBeamSpecific() function. Both
    record types are marked obsolete in gsf.h, having been replaced by
    CMP_SASS_SPECIFIC (see _decode_cmp_sass_specific()), but the
    reference gsflib library still ships a full decode and encode pair
    for them, so this library does too.

    This function has not been tested against a verified GSF file, since
    no sample data containing a SASS_SPECIFIC or TYPEIII_SEABEAM_SPECIFIC
    subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (leftmost_beam,) = struct.unpack_from('>H', payload, pos); pos += 2
    (rightmost_beam,) = struct.unpack_from('>H', payload, pos); pos += 2
    (total_beams,) = struct.unpack_from('>H', payload, pos); pos += 2
    (nav_mode,) = struct.unpack_from('>H', payload, pos); pos += 2
    (ping_number,) = struct.unpack_from('>H', payload, pos); pos += 2
    (mission_number,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = new_sass_specific()
    fields['LeftmostBeam'] = leftmost_beam
    fields['RightmostBeam'] = rightmost_beam
    fields['TotalBeams'] = total_beams
    fields['NavMode'] = nav_mode
    fields['PingNumber'] = ping_number
    fields['MissionNumber'] = mission_number
    return fields, pos - start


def _encode_sass_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SASS_SPECIFIC (id 108) or
    GSF_SWATH_BATHY_SUBRECORD_TYPEIII_SEABEAM_SPECIFIC (id 112)
    subrecord, including its own four-byte subrecord ID and size word.
    This is the inverse of _decode_sass_specific(), and is ported from
    the reference gsflib C library's EncodeSASSSpecific() function in
    gsf_enc.c, which was confirmed to be byte-for-byte identical to
    gsflib's separate EncodeTypeIIISeaBeamSpecific() function.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_sass_specific() returns, plus an optional
        'SubrecordID' key choosing which of the two subrecord ids to
        write (108 for SASS or 112 for TypeIII SeaBeam); if that key is
        absent, 108 is written. A field that is missing from the
        dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 108)
    g = record.get
    body = struct.pack(
        '>6H',
        int(g('LeftmostBeam', 0)), int(g('RightmostBeam', 0)), int(g('TotalBeams', 0)),
        int(g('NavMode', 0)), int(g('PingNumber', 0)), int(g('MissionNumber', 0)))
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[108] = ("SASS", _decode_sass_specific, _encode_sass_specific)
_PING_SENSOR_SPECIFIC_CODECS[112] = ("TypeIIISeaBeam", _decode_sass_specific, _encode_sass_specific)


def new_delta_t_specific():
    """
    Return a new dictionary with every DeltaT sensor-specific subrecord
    field name present, in the same shape _decode_delta_t_specific()
    returns, each pre-set to 0 or 0.0, an empty string, or None (for a
    time). gsf.h defines no "not available" value for these vendor-specific
    fields, so these are the only defaults available.
    _decode_delta_t_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord id (150).

    :return: a dictionary with every DeltaT sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'DecodeFileType': '', 'Version': 0, 'PingByteSize': 0, 'InterrogationTime': None,
        'SamplesPerBeam': 0, 'SectorSize_deg': 0.0, 'StartAngle_deg': 0.0,
        'AngleIncrement_deg': 0.0, 'AcousticRange_m': 0, 'AcousticFrequency_kHz': 0,
        'SoundVelocity_mps': 0.0, 'RangeResolution_cm': 0.0, 'ProfileTiltAngle_deg': 0.0,
        'RepetitionRate_ms': 0.0, 'PingNumber': 0, 'IntensityFlag': 0, 'PingLatency_s': 0.0,
        'DataLatency_s': 0.0, 'SampleRateFlag': 0, 'OptionFlags': 0, 'NumPingsAvg': 0,
        'CenterPingTimeOffset_s': 0.0, 'UserDefinedByte': 0, 'Altitude_m': 0.0,
        'ExternalSensorFlags': 0, 'PulseLength_s': 0.0, 'ForeAftBeamwidth_deg': 0.0,
        'AthwartshipsBeamwidth_deg': 0.0,
    }


def _decode_delta_t_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_DELTA_T_SPECIFIC subrecord (id
    150), which holds the per-ping metadata reported by an Imagenex
    Delta T multibeam sonar. This is ported from the reference gsflib C
    library's DecodeDeltaTSpecific() function in gsf_dec.c.

    The field scaling here follows the C source exactly, including its
    asymmetries between similar-looking fields. For example, the start
    angle is stored on the wire as (angle + 180) * 100, but the profile
    tilt angle is stored as plain (angle + 180) with no multiplication by
    100, and the sector size, acoustic range, acoustic frequency, range
    resolution, and repetition rate carry no scale factor at all even
    though they are represented as floating-point values.

    This function has not been tested against a verified GSF file, since
    no sample data containing a DELTA_T_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_delta_t_specific()
    fields['DecodeFileType'] = decode_file_type
    fields['Version'] = version
    fields['PingByteSize'] = ping_byte_size
    fields['InterrogationTime'] = _gsf_timestamp(sec, nsec)
    fields['SamplesPerBeam'] = samples_per_beam
    fields['SectorSize_deg'] = float(sector_size_raw)
    fields['StartAngle_deg'] = start_angle_raw / 100.0 - 180.0
    fields['AngleIncrement_deg'] = angle_increment_raw / 100.0
    fields['AcousticRange_m'] = acoustic_range
    fields['AcousticFrequency_kHz'] = acoustic_frequency
    fields['SoundVelocity_mps'] = sound_velocity_raw / 10.0
    fields['RangeResolution_cm'] = float(range_resolution_raw)
    fields['ProfileTiltAngle_deg'] = profile_tilt_angle_raw - 180.0
    fields['RepetitionRate_ms'] = float(repetition_rate_raw)
    fields['PingNumber'] = ping_number
    fields['IntensityFlag'] = intensity_flag
    fields['PingLatency_s'] = ping_latency_raw / 10000.0
    fields['DataLatency_s'] = data_latency_raw / 10000.0
    fields['SampleRateFlag'] = sample_rate_flag
    fields['OptionFlags'] = option_flags
    fields['NumPingsAvg'] = num_pings_avg
    fields['CenterPingTimeOffset_s'] = center_ping_time_offset_raw / 10000.0
    fields['UserDefinedByte'] = user_defined_byte
    fields['Altitude_m'] = altitude_raw / 100.0
    fields['ExternalSensorFlags'] = external_sensor_flags
    fields['PulseLength_s'] = pulse_length_raw / 1.0e6
    fields['ForeAftBeamwidth_deg'] = fore_aft_beamwidth_raw / 10.0
    fields['AthwartshipsBeamwidth_deg'] = athwartships_beamwidth_raw / 10.0
    return fields, pos - start


def _encode_delta_t_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_DELTA_T_SPECIFIC subrecord (id
    150), including its own four-byte subrecord ID and size word. This is
    the inverse of _decode_delta_t_specific(), and is ported from the
    reference gsflib C library's EncodeDeltaTSpecific() function in
    gsf_enc.c. See _decode_delta_t_specific()'s docstring for the
    per-field scaling asymmetries this preserves.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_delta_t_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    decode_file_type = str(g('DecodeFileType', '')).encode('ascii')[:4].ljust(4, b'\x00')
    body = decode_file_type
    body += struct.pack('>B', int(g('Version', 0)) & 0xFF)
    body += struct.pack('>H', int(g('PingByteSize', 0)))
    interrogation_time = g('InterrogationTime')
    sec, nsec = _gsf_epoch(interrogation_time) if interrogation_time is not None else (0, 0)
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
    header_word = ((150 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[150] = ("DeltaT", _decode_delta_t_specific, _encode_delta_t_specific)


def new_r2sonic_specific():
    """
    Return a new dictionary with every R2Sonic sensor-specific subrecord
    field name present, in the same shape _decode_r2sonic_specific()
    returns, each pre-set to 0 or 0.0, an empty string, or None (for a
    time). gsf.h defines no "not available" value for these vendor-specific
    fields, so these are the only defaults available.
    _decode_r2sonic_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord ids (151, 152,
    153).

    :return: a dictionary with every R2Sonic sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'ModelNumber': '', 'SerialNumber': '', 'PingTime': None, 'PingNumber': 0,
        'PingPeriod_s': 0.0, 'SoundSpeed_mps': 0.0, 'Frequency_Hz': 0.0, 'TxPower_dB': 0.0,
        'TxPulseWidth_s': 0.0, 'TxBeamwidthVert_deg': 0.0, 'TxBeamwidthHoriz_deg': 0.0,
        'TxSteeringVert_deg': 0.0, 'TxSteeringHoriz_deg': 0.0, 'TxMiscInfo': 0,
        'RxBandwidth_Hz': 0.0, 'RxSampleRate_Hz': 0.0, 'RxRange_m': 0.0, 'RxGain_dB': 0.0,
        'RxSpreading': 0.0, 'RxAbsorption_dBkm': 0.0, 'RxMountTilt_deg': 0.0, 'RxMiscInfo': 0,
        'Reserved': 0, 'NumBeams': 0, 'A0MoreInfo': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'A2MoreInfo': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'G0DepthGateMin_s': 0.0,
        'G0DepthGateMax_s': 0.0, 'G0DepthGateSlope_deg': 0.0,
    }


def _decode_r2sonic_specific(payload, pos):
    """
    Decode an R2Sonic 2020/2022/2024 sensor-specific subrecord (ids
    153, 151, and 152, respectively), which all three models share as a
    single wire format and struct definition. This is ported from the
    reference gsflib C library's DecodeR2SonicSpecific() function in
    gsf_dec.c. This function is distinct from
    _decode_r2sonic_imagery_specific(), which decodes a different,
    smaller struct nested inside the intensity-series subrecord (id 21);
    this function instead decodes the ping-level "_SPECIFIC" subrecord.

    This function has not been tested against a verified GSF file, since
    no sample data containing an R2SONIC_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
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

    fields = new_r2sonic_specific()
    fields['ModelNumber'] = model_number
    fields['SerialNumber'] = serial_number
    fields['PingTime'] = _gsf_timestamp(sec, nsec)
    fields['PingNumber'] = ping_number
    fields['PingPeriod_s'] = ping_period_raw / 1.0e6
    fields['SoundSpeed_mps'] = sound_speed_raw / 1.0e2
    fields['Frequency_Hz'] = frequency_raw / 1.0e3
    fields['TxPower_dB'] = tx_power_raw / 1.0e2
    fields['TxPulseWidth_s'] = tx_pulse_width_raw / 1.0e7
    fields['TxBeamwidthVert_deg'] = tx_beamwidth_vert_raw / 1.0e6
    fields['TxBeamwidthHoriz_deg'] = tx_beamwidth_horiz_raw / 1.0e6
    fields['TxSteeringVert_deg'] = tx_steering_vert_raw / 1.0e6
    fields['TxSteeringHoriz_deg'] = tx_steering_horiz_raw / 1.0e6
    fields['TxMiscInfo'] = tx_misc_info
    fields['RxBandwidth_Hz'] = rx_bandwidth_raw / 1.0e4
    fields['RxSampleRate_Hz'] = rx_sample_rate_raw / 1.0e3
    fields['RxRange_m'] = rx_range_raw / 1.0e5
    fields['RxGain_dB'] = rx_gain_raw / 1.0e2
    fields['RxSpreading'] = rx_spreading_raw / 1.0e3
    fields['RxAbsorption_dBkm'] = rx_absorption_raw / 1.0e3
    fields['RxMountTilt_deg'] = rx_mount_tilt_raw / 1.0e6
    fields['RxMiscInfo'] = rx_misc_info
    fields['Reserved'] = reserved
    fields['NumBeams'] = num_beams
    fields['A0MoreInfo'] = a0_more_info
    fields['A2MoreInfo'] = a2_more_info
    fields['G0DepthGateMin_s'] = g0_depth_gate_min_raw / 1.0e6
    fields['G0DepthGateMax_s'] = g0_depth_gate_max_raw / 1.0e6
    fields['G0DepthGateSlope_deg'] = g0_depth_gate_slope_raw / 1.0e6
    return fields, pos - start


def _encode_r2sonic_specific(record):
    """
    Encode an R2Sonic 2020/2022/2024 sensor-specific subrecord, including
    its own four-byte subrecord ID and size word. This is the inverse of
    _decode_r2sonic_specific(), and is ported from the reference gsflib C
    library's EncodeR2SonicSpecific() function in gsf_enc.c. Which of the
    three subrecord ids (151 for the 2022, 152 for the 2024, or 153 for
    the 2020) is stamped into the header is chosen by the record's
    optional 'SubrecordID' key; if that key is absent, 151 is written.

    The reference C encoder rounds most fields by adding an unconditional
    0.501 before truncating, which is only correct for fields that are
    physically non-negative, and rounds a handful of signed fields (the
    vertical and horizontal transmit steering angles, the receive mount
    tilt, the A0/A2 "more info" arrays, and the G0 depth-gate slope)
    with a sign-aware branch that adds 0.501 or subtracts 0.501 depending
    on the sign of the value. This Python port instead uses the standard
    sign-correct _gsf_round() convention for every field. That convention
    is numerically equivalent to the reference encoder's unconditional
    +0.501 for the non-negative fields, and matches the reference
    encoder's sign-aware branch exactly for the signed fields, so the
    encoded output is the same either way.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_r2sonic_specific() returns, plus an optional
        'SubrecordID' key choosing which of the three subrecord ids to
        write. A field that is missing from the dictionary is written as
        zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 151)
    g = record.get

    def model_bytes(key):
        """Encode a model/serial-number string field as a 12-byte, null-padded ASCII field."""
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
    Decode one t_gsfEMRunTime block: a set of Kongsberg EM-series
    run-time sonar parameters. This is not a standalone subrecord with
    its own header, but an inline sub-block embedded once inside the
    EM4_SPECIFIC subrecord (see _decode_em4_specific()) and, in exactly
    the same wire format, inside the EM3 "_RAW"-variant subrecords. The
    caller merges the fields this function returns into its own larger
    record under a 'RunTime.' key prefix. This is ported from the
    run-time-parameter field reads that the reference gsflib C library
    inlines identically in both its DecodeEM4Specific() and
    DecodeEM3RawSpecific() functions in gsf_dec.c; rather than duplicate
    that logic, this Python port factors it out into this shared helper
    function.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this sub-block's
        data begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload, which is always 63.
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
    Encode one t_gsfEMRunTime block, returning only the sub-block's raw
    bytes with no subrecord header of its own, since it is an inline
    sub-block embedded inside the EM4_SPECIFIC subrecord and the EM3
    "_RAW"-variant subrecords rather than a standalone subrecord. This
    is the inverse of _decode_em_run_time(), and is ported from the
    run-time-parameter field writes that the reference gsflib C library
    inlines identically in both its EncodeEM4Specific() and
    EncodeEM3RawSpecific() functions in gsf_enc.c.

    The reference C encoder writes the minimum depth, maximum depth,
    transmit pulse length, transmit power re maximum, receive fixed
    gain, and TVG cross-over angle fields using a direct truncating cast
    with no rounding offset added first, unlike every other scaled field
    in this block, which the reference encoder does round with a
    +/-0.501 offset. This Python port instead uses the standard
    _gsf_round() convention for all of these fields, consistent with
    every other encoder function in this module.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_em_run_time() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded sub-block, as bytes, with no subrecord header.
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
    Decode one t_gsfEMPUStatus block: Kongsberg EM-series processing-unit
    status fields. Like _decode_em_run_time(), this is not a standalone
    subrecord with its own header, but an inline sub-block embedded once
    inside the EM4_SPECIFIC subrecord and, in exactly the same wire
    format, inside the EM3 "_RAW"-variant subrecords. The caller merges
    the fields this function returns into its own larger record under a
    'PuStatus.' key prefix. This is ported from the field reads that the
    reference gsflib C library duplicates identically in both its
    DecodeEM4Specific() and DecodeEM3RawSpecific() functions in
    gsf_dec.c.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this sub-block's
        data begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload, which is always 23.
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
    Encode one t_gsfEMPUStatus block, returning only the sub-block's raw
    bytes with no subrecord header of its own, since it is an inline
    sub-block embedded inside the EM4_SPECIFIC subrecord and the EM3
    "_RAW"-variant subrecords rather than a standalone subrecord. This
    is the inverse of _decode_em_pu_status(), and is ported from the
    field writes that the reference gsflib C library duplicates
    identically in both its EncodeEM4Specific() and
    EncodeEM3RawSpecific() functions in gsf_enc.c.

    Like _encode_em_run_time(), the processing-unit CPU load field is
    written by the reference C encoder using a direct truncating cast
    with no rounding offset added first. This Python port instead uses
    _gsf_round() for that field, for consistency with the rest of this
    module.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_em_pu_status() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded sub-block, as bytes, with no subrecord header.
    """
    g = fields.get
    out = struct.pack('>B', _gsf_round(g('PuCpuLoad_pct', 0.0)) & 0xFF)
    out += struct.pack('>H', int(g('SensorStatus', 0)))
    out += struct.pack('>b', int(g('AchievedPortCoverage_deg', 0)))
    out += struct.pack('>b', int(g('AchievedStbdCoverage_deg', 0)))
    out += struct.pack('>h', _gsf_round(g('YawStabilization_deg', 0.0) * 100.0))
    out += b'\x00' * 16
    return out


def _rows_to_table(rows):
    """
    Convert a list of row dictionaries, all with the same keys, into a
    table: a dictionary mapping each key to a numpy array holding that
    column's value from every row. This is the table shape every decoder
    in this file returns for per-element data (per-beam arrays, transmit
    sectors, run-time blocks, and so on), for decoders whose rows are few
    and variably laid out enough that they are still decoded one row at a
    time.

    :param rows: a non-empty list of dictionaries sharing the same keys.

    :return: a dictionary of column name -> numpy array, one element per
        row.
    """
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


def _present_columns(table):
    """
    Return the columns of a table that hold data, leaving out any column
    whose value is None. A decoded swath bathymetry ping's 'Beams' table
    has a column for every beam array, set to None for each subrecord the
    ping did not carry, so code that reads, writes, or prints that table
    uses this to skip the absent arrays.

    :param table: the table (a dictionary of column name -> array-like,
        or anything with the same column access, such as a
        pandas.DataFrame), or None.

    :return: a new dictionary of column name -> array-like, holding only
        the columns that are not None; empty if table is None.
    """
    if table is None:
        return {}
    return {key: table[key] for key in table.keys() if table[key] is not None}


def _table_length(table):
    """
    Return the number of rows in a table: a dictionary of column name ->
    equal-length array-like, as the decoders in this file return, or
    anything else with the same column access, such as a
    pandas.DataFrame. Columns that are None, meaning absent, are ignored.

    :param table: the table, or None.

    :return: the number of rows; 0 if table is None or has no columns
        other than None ones.
    """
    for column in _present_columns(table).values():
        return len(column)
    return 0


def _table_rows(table, limit=None):
    """
    Yield each row of a table as a dictionary of column name -> value,
    for encoders that write a table's rows one at a time. The table may
    be a dictionary of column name -> equal-length array-like, as the
    decoders in this file return, or anything else with the same column
    access, such as a pandas.DataFrame, which is read column by column
    without any conversion.

    :param table: the table, or None (which yields no rows).
    :param limit: if given, yield at most this many rows.

    :return: a generator of row dictionaries.
    """
    if table is None:
        return
    columns = {key: np.asarray(table[key]) for key in table.keys()}
    n = _table_length(columns)
    if limit is not None:
        n = min(n, limit)
    for i in range(n):
        yield {key: values[i] for key, values in columns.items()}


def new_em4_specific():
    """
    Return a new dictionary with every EM4 sensor-specific subrecord field
    name present, in the same shape _decode_em4_specific() returns, each
    pre-set to 0 or 0.0, or None (for a time). gsf.h defines no "not
    available" value for these vendor-specific fields, so these are the only
    defaults available. _decode_em4_specific() builds its result starting
    from this same template, so the two can never define a different set of
    field names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    ids (133, 134, 135, 149, 157).

    :return: a dictionary with every EM4 sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'ModelNumber': 0, 'PingCounter': 0, 'SerialNumber': 0, 'SurfaceVelocity_mps': 0.0,
        'TransducerDepth_m': 0.0, 'ValidDetections': 0, 'SamplingFrequency_Hz': 0.0,
        'DopplerCorrScale': 0, 'VehicleDepth_m': 0.0, 'RunTime.ModelNumber': 0,
        'RunTime.PingTime': None, 'RunTime.PingCounter': 0, 'RunTime.SerialNumber': 0,
        'RunTime.OperatorStationStatus': 0, 'RunTime.ProcessingUnitStatus': 0,
        'RunTime.BspStatus': 0, 'RunTime.HeadTransceiverStatus': 0, 'RunTime.Mode': 0,
        'RunTime.FilterID': 0, 'RunTime.MinDepth_m': 0.0, 'RunTime.MaxDepth_m': 0.0,
        'RunTime.Absorption_dBkm': 0.0, 'RunTime.TxPulseLength_us': 0.0,
        'RunTime.TxBeamWidth_deg': 0.0, 'RunTime.TxPowerReMax_dB': 0.0,
        'RunTime.RxBeamWidth_deg': 0.0, 'RunTime.RxBandwidth_Hz': 0.0,
        'RunTime.RxFixedGain_dB': 0.0, 'RunTime.TvgCrossOverAngle_deg': 0.0,
        'RunTime.SsvSource': 0, 'RunTime.MaxPortSwathWidth_m': 0, 'RunTime.BeamSpacing': 0,
        'RunTime.MaxPortCoverage_deg': 0, 'RunTime.Stabilization': 0,
        'RunTime.MaxStbdCoverage_deg': 0, 'RunTime.MaxStbdSwathWidth_m': 0,
        'RunTime.TxAlongTilt_deg': 0.0, 'RunTime.FilterID2': 0, 'PuStatus.PuCpuLoad_pct': 0.0,
        'PuStatus.SensorStatus': 0, 'PuStatus.AchievedPortCoverage_deg': 0,
        'PuStatus.AchievedStbdCoverage_deg': 0, 'PuStatus.YawStabilization_deg': 0.0,
    }


def new_em4_tx_sector():
    """
    Return a new dictionary with every per-row field name of the EM4
    sensor-specific subrecord's 'TxSectors' table present, each pre-set to its
    default: one row of that table. _decode_em4_specific() builds each row
    starting from this same template, so the two can never define a
    different set of field names. A caller building the table can fill in
    one of these per row and assign pd.DataFrame([row, ...]) (or the
    equivalent dictionary of column name -> array) to
    record['SensorSpecific']['TxSectors'].

    :return: a dictionary with every 'TxSectors' column name as a key.
    """
    return {
        'TiltAngle_deg': 0.0, 'FocusRange_m': 0.0, 'SignalLength_sec': 0.0,
        'TransmitDelay_sec': 0.0, 'CenterFrequency_Hz': 0.0, 'MeanAbsorption_dBkm': 0.0,
        'WaveformID': 0, 'SectorNumber': 0, 'SignalBandwidth_Hz': 0.0,
    }


def _decode_em4_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM4_SPECIFIC subrecord (one of ids
    133, 134, 135, 149, or 157, covering the Kongsberg EM710, EM302,
    EM122, EM2040, and ME70BO sonars respectively, which all share this
    same wire layout). This holds the per-ping sensor metadata reported
    by one of those sonars, together with its per-transmit-sector table
    and the inline run-time-parameters and PU-status sub-blocks. This is
    ported from the reference gsflib C library's DecodeEM4Specific()
    function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM4_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        holding the 9 top-level scalar fields, every field decoded by
        _decode_em_run_time() merged in with a 'RunTime.' key prefix,
        every field decoded by _decode_em_pu_status() merged in with a
        'PuStatus.' key prefix, and an optional 'TxSectors' key holding
        a table (a dictionary of column name -> numpy array, one element
        per transmit sector), present only when the ping reported at
        least one transmit sector.
        bytes_consumed is the number of bytes read from payload.
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
        row = new_em4_tx_sector()
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

    record = new_em4_specific()
    record['ModelNumber'] = model_number
    record['PingCounter'] = ping_counter
    record['SerialNumber'] = serial_number
    record['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    record['TransducerDepth_m'] = transducer_depth_raw / 20000.0
    record['ValidDetections'] = valid_detections
    record['SamplingFrequency_Hz'] = sampling_frequency
    record['DopplerCorrScale'] = doppler_corr_scale
    record['VehicleDepth_m'] = vehicle_depth_raw / 1000.0
    record.update({'RunTime.' + k: v for k, v in run_time_fields.items()})
    record.update({'PuStatus.' + k: v for k, v in pu_status_fields.items()})

    if sector_rows:
        record['TxSectors'] = _rows_to_table(sector_rows)
    return record, pos - start


def _encode_em4_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM4_SPECIFIC subrecord (one of ids
    133, 134, 135, 149, or 157, chosen via record['SubrecordID']),
    including its own 4-byte subrecord id+size word. This is the exact
    inverse of _decode_em4_specific(), and is ported from the reference
    gsflib C library's EncodeEM4Specific() function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_em4_specific() returns, including its
        'RunTime.'-prefixed and 'PuStatus.'-prefixed fields (encoded by
        _encode_em_run_time() and _encode_em_pu_status() respectively).
        An optional 'TxSectors' key holds a table (a dictionary of column
        name -> array-like, or a pandas.DataFrame) with one row per
        transmit sector; the transmit-sector count written to the wire
        is taken from the number of rows in this table, so the exact
        table _decode_em4_specific() returns can be passed straight back
        in with no conversion required. An optional
        'SubrecordID' key selects which of the five registered ids to
        stamp, defaulting to 133 (EM710) if absent. A field missing from
        record is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 133)
    g = record.get
    sectors = record.get('TxSectors')

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

    body += struct.pack('>H', _table_length(sectors))
    for row in _table_rows(sectors):
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

    run_time_fields = {k[len('RunTime.'):]: v for k, v in record.items() if k.startswith('RunTime.')}
    pu_status_fields = {k[len('PuStatus.'):]: v for k, v in record.items() if k.startswith('PuStatus.')}
    body += _encode_em_run_time(run_time_fields)
    body += _encode_em_pu_status(pu_status_fields)

    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_PING_SENSOR_SPECIFIC_CODECS[133] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[134] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[135] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[149] = ("EM4", _decode_em4_specific, _encode_em4_specific)
_PING_SENSOR_SPECIFIC_CODECS[157] = ("EM4", _decode_em4_specific, _encode_em4_specific)


def new_em3raw_specific():
    """
    Return a new dictionary with every EM3Raw sensor-specific subrecord
    field name present, in the same shape _decode_em3raw_specific() returns,
    each pre-set to 0 or 0.0, or None (for a time). gsf.h defines no "not
    available" value for these vendor-specific fields, so these are the only
    defaults available. _decode_em3raw_specific() builds its result starting
    from this same template, so the two can never define a different set of
    field names. To write one, fill it in and set it as a ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    ids (140, 141, 142, 143, 144, 145, 146, 147, 148).

    :return: a dictionary with every EM3Raw sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'ModelNumber': 0, 'PingCounter': 0, 'SerialNumber': 0, 'SurfaceVelocity_mps': 0.0,
        'TransducerDepth_m': 0.0, 'ValidDetections': 0, 'SamplingFrequency_Hz': 0.0,
        'VehicleDepth_m': 0.0, 'DepthDifference_m': 0.0, 'OffsetMultiplier': 0,
        'RunTime.ModelNumber': 0, 'RunTime.PingTime': None, 'RunTime.PingCounter': 0,
        'RunTime.SerialNumber': 0, 'RunTime.OperatorStationStatus': 0,
        'RunTime.ProcessingUnitStatus': 0, 'RunTime.BspStatus': 0,
        'RunTime.HeadTransceiverStatus': 0, 'RunTime.Mode': 0, 'RunTime.FilterID': 0,
        'RunTime.MinDepth_m': 0.0, 'RunTime.MaxDepth_m': 0.0, 'RunTime.Absorption_dBkm': 0.0,
        'RunTime.TxPulseLength_us': 0.0, 'RunTime.TxBeamWidth_deg': 0.0,
        'RunTime.TxPowerReMax_dB': 0.0, 'RunTime.RxBeamWidth_deg': 0.0,
        'RunTime.RxBandwidth_Hz': 0.0, 'RunTime.RxFixedGain_dB': 0.0,
        'RunTime.TvgCrossOverAngle_deg': 0.0, 'RunTime.SsvSource': 0,
        'RunTime.MaxPortSwathWidth_m': 0, 'RunTime.BeamSpacing': 0,
        'RunTime.MaxPortCoverage_deg': 0, 'RunTime.Stabilization': 0,
        'RunTime.MaxStbdCoverage_deg': 0, 'RunTime.MaxStbdSwathWidth_m': 0,
        'RunTime.TxAlongTilt_deg': 0.0, 'RunTime.FilterID2': 0, 'PuStatus.PuCpuLoad_pct': 0.0,
        'PuStatus.SensorStatus': 0, 'PuStatus.AchievedPortCoverage_deg': 0,
        'PuStatus.AchievedStbdCoverage_deg': 0, 'PuStatus.YawStabilization_deg': 0.0,
    }


def new_em3raw_tx_sector():
    """
    Return a new dictionary with every per-row field name of the EM3Raw
    sensor-specific subrecord's 'TxSectors' table present, each pre-set to its
    default: one row of that table. _decode_em3raw_specific() builds each row
    starting from this same template, so the two can never define a
    different set of field names. A caller building the table can fill in
    one of these per row and assign pd.DataFrame([row, ...]) (or the
    equivalent dictionary of column name -> array) to
    record['SensorSpecific']['TxSectors'].

    :return: a dictionary with every 'TxSectors' column name as a key.
    """
    return {
        'TiltAngle_deg': 0.0, 'FocusRange_m': 0.0, 'SignalLength_sec': 0.0,
        'TransmitDelay_sec': 0.0, 'CenterFrequency_Hz': 0.0, 'WaveformID': 0,
        'SectorNumber': 0, 'SignalBandwidth_Hz': 0.0,
    }


def _decode_em3raw_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_RAW_SPECIFIC subrecord (one
    of ids 140 through 148, covering the Kongsberg EM300, EM1002, EM2000,
    EM3000, EM120, EM3002, EM3000D, EM3002D, and EM121A_SIS "raw"
    variants, which report raw range and beam-angle data and all share
    this same wire layout). This holds the per-ping sensor metadata
    reported by one of those sonars, together with its per-transmit-sector
    table and the inline run-time-parameters and PU-status sub-blocks --
    reusing _decode_em_run_time() and _decode_em_pu_status(), the same
    helpers EM4_SPECIFIC uses, since gsf.h defines both subrecords'
    run-time and PU-status fields using the same t_gsfEMRunTime and
    t_gsfEMPUStatus struct types. This is ported from the reference
    gsflib C library's DecodeEM3RawSpecific() function in gsf_dec.c.

    Despite the superficially similar shape, the field order and scaling
    here differ from EM4_SPECIFIC in several places -- verified directly
    against DecodeEM3RawSpecific() rather than assumed from EM4: there is
    no doppler_corr_scale field; vehicle depth precedes a depth-difference
    field that EM4 does not have at all; the offset multiplier is a
    signed byte; the per-sector table has no mean-absorption field (8
    fields per row, not EM4's 9); and the maximum sector count is 20
    (GSF_MAX_EM3_SECTORS), not EM4's 9 (GSF_MAX_EM4_SECTORS).

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM3xxx_RAW_SPECIFIC subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        holding the 10 top-level scalar fields, every field decoded by
        _decode_em_run_time() merged in with a 'RunTime.' key prefix,
        every field decoded by _decode_em_pu_status() merged in with a
        'PuStatus.' key prefix, and an optional 'TxSectors' key holding
        a table (a dictionary of column name -> numpy array, one element
        per transmit sector), present only when the ping reported at
        least one transmit sector.
        bytes_consumed is the number of bytes read from payload.
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
        row = new_em3raw_tx_sector()
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

    record = new_em3raw_specific()
    record['ModelNumber'] = model_number
    record['PingCounter'] = ping_counter
    record['SerialNumber'] = serial_number
    record['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    record['TransducerDepth_m'] = transducer_depth_raw / 20000.0
    record['ValidDetections'] = valid_detections
    record['SamplingFrequency_Hz'] = sampling_frequency
    record['VehicleDepth_m'] = vehicle_depth_raw / 1000.0
    record['DepthDifference_m'] = depth_difference_raw / 100.0
    record['OffsetMultiplier'] = offset_multiplier
    record.update({'RunTime.' + k: v for k, v in run_time_fields.items()})
    record.update({'PuStatus.' + k: v for k, v in pu_status_fields.items()})

    if sector_rows:
        record['TxSectors'] = _rows_to_table(sector_rows)
    return record, pos - start


def _encode_em3raw_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_RAW_SPECIFIC subrecord (one
    of ids 140 through 148, chosen via record['SubrecordID']), including
    its own 4-byte subrecord id+size word. This is the exact inverse of
    _decode_em3raw_specific(), and is ported from the reference gsflib C
    library's EncodeEM3RawSpecific() function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_em3raw_specific() returns, including its
        'RunTime.'-prefixed and 'PuStatus.'-prefixed fields (encoded by
        _encode_em_run_time() and _encode_em_pu_status() respectively).
        An optional 'TxSectors' key holds a table (a dictionary of column
        name -> array-like, or a pandas.DataFrame) with one row per
        transmit sector; the transmit-sector count written to the wire
        is taken from the number of rows in this table, so the exact
        table _decode_em3raw_specific() returns can be passed straight back
        in with no conversion required. An optional
        'SubrecordID' key selects which of the nine registered ids to
        stamp, defaulting to 140 (EM300_RAW) if absent. A field missing
        from record is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 140)
    g = record.get
    sectors = record.get('TxSectors')

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

    body += struct.pack('>H', _table_length(sectors))
    for row in _table_rows(sectors):
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

    run_time_fields = {k[len('RunTime.'):]: v for k, v in record.items() if k.startswith('RunTime.')}
    pu_status_fields = {k[len('PuStatus.'):]: v for k, v in record.items() if k.startswith('PuStatus.')}
    body += _encode_em_run_time(run_time_fields)
    body += _encode_em_pu_status(pu_status_fields)

    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


for _em3raw_id in (140, 141, 142, 143, 144, 145, 146, 147, 148):
    _PING_SENSOR_SPECIFIC_CODECS[_em3raw_id] = ("EM3Raw", _decode_em3raw_specific, _encode_em3raw_specific)
del _em3raw_id


def _decode_em3_run_time(payload, pos):
    """
    Decode one gsfEM3RunTime sub-block: the older run-time-parameters
    struct used only by the plain (non-"_RAW") EM3_SPECIFIC family below,
    distinct from t_gsfEMRunTime and _decode_em_run_time(), which
    EM4_SPECIFIC and the EM3 "_RAW"-variant subrecords use instead. This
    is not a standalone subrecord with its own header and is not
    registered in any subrecord-id registry; it is used inline, 0, 1, or
    2 times, depending on EM3_SPECIFIC's run_time_id bitmask (see
    _decode_em3_specific()), once per sonar head for dual-head EM3000D
    systems. This is ported from the run-time-parameter field reads that
    the reference gsflib C library inlines directly in its
    DecodeEM3Specific() function in gsf_dec.c (duplicated there once per
    head).

    This function also computes the SwathWidth_m and CoverageSector_deg
    fields, matching gsf_dec.c's own post-decode derivation: each is
    derived from the port-side value alone (with the total then split
    evenly back into port and starboard halves) when the starboard-side
    value is zero, or as the sum of the port and starboard values
    otherwise. These two fields exist purely for read-side convenience --
    there is no separate wire storage for them, and
    _encode_em3_run_time() ignores them entirely.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this sub-block's
        data begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload, which is always 49.
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
    Encode one gsfEM3RunTime sub-block, returning only the sub-block's
    raw bytes with no subrecord header of its own, since it is an inline
    sub-block used by the plain (non-"_RAW") EM3_SPECIFIC family rather
    than a standalone subrecord. This is the inverse of
    _decode_em3_run_time(), except that the SwathWidth_m and
    CoverageSector_deg fields it accepts are ignored: they are derived,
    read-only fields with no wire storage of their own (see
    _decode_em3_run_time()'s docstring), and only PortSwathWidth_m,
    StbdSwathWidth_m, PortCoverageSector_deg, and StbdCoverageSector_deg
    are actually written. This is ported from the run-time-parameter
    field writes that the reference gsflib C library inlines directly in
    its EncodeEM3Specific() function in gsf_enc.c (duplicated there once
    per head).

    Deliberate deviation: the reference encoder writes the minimum
    depth, maximum depth, and pulse-length fields using a direct
    truncating cast with no +/-0.501 rounding offset added first, unlike
    every other scaled field in this block, which the reference encoder
    does round. This Python port instead uses the standard _gsf_round()
    convention for all of these fields, consistent with every other
    encoder function in this module.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_em3_run_time() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded sub-block, as bytes, with no subrecord header.
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


def new_em3_specific():
    """
    Return a new dictionary with every EM3 sensor-specific subrecord field
    name present, in the same shape _decode_em3_specific() returns, each
    pre-set to 0 or 0.0. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_em3_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a ping record's 'SensorSpecific',
    with 'SensorSpecificID' set to this family's subrecord ids (118, 119,
    120, 128, 129, 130, 131, 132, 139).

    :return: a dictionary with every EM3 sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'ModelNumber': 0, 'PingNumber': 0, 'SerialNumber': 0, 'SurfaceVelocity_mps': 0.0,
        'TransducerDepth_m': 0.0, 'ValidBeams': 0, 'SampleRate_Hz': 0,
        'DepthDifference_m': 0.0, 'OffsetMultiplier': 0,
    }


def new_em3_run_time():
    """
    Return a new dictionary with every per-row field name of the EM3
    sensor-specific subrecord's 'RunTime' table present, each pre-set to its
    default: one row of that table. _decode_em3_specific() builds each row
    starting from this same template, so the two can never define a
    different set of field names. A caller building the table can fill in
    one of these per row and assign pd.DataFrame([row, ...]) (or the
    equivalent dictionary of column name -> array) to
    record['SensorSpecific']['RunTime']. 'Head' (0 or 1) selects which
    sonar head's run-time block the row is.

    :return: a dictionary with every 'RunTime' column name as a key.
    """
    return {
        'Head': 0, 'ModelNumber': 0, 'PingTime': None, 'PingNumber': 0, 'SerialNumber': 0,
        'SystemStatus': 0, 'Mode': 0, 'FilterID': 0, 'MinDepth_m': 0.0, 'MaxDepth_m': 0.0,
        'Absorption_dBkm': 0.0, 'PulseLength_us': 0.0, 'TransmitBeamWidth_deg': 0.0,
        'PowerReduction_dB': 0, 'ReceiveBeamWidth_deg': 0.0, 'ReceiveBandwidth_Hz': 0,
        'ReceiveGain_dB': 0, 'CrossOverAngle_deg': 0, 'SsvSource': 0, 'PortSwathWidth_m': 0,
        'BeamSpacing': 0, 'PortCoverageSector_deg': 0, 'Stabilization': 0,
        'StbdCoverageSector_deg': 0, 'StbdSwathWidth_m': 0, 'HiloFreqAbsorpRatio': 0,
        'SwathWidth_m': 0, 'CoverageSector_deg': 0,
    }


def _decode_em3_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_SPECIFIC subrecord (the
    plain, non-"_RAW" EM3-series family, covering ids 118, 119, 120, 128,
    129, 130, 131, 132, and 139 -- the EM3000, EM1002, EM300, EM120,
    EM3002, EM3000D, EM3002D, EM121A_SIS, and EM2000 sonars
    respectively). This is ported from the reference gsflib C library's
    DecodeEM3Specific() function in gsf_dec.c.

    Unlike every other ping-level sensor-specific subrecord in this
    module, this one is genuinely variable-length. After its 17 fixed
    bytes plus a 4-byte run_time_id bitmask, bit 0 (0x1) of that bitmask
    gates whether a run-time-parameters sub-block for head 0 follows.
    Bit 1 (0x2) gates a second sub-block for head 1 (present on EM3000D
    dual-head systems), but -- matching DecodeEM3Specific()'s own
    nesting exactly -- bit 1 is only even inspected, and a head-1 block
    only ever decoded, when bit 0 is also set; a file with bit 1 set but
    bit 0 clear has no run-time blocks at all on the wire, since there is
    nothing there to read a head-1 block from. Each sub-block that is
    present is decoded by _decode_em3_run_time() and returned as one row,
    tagged with a 'Head' column (0 or 1), in record['RunTime'].

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM3_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, plus an optional 'RunTime' key
        holding a table (a dictionary of column name -> numpy array, one
        element per decoded run-time-parameters sub-block), present only when bit 0 of
        run_time_id was set (so at least one sub-block was decoded).
        bytes_consumed is the number of bytes read from payload.
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

    record = new_em3_specific()
    record['ModelNumber'] = model_number
    record['PingNumber'] = ping_number
    record['SerialNumber'] = serial_number
    record['SurfaceVelocity_mps'] = surface_velocity_raw / 10.0
    record['TransducerDepth_m'] = transducer_depth_raw / 100.0
    record['ValidBeams'] = valid_beams
    record['SampleRate_Hz'] = sample_rate
    record['DepthDifference_m'] = depth_difference_raw / 100.0
    record['OffsetMultiplier'] = offset_multiplier

    run_time_rows = []
    if run_time_id & 0x1:
        head0_fields, consumed = _decode_em3_run_time(payload, pos)
        pos += consumed
        run_time_rows.append(dict(new_em3_run_time(), **head0_fields, Head=0))

        if run_time_id & 0x2:
            head1_fields, consumed = _decode_em3_run_time(payload, pos)
            pos += consumed
            run_time_rows.append(dict(new_em3_run_time(), **head1_fields, Head=1))

    if run_time_rows:
        record['RunTime'] = _rows_to_table(run_time_rows)
    return record, pos - start


def _encode_em3_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_EM3xxx_SPECIFIC subrecord (one of
    ids 118, 119, 120, 128, 129, 130, 131, 132, or 139, chosen via
    record['SubrecordID']), including its own 4-byte subrecord id+size
    word. This is the inverse of _decode_em3_specific(), and is ported
    from the reference gsflib C library's EncodeEM3Specific() function in
    gsf_enc.c, with one deliberate improvement: the reference encoder, as
    currently shipped, hardcodes run_time_id to 1 unconditionally -- the
    code path that would set bit 1 to write a second, head-1
    run-time-parameters block for an EM3000D dual-head system's run-time
    update is entirely commented out and dead in gsf_enc.c, a real
    limitation of gsflib itself. Since both the wire format and
    DecodeEM3Specific() fully support writing zero, one (head 0 only), or
    two (both heads) run-time blocks, this encoder writes exactly the
    blocks the caller supplies via record['RunTime'] instead of always
    forcing exactly one.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_em3_specific() returns. An optional 'RunTime'
        key holds a table (a dictionary of column name -> array-like, or
        a pandas.DataFrame) in which each row needs a 'Head' column (0 or
        1) selecting which position it is written at -- the exact table
        _decode_em3_specific() returns can be passed straight back in
        with no conversion required. Bit 0 of the
        on-disk run_time_id is set if and only if a row with Head == 0
        is present, and bit 1 is set if and only if a row with Head == 1
        is present; an empty or absent table writes run_time_id = 0 (no
        run-time blocks at all). An optional 'SubrecordID' key selects
        which of the nine registered ids to stamp, defaulting to 118
        (EM3000) if absent. A field missing from record is written as
        zero.

    :return: the encoded subrecord, as bytes, including its own header.

    :raises ValueError: raised if a row's 'Head' value is neither 0 nor
        1; if more than one row shares the same 'Head' value; or if a
        row with Head == 1 is present without a matching row with
        Head == 0, since the wire format has no way to represent a
        head-1 block on its own -- matching DecodeEM3Specific()'s own
        nesting, in which bit 1 only means anything, and a head-1 block
        is only ever written, alongside a head-0 block.
    """
    subrecord_id = record.get('SubrecordID', 118)
    g = record.get
    rows_by_head = {}
    for row in _table_rows(record.get('RunTime')):
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


def new_echotrac_specific():
    """
    Return a new dictionary with every Echotrac sensor-specific subrecord
    field name present, in the same shape _decode_echotrac_specific()
    returns, each pre-set to 0 or 0.0. gsf.h defines no "not available"
    value for these vendor-specific fields, so these are the only defaults
    available. _decode_echotrac_specific() builds its result starting from
    this same template, so the two can never define a different set of field
    names. To write one, fill it in and set it as a single-beam ping
    record's 'SensorSpecific', with 'SensorSpecificID' set to this family's
    subrecord ids (201, 202).

    :return: a dictionary with every Echotrac sensor-specific field name as
        a key, pre-set to its default.
    """
    return {
        'NavigationError': 0, 'MppSource': 0, 'TideSource': 0,
    }


def _decode_echotrac_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_ECHOTRAC_SPECIFIC subrecord (id
    201) or a GSF_SINGLE_BEAM_SUBRECORD_BATHY2000_SPECIFIC subrecord (id
    202), part of the GSF_RECORD_SINGLE_BEAM_PING sensor-specific
    registry. Both ids share an identical wire format, so gsflib decodes
    them with a single function, and this function is ported from the
    reference gsflib C library's DecodeEchotracSpecific() function in
    gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an ECHOTRAC_SPECIFIC or BATHY2000_SPECIFIC
    subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (navigation_error,) = struct.unpack_from('>h', payload, pos); pos += 2
    mpp_source = payload[pos]; pos += 1
    tide_source = payload[pos]; pos += 1

    fields = new_echotrac_specific()
    fields['NavigationError'] = navigation_error
    fields['MppSource'] = mpp_source
    fields['TideSource'] = tide_source
    return fields, pos - start


def _encode_echotrac_specific(record):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_ECHOTRAC_SPECIFIC (id 201) or
    _BATHY2000_SPECIFIC (id 202) subrecord, including its own four-byte
    subrecord ID and size word. This is the inverse of
    _decode_echotrac_specific(), and is ported from the reference gsflib
    C library's EncodeEchotracSpecific() function in gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_echotrac_specific() returns, plus an optional
        'SubrecordID' key choosing which of the two subrecord ids to
        write (201 for Echotrac or 202 for Bathy2000); if that key is
        absent, 201 is written. A field that is missing from the
        dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    subrecord_id = record.get('SubrecordID', 201)
    g = record.get
    body = struct.pack('>h', int(g('NavigationError', 0)))
    body += struct.pack('>B', int(g('MppSource', 0)) & 0xFF)
    body += struct.pack('>B', int(g('TideSource', 0)) & 0xFF)
    header_word = ((subrecord_id & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[201] = ("Echotrac", _decode_echotrac_specific, _encode_echotrac_specific)
_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[202] = ("Bathy2000", _decode_echotrac_specific, _encode_echotrac_specific)


def new_mgd77_specific():
    """
    Return a new dictionary with every MGD77 sensor-specific subrecord field
    name present, in the same shape _decode_mgd77_specific() returns, each
    pre-set to 0 or 0.0. gsf.h defines no "not available" value for these
    vendor-specific fields, so these are the only defaults available.
    _decode_mgd77_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a single-beam ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (203).

    :return: a dictionary with every MGD77 sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'TimeZoneCorr': 0, 'PositionTypeCode': 0, 'CorrectionCode': 0, 'BathyTypeCode': 0,
        'QualityCode': 0, 'TravelTime_sec': 0.0,
    }


def _decode_mgd77_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_MGD77_SPECIFIC subrecord (id
    203), which holds MGD77-format survey trackline metadata for a
    single-beam ping. This is ported from the reference gsflib C
    library's DecodeMGD77Specific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing an MGD77_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (time_zone_corr,) = struct.unpack_from('>H', payload, pos); pos += 2
    (position_type_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (correction_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (bathy_type_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (quality_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (travel_time_raw,) = struct.unpack_from('>I', payload, pos); pos += 4

    fields = new_mgd77_specific()
    fields['TimeZoneCorr'] = time_zone_corr
    fields['PositionTypeCode'] = position_type_code
    fields['CorrectionCode'] = correction_code
    fields['BathyTypeCode'] = bathy_type_code
    fields['QualityCode'] = quality_code
    fields['TravelTime_sec'] = travel_time_raw / 10000.0
    return fields, pos - start


def _encode_mgd77_specific(record):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_MGD77_SPECIFIC subrecord (id 203),
    including its own four-byte subrecord ID and size word. This is the
    inverse of _decode_mgd77_specific(), and is ported from the
    reference gsflib C library's EncodeMGD77Specific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_mgd77_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>H', int(g('TimeZoneCorr', 0)))
    body += struct.pack('>H', int(g('PositionTypeCode', 0)))
    body += struct.pack('>H', int(g('CorrectionCode', 0)))
    body += struct.pack('>H', int(g('BathyTypeCode', 0)))
    body += struct.pack('>H', int(g('QualityCode', 0)))
    body += struct.pack('>I', _gsf_round(float(g('TravelTime_sec', 0.0)) * 10000.0))
    header_word = ((203 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[203] = ("MGD77", _decode_mgd77_specific, _encode_mgd77_specific)


def new_bdb_specific():
    """
    Return a new dictionary with every BDB sensor-specific subrecord field
    name present, in the same shape _decode_bdb_specific() returns, each
    pre-set to 0 or 0.0, or an empty string. gsf.h defines no "not
    available" value for these vendor-specific fields, so these are the only
    defaults available. _decode_bdb_specific() builds its result starting
    from this same template, so the two can never define a different set of
    field names. To write one, fill it in and set it as a single-beam ping
    record's 'SensorSpecific', with 'SensorSpecificID' set to this family's
    subrecord id (204).

    :return: a dictionary with every BDB sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'DocNo': 0, 'Eval': '', 'Classification': '', 'TrackAdjFlag': '', 'SourceFlag': '',
        'PtOrTrackLn': '', 'DatumFlag': '',
    }


def _decode_bdb_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_BDB_SPECIFIC subrecord (id 204),
    which holds BDB-format survey trackline metadata for a single-beam
    ping. This is ported from the reference gsflib C library's
    DecodeBDBSpecific() function in gsf_dec.c. Every flag field in this
    subrecord is stored on disk as a single ASCII character (for
    example, per gsf.h's field comments, eval is one of '1' through '4'
    and datum_flag is 'W' or 'D'), so each of those fields is decoded
    here as a one-character string rather than as a raw integer.

    This function has not been tested against a verified GSF file, since
    no sample data containing a BDB_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (doc_no,) = struct.unpack_from('>i', payload, pos); pos += 4
    # An unset (NUL) flag byte decodes to '', as _encode_bdb_specific() writes it.
    eval_flag = payload[pos:pos + 1].decode('ascii', 'replace').rstrip('\x00'); pos += 1
    classification = payload[pos:pos + 1].decode('ascii', 'replace').rstrip('\x00'); pos += 1
    track_adj_flag = payload[pos:pos + 1].decode('ascii', 'replace').rstrip('\x00'); pos += 1
    source_flag = payload[pos:pos + 1].decode('ascii', 'replace').rstrip('\x00'); pos += 1
    pt_or_track_ln = payload[pos:pos + 1].decode('ascii', 'replace').rstrip('\x00'); pos += 1
    datum_flag = payload[pos:pos + 1].decode('ascii', 'replace').rstrip('\x00'); pos += 1

    fields = new_bdb_specific()
    fields['DocNo'] = doc_no
    fields['Eval'] = eval_flag
    fields['Classification'] = classification
    fields['TrackAdjFlag'] = track_adj_flag
    fields['SourceFlag'] = source_flag
    fields['PtOrTrackLn'] = pt_or_track_ln
    fields['DatumFlag'] = datum_flag
    return fields, pos - start


def _encode_bdb_specific(record):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_BDB_SPECIFIC subrecord (id 204),
    including its own four-byte subrecord ID and size word. This is the
    inverse of _decode_bdb_specific(), and is ported from the reference
    gsflib C library's EncodeBDBSpecific() function in gsf_enc.c. Each
    flag field is written as the first byte of the given string,
    matching the single-ASCII-character on-disk format described in
    _decode_bdb_specific(); a field missing from record, or given as an
    empty string, is written as a NUL byte instead of zero.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_bdb_specific() returns.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    def _char(value):
        """ Encode value as a single ASCII byte, or NUL if value is missing or empty. """
        text = str(value) if value else '\x00'
        return text.encode('ascii')[:1]

    g = record.get
    body = struct.pack('>i', int(g('DocNo', 0)))
    body += _char(g('Eval', ''))
    body += _char(g('Classification', ''))
    body += _char(g('TrackAdjFlag', ''))
    body += _char(g('SourceFlag', ''))
    body += _char(g('PtOrTrackLn', ''))
    body += _char(g('DatumFlag', ''))
    header_word = ((204 & 0xFF) << 24) | len(body)
    return struct.pack('>I', header_word) + body


_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[204] = ("BDB", _decode_bdb_specific, _encode_bdb_specific)


def new_noshdb_specific():
    """
    Return a new dictionary with every NOSHDB sensor-specific subrecord
    field name present, in the same shape _decode_noshdb_specific() returns,
    each pre-set to 0 or 0.0. gsf.h defines no "not available" value for
    these vendor-specific fields, so these are the only defaults available.
    _decode_noshdb_specific() builds its result starting from this same
    template, so the two can never define a different set of field names. To
    write one, fill it in and set it as a single-beam ping record's
    'SensorSpecific', with 'SensorSpecificID' set to this family's subrecord
    id (205).

    :return: a dictionary with every NOSHDB sensor-specific field name as a
        key, pre-set to its default.
    """
    return {
        'TypeCode': 0, 'CartoCode': 0,
    }


def _decode_noshdb_specific(payload, pos):
    """
    Decode a GSF_SINGLE_BEAM_SUBRECORD_NOSHDB_SPECIFIC subrecord (id
    205), which holds NOS HDB-format survey trackline metadata for a
    single-beam ping. This is ported from the reference gsflib C
    library's DecodeNOSHDBSpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a NOSHDB_SPECIFIC subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a dictionary
        of the decoded scalar fields, and bytes_consumed is the number of
        bytes read from payload.
    """
    start = pos
    (type_code,) = struct.unpack_from('>H', payload, pos); pos += 2
    (carto_code,) = struct.unpack_from('>H', payload, pos); pos += 2

    fields = new_noshdb_specific()
    fields['TypeCode'] = type_code
    fields['CartoCode'] = carto_code
    return fields, pos - start


def _encode_noshdb_specific(record):
    """
    Encode a GSF_SINGLE_BEAM_SUBRECORD_NOSHDB_SPECIFIC subrecord (id
    205), including its own four-byte subrecord ID and size word. This
    is the inverse of _decode_noshdb_specific(), and is ported from the
    reference gsflib C library's EncodeNOSHDBSpecific() function in
    gsf_enc.c.

    :param record: a dictionary of the scalar fields to encode, in the
        same shape _decode_noshdb_specific() returns. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded subrecord, as bytes, including its own header.
    """
    g = record.get
    body = struct.pack('>H', int(g('TypeCode', 0)))
    body += struct.pack('>H', int(g('CartoCode', 0)))
    header_word = ((205 & 0xFF) << 24) | len(body)
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
    Skip over the KMALL sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21).
    Some sonar families place a small, vendor-specific "imagery
    preamble" at the start of this subrecord's per-beam backscatter time
    series, before the actual per-beam samples begin; that preamble is a
    separate, smaller block from the family's own ping-level
    sensor-specific subrecord, and lives only inside subrecord 21. This
    function is called by _decode_brb_intensity() whenever the ping's
    sensor-specific subrecord id identifies a KMALL system. It is
    ported from the reference gsflib C library's
    DecodeKMALLImagerySpecific() function in gsf_dec.c, which leaves
    this preamble entirely spare/reserved for the KMALL sensor, so this
    function exists only to advance past its 64 bytes; unlike its
    sibling _decode_*_imagery_specific() functions, it has no fields to
    decode and so returns only the byte count rather than a (fields,
    bytes_consumed) tuple.

    :param payload: the raw bytes of the ping record. Unused, since this
        preamble's content is entirely skipped, but accepted for a
        consistent call signature with the other
        _decode_*_imagery_specific() functions.
    :param pos: the byte offset within payload where this preamble
        begins. Also unused, for the same reason.

    :return: bytes_consumed, always 64.
    """
    return 64


def _encode_kmall_imagery_specific():
    """
    Encode the KMALL sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21),
    which is entirely spare/reserved for the KMALL sensor. This is the
    inverse of _decode_kmall_imagery_specific(), and is ported from the
    reference gsflib C library's EncodeKMALLImagerySpecific() function in
    gsf_enc.c. It takes no fields, since there is nothing in this
    preamble to encode.

    :return: the encoded preamble, as bytes. Always 64 bytes, all zero.
    """
    return b'\x00' * 64


def new_em3_imagery_specific():
    """
    Return a new dictionary with every EM3-series sensor-imagery preamble
    field name present, each pre-set to 0 or 0.0. gsf.h defines no
    GSF_NULL_* sentinel for these vendor-specific fields, so 0 is the
    only "not specified" marker available. _decode_em3_imagery_specific()
    builds its result starting from this same template, so the two can
    never define a different set of field names.

    :return: a dictionary with every EM3-series imagery preamble field
        name present, pre-set to 0 or 0.0.
    """
    return {
        'RangeNorm_samples': 0, 'StartTvgRamp_samples': 0, 'StopTvgRamp_samples': 0,
        'BSNormal_dB': 0, 'BSOblique_dB': 0, 'MeanAbsorption_dBkm': 0.0,
        'Offset': 0, 'Scale': 0,
    }


def _decode_em3_imagery_specific(payload, pos):
    """
    Decode the EM3-series sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21),
    covering the older Simrad/Kongsberg EM3000, EM1002, EM300, EM120,
    EM3002, EM3000D, EM3002D, EM121A-SIS, and EM2000 sonars, and their
    "_RAW" range/angle variants. Some sonar families place a small,
    vendor-specific "imagery preamble" at the start of this subrecord's
    per-beam backscatter time series, before the actual per-beam samples
    begin; that preamble is a separate, smaller block from the family's
    own ping-level sensor-specific subrecord, and lives only inside
    subrecord 21. This function is called by _decode_brb_intensity() to
    decode that preamble, and is ported from the reference gsflib C
    library's DecodeEM3ImagerySpecific() function in gsf_dec.c. Its
    result starts from new_em3_imagery_specific()'s template, so its
    field names always match what _encode_em3_imagery_specific() expects.

    This function has not been tested against a verified GSF file, since
    no sample data containing an EM3-series intensity series subrecord
    is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this preamble
        begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        of the decoded scalar fields, and bytes_consumed is always 18.
    """
    start = pos
    fields = new_em3_imagery_specific()
    (fields['RangeNorm_samples'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (fields['StartTvgRamp_samples'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (fields['StopTvgRamp_samples'],) = struct.unpack_from('>H', payload, pos); pos += 2
    fields['BSNormal_dB'] = payload[pos]; pos += 1
    fields['BSOblique_dB'] = payload[pos]; pos += 1
    (mean_absorption_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    fields['MeanAbsorption_dBkm'] = mean_absorption_raw / 100.0
    (fields['Offset'],) = struct.unpack_from('>h', payload, pos); pos += 2
    (fields['Scale'],) = struct.unpack_from('>h', payload, pos); pos += 2
    pos += 4  # spare

    return fields, pos - start


def _encode_em3_imagery_specific(fields):
    """
    Encode the EM3-series sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21).
    This is the inverse of _decode_em3_imagery_specific(), and is ported
    from the reference gsflib C library's EncodeEM3ImagerySpecific()
    function in gsf_enc.c. BSNormal_dB and BSOblique_dB are written as
    plain truncating byte casts with no rounding, matching how
    _decode_em3_imagery_specific() reads them: as whole-number bytes
    with no fractional scale factor applied.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_em3_imagery_specific() returns or
        new_em3_imagery_specific() creates. A field that is missing from
        the dictionary is written as zero.

    :return: the encoded preamble, as bytes. Always 18 bytes.
    """
    g = fields.get
    body = struct.pack('>H', int(g('RangeNorm_samples', 0)))
    body += struct.pack('>H', int(g('StartTvgRamp_samples', 0)))
    body += struct.pack('>H', int(g('StopTvgRamp_samples', 0)))
    body += struct.pack('>B', int(g('BSNormal_dB', 0)) & 0xFF)
    body += struct.pack('>B', int(g('BSOblique_dB', 0)) & 0xFF)
    body += struct.pack('>H', _gsf_round(g('MeanAbsorption_dBkm', 0.0) * 100.0))
    body += struct.pack('>h', int(g('Offset', 0)))
    body += struct.pack('>h', int(g('Scale', 0)))
    body += b'\x00' * 4  # spare
    return body


def new_em4_imagery_specific():
    """
    Return a new dictionary with every EM4-series sensor-imagery preamble
    field name present, each pre-set to 0 or 0.0. gsf.h defines no
    GSF_NULL_* sentinel for these vendor-specific fields, so 0 is the
    only "not specified" marker available. _decode_em4_imagery_specific()
    builds its result starting from this same template, so the two can
    never define a different set of field names.

    :return: a dictionary with every EM4-series imagery preamble field
        name present, pre-set to 0 or 0.0.
    """
    return {
        'SamplingFrequency_Hz': 0.0, 'MeanAbsorption_dBkm': 0.0,
        'TxPulseLength_us': 0, 'RangeNorm_samples': 0, 'StartTvgRamp_samples': 0,
        'StopTvgRamp_samples': 0, 'BSNormal_dB': 0.0, 'BSOblique_dB': 0.0,
        'TxBeamWidth_deg': 0.0, 'TvgCrossOver_deg': 0.0, 'Offset': 0, 'Scale': 0,
    }


def _decode_em4_imagery_specific(payload, pos):
    """
    Decode the EM4-series sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21),
    covering the EM122, EM302, EM710, EM2040, and ME70BO sonars. Some
    sonar families place a small, vendor-specific "imagery preamble" at
    the start of this subrecord's per-beam backscatter time series,
    before the actual per-beam samples begin; that preamble is a
    separate, smaller block from the family's own ping-level
    sensor-specific subrecord, and lives only inside subrecord 21. This
    function is called by _decode_brb_intensity() to decode that
    preamble, and is ported from the reference gsflib C library's
    DecodeEM4ImagerySpecific() function in gsf_dec.c. Its result starts
    from new_em4_imagery_specific()'s template, so its field names always
    match what _encode_em4_imagery_specific() expects.

    This function has not been tested against a verified GSF file, since
    none of the sample files checked into this project carry an
    intensity series subrecord for these sensors -- the EM712 sample
    data present is all KMALL_SPECIFIC, which uses the separate
    Kongsberg SIS 5 imagery format decoded by
    _decode_kmall_imagery_specific() instead.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this preamble
        begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        of the decoded scalar fields, and bytes_consumed is always 50.
    """
    start = pos
    fields = new_em4_imagery_specific()
    (freq_int,) = struct.unpack_from('>I', payload, pos); pos += 4
    (freq_frac,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['SamplingFrequency_Hz'] = freq_int + freq_frac / 4.0e9
    (mean_absorption_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    fields['MeanAbsorption_dBkm'] = mean_absorption_raw / 100.0
    (fields['TxPulseLength_us'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (fields['RangeNorm_samples'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (fields['StartTvgRamp_samples'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (fields['StopTvgRamp_samples'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (bsn_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    fields['BSNormal_dB'] = bsn_raw / 10.0
    (bso_raw,) = struct.unpack_from('>h', payload, pos); pos += 2
    fields['BSOblique_dB'] = bso_raw / 10.0
    (tx_beam_width_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    fields['TxBeamWidth_deg'] = tx_beam_width_raw / 10.0
    (tvg_cross_over_raw,) = struct.unpack_from('>H', payload, pos); pos += 2
    fields['TvgCrossOver_deg'] = tvg_cross_over_raw / 10.0
    (fields['Offset'],) = struct.unpack_from('>h', payload, pos); pos += 2
    (fields['Scale'],) = struct.unpack_from('>h', payload, pos); pos += 2
    pos += 20  # spare

    return fields, pos - start


def _encode_em4_imagery_specific(fields):
    """
    Encode the EM4-series sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21).
    This is the inverse of _decode_em4_imagery_specific(), and is ported
    from the reference gsflib C library's EncodeEM4ImagerySpecific()
    function in gsf_enc.c.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_em4_imagery_specific() returns or
        new_em4_imagery_specific() creates. A field that is missing from
        the dictionary is written as zero.

    :return: the encoded preamble, as bytes. Always 50 bytes.
    """
    g = fields.get
    sampling_frequency = g('SamplingFrequency_Hz', 0.0)
    freq_int = int(sampling_frequency)
    freq_frac = _gsf_round((sampling_frequency - freq_int) * 4.0e9)
    body = struct.pack('>I', freq_int)
    body += struct.pack('>I', freq_frac)
    body += struct.pack('>H', _gsf_round(g('MeanAbsorption_dBkm', 0.0) * 100.0))
    body += struct.pack('>H', _gsf_round(g('TxPulseLength_us', 0.0)))
    body += struct.pack('>H', int(g('RangeNorm_samples', 0)))
    body += struct.pack('>H', int(g('StartTvgRamp_samples', 0)))
    body += struct.pack('>H', int(g('StopTvgRamp_samples', 0)))
    body += struct.pack('>h', _gsf_round(g('BSNormal_dB', 0.0) * 10.0))
    body += struct.pack('>h', _gsf_round(g('BSOblique_dB', 0.0) * 10.0))
    body += struct.pack('>H', _gsf_round(g('TxBeamWidth_deg', 0.0) * 10.0))
    body += struct.pack('>H', _gsf_round(g('TvgCrossOver_deg', 0.0) * 10.0))
    body += struct.pack('>h', int(g('Offset', 0)))
    body += struct.pack('>h', int(g('Scale', 0)))
    body += b'\x00' * 20  # spare
    return body


def new_reson_size_spare_imagery_specific():
    """
    Return a new dictionary with the single field name used by the
    Reson 7125 and Reson T-series sensor-imagery preamble, pre-set to 0.
    _decode_reson_size_spare_imagery_specific() builds its result
    starting from this same template, so the two can never define a
    different set of field names.

    :return: a dictionary with a single 'Size' key, pre-set to 0.
    """
    return {'Size': 0}


def _decode_reson_size_spare_imagery_specific(payload, pos):
    """
    Decode the sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21)
    for the Reson 7125 and Reson T-series sonars, which share an
    identical 66-byte preamble layout: a two-byte record size followed
    by 64 spare bytes. Some sonar families place a small, vendor-specific
    "imagery preamble" at the start of this subrecord's per-beam
    backscatter time series, before the actual per-beam samples begin;
    that preamble is a separate, smaller block from the family's own
    ping-level sensor-specific subrecord, and lives only inside
    subrecord 21. This function is called by _decode_brb_intensity() to
    decode that preamble, and is ported from the reference gsflib C
    library's DecodeReson7100ImagerySpecific() and
    DecodeResonTSeriesImagerySpecific() functions in gsf_dec.c, which
    are byte-for-byte identical apart from their names, so this single
    function serves both. Its result starts from
    new_reson_size_spare_imagery_specific()'s template, so its field
    names always match what _encode_reson_size_spare_imagery_specific()
    expects.

    This function has not been tested against a verified GSF file, since
    no sample data containing a Reson 7125 or Reson T-series intensity
    series subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this preamble
        begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        with a single 'Size' key holding the decoded record size, and
        bytes_consumed is always 66.
    """
    start = pos
    fields = new_reson_size_spare_imagery_specific()
    (fields['Size'],) = struct.unpack_from('>H', payload, pos); pos += 2
    pos += 64  # spare
    return fields, pos - start


def _encode_reson_size_spare_imagery_specific(fields):
    """
    Encode the sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21)
    for the Reson 7125 and Reson T-series sonars. This is the inverse of
    _decode_reson_size_spare_imagery_specific(), and is ported from the
    reference gsflib C library's EncodeReson7100ImagerySpecific() and
    EncodeResonTSeriesImagerySpecific() functions in gsf_enc.c, which are
    byte-for-byte identical apart from their names, so this single
    function serves both.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_reson_size_spare_imagery_specific() returns
        or new_reson_size_spare_imagery_specific() creates. A field that
        is missing from the dictionary is written as zero.

    :return: the encoded preamble, as bytes. Always 66 bytes.
    """
    body = struct.pack('>H', int(fields.get('Size', 0)))
    body += b'\x00' * 64  # spare
    return body


def _decode_reson8100_imagery_specific(payload, pos):
    """
    Skip over the sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21)
    for the Reson 8100-family sonars (8101, 8111, 8124, 8125, 8150, and
    8160), whose preamble is entirely spare/reserved. Some sonar
    families place a small, vendor-specific "imagery preamble" at the
    start of this subrecord's per-beam backscatter time series, before
    the actual per-beam samples begin; that preamble is a separate,
    smaller block from the family's own ping-level sensor-specific
    subrecord, and lives only inside subrecord 21. This function is
    called by _decode_brb_intensity() to decode that preamble, and is
    ported from the reference gsflib C library's
    DecodeReson8100ImagerySpecific() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a Reson 8100-family intensity series
    subrecord is available.

    :param payload: the raw bytes of the ping record. Unused, since this
        preamble's content is entirely skipped, but accepted for a
        consistent call signature with the other
        _decode_*_imagery_specific() functions.
    :param pos: the byte offset within payload where this preamble
        begins. Also unused, for the same reason.

    :return: a tuple of (fields, bytes_consumed). fields is always an
        empty dictionary, and bytes_consumed is always 8.
    """
    return {}, 8


def _encode_reson8100_imagery_specific():
    """
    Encode the sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21)
    for the Reson 8100-family sonars, whose preamble is entirely
    spare/reserved. This is the inverse of
    _decode_reson8100_imagery_specific(), and is ported from the
    reference gsflib C library's EncodeReson8100ImagerySpecific()
    function in gsf_enc.c. It takes no fields, since there is nothing in
    this preamble to encode.

    :return: the encoded preamble, as bytes. Always 8 bytes, all zero.
    """
    return b'\x00' * 8


def new_klein5410bss_imagery_specific():
    """
    Return a new dictionary with every Klein 5410 BSS sensor-imagery
    preamble field name present, each pre-set to 0 or an all-zero list.
    gsf.h defines no GSF_NULL_* sentinel for these vendor-specific
    fields, so 0 is the only "not specified" marker available.
    _decode_klein5410bss_imagery_specific() builds its result starting
    from this same template, so the two can never define a different set
    of field names.

    :return: a dictionary with every Klein 5410 BSS imagery preamble
        field name present, pre-set to 0 or an all-zero five-element
        list.
    """
    return {'ResMode': 0, 'TvgPage': 0, 'BeamID': [0, 0, 0, 0, 0]}


def _decode_klein5410bss_imagery_specific(payload, pos):
    """
    Decode the Klein 5410 BSS sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21).
    Some sonar families place a small, vendor-specific "imagery
    preamble" at the start of this subrecord's per-beam backscatter time
    series, before the actual per-beam samples begin; that preamble is a
    separate, smaller block from the family's own ping-level
    sensor-specific subrecord, and lives only inside subrecord 21. This
    function is called by _decode_brb_intensity() to decode that
    preamble, and is ported from the reference gsflib C library's
    DecodeKlein5410BssImagerySpecific() function in gsf_dec.c. Its result
    starts from new_klein5410bss_imagery_specific()'s template, so its
    field names always match what
    _encode_klein5410bss_imagery_specific() expects.

    This function has not been tested against a verified GSF file, since
    no sample data containing a Klein 5410 BSS intensity series
    subrecord is available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this preamble
        begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        of the decoded scalar fields, and bytes_consumed is always 18.
    """
    start = pos
    fields = new_klein5410bss_imagery_specific()
    (fields['ResMode'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (fields['TvgPage'],) = struct.unpack_from('>H', payload, pos); pos += 2
    fields['BeamID'] = list(struct.unpack_from('>5H', payload, pos)); pos += 10
    pos += 4  # spare
    return fields, pos - start


def _encode_klein5410bss_imagery_specific(fields):
    """
    Encode the Klein 5410 BSS sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21).
    This is the inverse of _decode_klein5410bss_imagery_specific(), and
    is ported from the reference gsflib C library's
    EncodeKlein5410BssImagerySpecific() function in gsf_enc.c.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_klein5410bss_imagery_specific() returns or
        new_klein5410bss_imagery_specific() creates. A field that is
        missing from the dictionary is written as zero.

    :return: the encoded preamble, as bytes. Always 18 bytes.
    """
    beam_id = list(fields.get('BeamID') or [0, 0, 0, 0, 0])
    beam_id = (beam_id + [0, 0, 0, 0, 0])[:5]
    body = struct.pack('>H', int(fields.get('ResMode', 0)))
    body += struct.pack('>H', int(fields.get('TvgPage', 0)))
    body += struct.pack('>5H', *[int(v) for v in beam_id])
    body += b'\x00' * 4  # spare
    return body


def new_r2sonic_imagery_specific():
    """
    Return a new dictionary with every R2Sonic sensor-imagery preamble
    field name present, pre-set to 0, 0.0, an empty string, an all-zero
    six-element list, or None for 'PingTime'. gsf.h defines no
    GSF_NULL_* sentinel for these vendor-specific fields, so these are
    the only "not specified" markers available.
    _decode_r2sonic_imagery_specific() builds its result starting from
    this same template, so the two can never define a different set of
    field names.

    :return: a dictionary with every R2Sonic imagery preamble field name
        present, pre-set to its default value.
    """
    return {
        'ModelNumber': "", 'SerialNumber': "", 'PingTime': None, 'PingNumber': 0,
        'PingPeriod_s': 0.0, 'SoundSpeed_mps': 0.0, 'Frequency_Hz': 0.0,
        'TxPower_dB': 0.0, 'TxPulseWidth_s': 0.0,
        'TxBeamwidthVert_deg': 0.0, 'TxBeamwidthHoriz_deg': 0.0,
        'TxSteeringVert_deg': 0.0, 'TxSteeringHoriz_deg': 0.0, 'TxMiscInfo': 0,
        'RxBandwidth_Hz': 0.0, 'RxSampleRate_Hz': 0.0, 'RxRange_m': 0.0,
        'RxGain_dB': 0.0, 'RxSpreading': 0.0, 'RxAbsorption_dBkm': 0.0,
        'RxMountTilt_deg': 0.0, 'RxMiscInfo': 0,
        'Reserved': 0, 'NumBeams': 0, 'MoreInfo': [0.0] * 6,
    }


def _decode_r2sonic_imagery_specific(payload, pos):
    """
    Decode the R2Sonic sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21),
    covering the R2Sonic 2020, 2022, and 2024 sonars. Some sonar
    families place a small, vendor-specific "imagery preamble" at the
    start of this subrecord's per-beam backscatter time series, before
    the actual per-beam samples begin; that preamble is a separate,
    smaller block from the family's own ping-level sensor-specific
    subrecord, and lives only inside subrecord 21. This function is
    called by _decode_brb_intensity() to decode that preamble, and is
    ported from the reference gsflib C library's
    DecodeR2SonicImagerySpecific() function in gsf_dec.c. Its result
    starts from new_r2sonic_imagery_specific()'s template, so its field
    names always match what _encode_r2sonic_imagery_specific() expects.

    This function has not been tested against a verified GSF file, since
    no sample data containing an R2Sonic intensity series subrecord is
    available.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this preamble
        begins.

    :return: a tuple of (fields, bytes_consumed). fields is a dictionary
        of the decoded scalar fields, and bytes_consumed is always 168.
    """
    start = pos
    fields = new_r2sonic_imagery_specific()
    fields['ModelNumber'] = payload[pos:pos + 12].split(b'\x00', 1)[0].decode('ascii', 'replace'); pos += 12
    fields['SerialNumber'] = payload[pos:pos + 12].split(b'\x00', 1)[0].decode('ascii', 'replace'); pos += 12
    (sec,) = struct.unpack_from('>i', payload, pos); pos += 4
    (nsec,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['PingTime'] = _gsf_timestamp(sec, nsec)
    (fields['PingNumber'],) = struct.unpack_from('>I', payload, pos); pos += 4
    (ping_period_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['PingPeriod_s'] = ping_period_raw / 1.0e6
    (sound_speed_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['SoundSpeed_mps'] = sound_speed_raw / 1.0e2
    (frequency_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['Frequency_Hz'] = frequency_raw / 1.0e3
    (tx_power_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['TxPower_dB'] = tx_power_raw / 1.0e2
    (tx_pulse_width_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['TxPulseWidth_s'] = tx_pulse_width_raw / 1.0e7
    (tx_beamwidth_vert_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['TxBeamwidthVert_deg'] = tx_beamwidth_vert_raw / 1.0e6
    (tx_beamwidth_horiz_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['TxBeamwidthHoriz_deg'] = tx_beamwidth_horiz_raw / 1.0e6
    (tx_steering_vert_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    fields['TxSteeringVert_deg'] = tx_steering_vert_raw / 1.0e6
    (tx_steering_horiz_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    fields['TxSteeringHoriz_deg'] = tx_steering_horiz_raw / 1.0e6
    (fields['TxMiscInfo'],) = struct.unpack_from('>I', payload, pos); pos += 4
    (rx_bandwidth_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['RxBandwidth_Hz'] = rx_bandwidth_raw / 1.0e4
    (rx_sample_rate_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['RxSampleRate_Hz'] = rx_sample_rate_raw / 1.0e3
    (rx_range_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['RxRange_m'] = rx_range_raw / 1.0e5
    (rx_gain_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['RxGain_dB'] = rx_gain_raw / 1.0e2
    (rx_spreading_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['RxSpreading'] = rx_spreading_raw / 1.0e3
    (rx_absorption_raw,) = struct.unpack_from('>I', payload, pos); pos += 4
    fields['RxAbsorption_dBkm'] = rx_absorption_raw / 1.0e3
    (rx_mount_tilt_raw,) = struct.unpack_from('>i', payload, pos); pos += 4
    fields['RxMountTilt_deg'] = rx_mount_tilt_raw / 1.0e6
    (fields['RxMiscInfo'],) = struct.unpack_from('>I', payload, pos); pos += 4
    (fields['Reserved'],) = struct.unpack_from('>H', payload, pos); pos += 2
    (fields['NumBeams'],) = struct.unpack_from('>H', payload, pos); pos += 2
    fields['MoreInfo'] = [v / 1.0e6 for v in struct.unpack_from('>6i', payload, pos)]; pos += 24
    pos += 32  # spare
    return fields, pos - start


def _encode_r2sonic_imagery_specific(fields):
    """
    Encode the R2Sonic sensor-imagery preamble embedded in a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21).
    This is the inverse of _decode_r2sonic_imagery_specific(), and is
    ported from the reference gsflib C library's
    EncodeR2SonicImagerySpecific() function in gsf_enc.c.

    The reference C encoder rounds most fields by adding an unconditional
    0.501 before truncating, which is only correct for fields that are
    physically non-negative, and rounds a handful of signed fields (the
    vertical and horizontal transmit steering angles, the receive mount
    tilt, and the "more info" array) with a sign-aware branch that adds
    0.501 or subtracts 0.501 depending on the sign of the value. This
    Python port instead uses the standard sign-correct _gsf_round()
    convention for every field. That convention is numerically
    equivalent to the reference encoder's unconditional +0.501 for the
    non-negative fields, and matches the reference encoder's sign-aware
    branch exactly for the signed fields, so the encoded output is the
    same either way.

    :param fields: a dictionary of the scalar fields to encode, in the
        same shape _decode_r2sonic_imagery_specific() returns or
        new_r2sonic_imagery_specific() creates. A field that is missing
        from the dictionary is written as zero. 'PingTime' of None is
        written as zero seconds and nanoseconds.

    :return: the encoded preamble, as bytes. Always 168 bytes.
    """
    g = fields.get

    def model_bytes(key):
        return (g(key) or "").encode('ascii')[:12].ljust(12, b'\x00')

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
    more_info = list(g('MoreInfo') or [0.0] * 6)
    more_info = (more_info + [0.0] * 6)[:6]
    body += struct.pack('>6i', *(_gsf_round(v * 1.0e6) for v in more_info))
    body += b'\x00' * 32  # spare
    return body


#: One KMALL_SPECIFIC transmit-sector entry as stored on the wire (gsf_dec.c's
#: DecodeKMALLSpecific(); 53 bytes, including 8 spare).
_KMALL_TX_SECTOR_DTYPE = np.dtype([
    ('TxSectorNumb', 'u1'), ('TxArrNumber', 'u1'), ('TxSubArray', 'u1'),
    ('SectorTransmitDelay_sec', '>i4'), ('TiltAngleReTx_deg', '>i4'),
    ('TxNominalSourceLevel_dB', '>i4'), ('TxFocusRange_m', '>i4'),
    ('CentreFreq_Hz', '>i4'), ('SignalBandWidth_Hz', '>i4'),
    ('TotalSignalLength_sec', '>i4'), ('PulseShading', 'u1'), ('SignalWaveForm', 'u1'),
    ('HighVoltageLevel_dB', '>i4'), ('SectorTrackingCorr_dB', '>i4'),
    ('EffectiveSignalLength_sec', '>i4'), ('spare', 'V8'),
])

#: Divisor that turns each KMALL transmit-sector wire integer into its
#: physical value; None keeps the raw integer.
_KMALL_TX_SECTOR_SCALE = {
    'TxSectorNumb': None, 'TxArrNumber': None, 'TxSubArray': None,
    'SectorTransmitDelay_sec': 1.0e6, 'TiltAngleReTx_deg': 1.0e6,
    'TxNominalSourceLevel_dB': 1.0e6, 'TxFocusRange_m': 1.0e3,
    'CentreFreq_Hz': 1.0e3, 'SignalBandWidth_Hz': 1.0e3,
    'TotalSignalLength_sec': 1.0e6, 'PulseShading': None, 'SignalWaveForm': None,
    'HighVoltageLevel_dB': 1.0e6, 'SectorTrackingCorr_dB': 1.0e6,
    'EffectiveSignalLength_sec': 1.0e6,
}

#: One KMALL_SPECIFIC extra-detection-class entry as stored on the wire
#: (35 bytes, including 32 spare).
_KMALL_EXTRA_CLASS_DTYPE = np.dtype([
    ('NumExtraDetInClass', '>u2'), ('AlarmFlag', 'u1'), ('spare', 'V32'),
])

_KMALL_EXTRA_CLASS_SCALE = {'NumExtraDetInClass': None, 'AlarmFlag': None}


def _scale_table(raw, scales):
    """
    Turn a structured numpy array of wire integers (one element per table
    row, typically a zero-copy np.frombuffer() view of the payload) into
    a table: a dictionary of column name -> numpy array. Each column in
    scales is divided by its divisor in one vectorized operation, giving
    float64 values; a column whose divisor is None is kept as integers,
    copied out to a native int64 array so the result is writable and
    independent of the payload.

    :param raw: a structured numpy array of the wire fields.
    :param scales: a dictionary of column name -> divisor or None, in
        the column order the table should have. Spare fields are simply
        left out of it.

    :return: a dictionary of column name -> numpy array.
    """
    return {name: raw[name] / divisor if divisor is not None else raw[name].astype(np.int64)
            for name, divisor in scales.items()}


def _decode_kmall_specific(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC subrecord (id
    156), which holds the per-ping metadata reported by a Kongsberg
    SIS 5 / .kmall-derived multibeam sonar, carried over from the
    originating #MRZ datagram's header, cmnPart, and pingInfo
    structures, along with its per-transmit-sector and
    per-extra-detection-class arrays. This is ported from the reference
    gsflib C library's DecodeKMALLSpecific() function in gsf_dec.c. It
    is registered in _PING_SENSOR_SPECIFIC_CODECS under id 156,
    following the same generic decode_fn(payload, pos) -> (record,
    bytes_consumed) contract every other family follows, and is paired
    with the separate _encode_kmall_specific() function.

    The reference decoder contains a quirk that this function
    reproduces in order to stay byte-aligned with the rest of the
    subrecord: gsf_dec.c reads a port/starboard mean-coverage-in-degrees
    pair from the wire, then, a few lines later, immediately reads a
    second port/starboard pair and overwrites the first with it -- an
    apparent copy/paste artifact in the reference decoder. The first
    pair's bytes are consumed from the stream but its decoded values
    are discarded; only the second pair's values end up in the decoded
    result. This function consumes and discards that first pair the
    same way, solely to keep byte alignment correct for everything that
    follows.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (record, bytes_consumed). record is a
        dictionary of the decoded scalar fields, plus two optional
        keys: 'TxSectors' and 'ExtraDetectionClasses', each a table (a
        dictionary of column name -> numpy array, one element per
        decoded transmit-sector or extra-detection-class entry
        respectively), present only when the ping reported at least one
        row of that kind. Each table is unpacked from the wire in a
        single step, as a numpy view of all of its rows at once. bytes_consumed is
        the number of bytes read from payload.
    """
    start = pos
    s = new_kmall_specific()

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

    num_sectors = min(num_tx_sectors, 9)  # gsf.h: GSF_MAX_KMALL_SECTORS
    sectors = np.frombuffer(payload, dtype=_KMALL_TX_SECTOR_DTYPE, count=num_sectors, offset=pos)
    pos += num_sectors * _KMALL_TX_SECTOR_DTYPE.itemsize

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

    num_classes = min(num_extra_classes, 11)  # gsf.h: GSF_MAX_KMALL_EXTRA_CLASSES
    classes = np.frombuffer(payload, dtype=_KMALL_EXTRA_CLASS_DTYPE, count=num_classes, offset=pos)
    pos += num_classes * _KMALL_EXTRA_CLASS_DTYPE.itemsize

    pos += 32  # spare5

    if num_sectors:
        s['TxSectors'] = _scale_table(sectors, _KMALL_TX_SECTOR_SCALE)
    if num_classes:
        s['ExtraDetectionClasses'] = _scale_table(classes, _KMALL_EXTRA_CLASS_SCALE)
    return s, pos - start


def new_intensity_time_series_header(sensor_id=None):
    """
    Return a new dictionary with the fixed 'BitsPerSample' and
    'AppliedCorrections' fields that begin every
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (id 21),
    merged with the sensor-imagery preamble template for sensor_id, if
    that vendor places a preamble in this subrecord.
    _decode_brb_intensity() builds its result starting from this same
    template, so the two can never define a different set of field
    names.

    :param sensor_id: the vendor "_SPECIFIC" subrecord id whose
        sensor-imagery preamble fields, if any, should be merged into
        the returned dictionary. Left as None, or naming a sensor that
        gsflib gives no preamble to (or whose preamble carries no
        fields, such as KMALL or the Reson 8100 family), the returned
        dictionary holds only the two fixed fields.

    :return: a dictionary with 'BitsPerSample' (pre-set to 16) and
        'AppliedCorrections' (pre-set to 0), plus that sensor's own
        imagery preamble fields, if any.
    """
    header = {'BitsPerSample': 16, 'AppliedCorrections': 0}
    if sensor_id in _SUBRECORD_EM3_IMAGERY_IDS:
        header.update(new_em3_imagery_specific())
    elif sensor_id in _SUBRECORD_EM4_IMAGERY_IDS:
        header.update(new_em4_imagery_specific())
    elif sensor_id in _SUBRECORD_RESON_SIZE_SPARE_IMAGERY_IDS:
        header.update(new_reson_size_spare_imagery_specific())
    elif sensor_id == _SUBRECORD_KLEIN_5410_BSS_SPECIFIC:
        header.update(new_klein5410bss_imagery_specific())
    elif sensor_id in _SUBRECORD_R2SONIC_IMAGERY_IDS:
        header.update(new_r2sonic_imagery_specific())
    # KMALL and the Reson 8100 family have a preamble with no fields;
    # every other sensor_id has no preamble at all.
    return header


def new_intensity_time_series_beam():
    """
    Return a new dictionary with every per-beam field name used within a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord's 'Beams'
    table, pre-set to 0: one row of that table. _decode_brb_intensity()
    returns exactly these columns, so the two can never define a different
    set of field names. 'DetectRangeSample' is the range of the beam's
    bottom detection, counted in samples, in the same units as the
    'Samples' table's 'RangeSample'. Each beam's samples are not part of
    its row. They are rows of the subrecord's 'Samples' table, which
    records the beam each sample belongs to.

    :return: a dictionary with 'DetectRangeSample' present, pre-set to 0.
    """
    return {'DetectRangeSample': 0}


def new_intensity_time_series_sample():
    """
    Return a new dictionary with every per-sample field name used within a
    GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord's 'Samples'
    table, each pre-set to 0: one row of that table.
    _decode_brb_intensity() returns exactly these columns, so the two can
    never define a different set of field names. 'Beam' is the number of
    the beam the sample belongs to. 'RangeSample' is the sample's range,
    counted in samples. 'Value' is the sample itself.

    :return: a dictionary with 'Beam', 'RangeSample', and 'Value' present,
        each pre-set to 0.
    """
    return {'Beam': 0, 'RangeSample': 0, 'Value': 0}


#: The first 6 bytes of each beam's 12-byte header in a
#: GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord: sample count,
#: bottom-detect sample index, and start-range sample index.
_INTENSITY_BEAM_HEADER = struct.Struct('>3H')

#: Just the sample count at the start of each beam's 12-byte header.
_INTENSITY_SAMPLE_COUNT = struct.Struct('>H')

#: Native unsigned sample dtype for each supported intensity sample width,
#: in bytes. As in gsf_dec.c, any bits-per-sample value other than 12 is
#: read as bits_per_sample // 8 whole bytes per sample.
_INTENSITY_SAMPLE_DTYPE = {1: np.uint8, 2: np.uint16, 4: np.uint32}


def _gather_intensity_samples(payload, sample_starts, sample_counts, bits_per_sample):
    """
    Gather every beam's intensity samples out of a ping payload into one
    flat array, with vectorized indexing rather than a per-beam loop.
    Beam i's samples are the sample_counts[i] consecutive entries that
    follow those of beams 0..i-1.

    :param payload: the raw bytes of the ping record.
    :param sample_starts: a numpy int64 array of the byte offset at which
        each beam's samples begin within payload.
    :param sample_counts: a numpy int64 array of each beam's number of
        samples.
    :param bits_per_sample: 12, or a width whose bits_per_sample // 8 is
        a key of _INTENSITY_SAMPLE_DTYPE.

    :return: a flat numpy array of every beam's samples, in beam order:
        uint16 for 12-bit samples, otherwise the native unsigned integer
        type of the sample width.
    """
    byte_counts = (sample_counts + 1) // 2 * 3 if bits_per_sample == 12 \
        else sample_counts * (bits_per_sample // 8)
    # Source byte index of every sample byte: each beam's run of bytes,
    # shifted from its position in the output to its position in payload.
    total_bytes = int(byte_counts.sum())
    first_out = np.cumsum(byte_counts) - byte_counts
    index = np.repeat(sample_starts - first_out, byte_counts)
    index += np.arange(total_bytes)
    raw = np.frombuffer(payload, dtype=np.uint8)[index]

    if bits_per_sample != 12:
        native = _INTENSITY_SAMPLE_DTYPE[bits_per_sample // 8]
        return raw.view(np.dtype(native).newbyteorder('>')).astype(native)

    # Two 12-bit samples per 3 bytes: (b0 b1_hi) and (b1_lo b2), as in
    # gsf_dec.c's DecodeBRBIntensity().
    triples = raw.reshape(-1, 3).astype(np.uint16)
    pairs = np.empty((len(triples), 2), dtype=np.uint16)
    pairs[:, 0] = (triples[:, 0] << 4) | (triples[:, 1] >> 4)
    pairs[:, 1] = ((triples[:, 1] & 0x0F) << 8) | triples[:, 2]
    samples = pairs.reshape(-1)
    # A beam with an odd sample count carries one padding sample at the
    # end of its last pair; drop it.
    odd = (sample_counts % 2) == 1
    if odd.any():
        pair_counts = (sample_counts + 1) // 2
        padding = (np.cumsum(pair_counts) * 2 - 1)[odd]
        samples = np.delete(samples, padding)
    return samples


def _intensity_beam_layout(intensity_record):
    """
    Derive, for each beam of an intensity time series record, the three
    values the file stores in that beam's header: its sample count, the
    range of its first sample, and the position of its bottom detection
    counted from that first sample. The record holds these implicitly: a
    beam's count is how many 'Samples' rows name it, its start range is
    the 'RangeSample' of its first sample, and its detect position is its
    'DetectRangeSample' minus that start range. A beam with no samples has
    no first sample, so its start range is taken to be 0.

    :param intensity_record: a dictionary in the shape
        _decode_brb_intensity() returns: a 'Beams' table with a
        'DetectRangeSample' column, and a 'Samples' table with 'Beam',
        'RangeSample', and 'Value' columns, its rows in beam order.

    :return: a tuple of (sample_counts, start_ranges, detect_samples),
        each a numpy int64 array with one element per beam.

    :raises ValueError: the 'Samples' rows are not in beam order, name a
        beam the 'Beams' table does not have, or a beam's 'RangeSample'
        values are not consecutive, starting from its first sample; or a
        derived value does not fit its unsigned 16-bit field.
    """
    beams = intensity_record.get('Beams')
    num_beams = _table_length(beams)
    detect_ranges = (np.asarray(beams['DetectRangeSample'], dtype=np.int64) if num_beams
                     else np.zeros(0, dtype=np.int64))
    sample_table = intensity_record.get('Samples')
    if _table_length(sample_table):
        beam = np.asarray(sample_table['Beam'], dtype=np.int64)
        range_sample = np.asarray(sample_table['RangeSample'], dtype=np.int64)
    else:
        beam = range_sample = np.zeros(0, dtype=np.int64)

    if len(beam) and (np.any(np.diff(beam) < 0) or beam[0] < 0 or beam[-1] >= num_beams):
        raise ValueError("IntensityTimeSeries: Samples['Beam'] must be in beam order, "
                         "and each beam must have a row in Beams")
    sample_counts = np.bincount(beam, minlength=num_beams).astype(np.int64)
    first = np.cumsum(sample_counts) - sample_counts
    start_ranges = np.zeros(num_beams, dtype=np.int64)
    has_samples = sample_counts > 0
    start_ranges[has_samples] = range_sample[first[has_samples]]
    position = np.arange(len(beam)) - first[beam]
    if np.any(range_sample != start_ranges[beam] + position):
        raise ValueError("IntensityTimeSeries: each beam's Samples['RangeSample'] values must be "
                         "consecutive, one per sample, since that is how the file stores them")
    detect_samples = detect_ranges - start_ranges
    for name, values in (('sample count', sample_counts), ('start range', start_ranges),
                         ('detect position', detect_samples)):
        if len(values) and (values.min() < 0 or values.max() > 0xFFFF):
            raise ValueError("IntensityTimeSeries: a beam's %s does not fit its 16-bit field" % name)
    return sample_counts, start_ranges, detect_samples


def _split_intensity_samples(intensity_record):
    """
    Split an intensity time series record's sample values into one array
    per beam. The 'Samples' table's rows are in beam order, and each
    returned array is a view into its 'Value' column, not a copy.

    :param intensity_record: a dictionary in the shape
        _decode_brb_intensity() returns.

    :return: a list of numpy arrays, one per beam, in beam order.
    """
    counts, _starts, _detects = _intensity_beam_layout(intensity_record)
    return np.split(np.asarray(intensity_record['Samples']['Value']), np.cumsum(counts)[:-1])


def _decode_brb_intensity(payload, pos, num_beams, sensor_id):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord
    (id 21), the per-beam receive-beam backscatter time series that can
    accompany a swath bathymetry ping. This is ported from the reference
    gsflib C library's DecodeBRBIntensity() function in gsf_dec.c.

    The fixed header and the per-beam sample loop that follows it are the
    same for every sensor, but gsflib inserts an optional sensor-specific
    "imagery" preamble between them, whose size and layout depend on
    sensor_id. Every vendor format that gsf_dec.c special-cases (KMALL,
    the EM3 series, the EM4 series, the Reson 7125/T-series/8100 family,
    the Klein 5410 BSS, and R2Sonic) is handled here by dispatching to the
    matching _decode_*_imagery_specific() helper. Any other sensor_id,
    including sensors gsflib does not special-case here (e.g. SeaBat,
    SeaBeam, the EM12/100/950/1000/121 family, GeoSwath, DeltaT), has no
    preamble at all, matching gsf_dec.c's switch default of a zero-length
    sensor block, and decoding proceeds straight to the per-beam samples.
    The beam headers are walked in order, since each one's sample count
    fixes where the next begins, but every beam's samples are then
    gathered from the payload in one vectorized step. The result starts
    from new_intensity_time_series_header()'s template, and its 'Beams'
    and 'Samples' columns are those of new_intensity_time_series_beam()
    and new_intensity_time_series_sample(), so its field names always
    match what _encode_brb_intensity() expects.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.
    :param num_beams: the number of beams in this ping, and therefore the
        number of per-beam sample blocks to decode.
    :param sensor_id: the vendor "_SPECIFIC" subrecord id most recently
        seen for this ping, used to decide which sensor-imagery preamble
        format, if any, precedes the per-beam samples.

    :return: None if num_beams is not positive, or if the encoded
        bits-per-sample value does not resolve to a supported sample
        width. Otherwise, a tuple of (record, bytes_consumed). record is
        a dictionary with 'BitsPerSample', 'AppliedCorrections', any
        fields decoded from a sensor-imagery preamble, 'Beams', a table
        (a dictionary of column name -> numpy array, one element per beam)
        with one column, 'DetectRangeSample', the range of each beam's
        bottom detection counted in samples, and 'Samples', a table with
        one row per sample, in beam order. The 'Samples' table's 'Beam'
        column is the number of the beam each sample belongs to. Its
        'RangeSample' column is the sample's range, counted in samples.
        Its 'Value' column is the sample itself. The file's per-beam
        sample count, start range, and detect position are all implicit in
        these columns (see _intensity_beam_layout()). The per-sample
        columns let a beam-wise, sector-wise, or range-dependent correction
        be applied to every sample in one vectorized operation. bytes_consumed is the number of bytes read
        from payload.
    """
    if num_beams <= 0:
        return None

    start = pos
    header = new_intensity_time_series_header(sensor_id)
    header['BitsPerSample'] = payload[pos]; pos += 1
    (header['AppliedCorrections'],) = struct.unpack_from('>I', payload, pos); pos += 4
    pos += 16  # spare

    bits_per_sample = header['BitsPerSample']
    if sensor_id == _SUBRECORD_KMALL_SPECIFIC:
        pos += _decode_kmall_imagery_specific(payload, pos)
    elif sensor_id in _SUBRECORD_EM3_IMAGERY_IDS:
        sensor_fields, consumed = _decode_em3_imagery_specific(payload, pos)
        header.update(sensor_fields); pos += consumed
    elif sensor_id in _SUBRECORD_EM4_IMAGERY_IDS:
        sensor_fields, consumed = _decode_em4_imagery_specific(payload, pos)
        header.update(sensor_fields); pos += consumed
    elif sensor_id in _SUBRECORD_RESON_SIZE_SPARE_IMAGERY_IDS:
        sensor_fields, consumed = _decode_reson_size_spare_imagery_specific(payload, pos)
        header.update(sensor_fields); pos += consumed
    elif sensor_id in _SUBRECORD_RESON_8100_IMAGERY_IDS:
        _sensor_fields, consumed = _decode_reson8100_imagery_specific(payload, pos)
        pos += consumed
    elif sensor_id == _SUBRECORD_KLEIN_5410_BSS_SPECIFIC:
        sensor_fields, consumed = _decode_klein5410bss_imagery_specific(payload, pos)
        header.update(sensor_fields); pos += consumed
    elif sensor_id in _SUBRECORD_R2SONIC_IMAGERY_IDS:
        sensor_fields, consumed = _decode_r2sonic_imagery_specific(payload, pos)
        header.update(sensor_fields); pos += consumed
    # else: no sensor-imagery preamble precedes the per-beam samples for
    # this sensor_id (gsf_dec.c's switch default, sensor_size=0).

    if bits_per_sample != 12 and bits_per_sample // 8 not in _INTENSITY_SAMPLE_DTYPE:
        return None

    # Each beam's fixed 12-byte header begins with its sample count, which
    # fixes where the next beam's header starts, so the beams are located
    # one at a time, reading only that count. Every beam's full header and
    # samples are then gathered from the payload at once.
    unpack_count = _INTENSITY_SAMPLE_COUNT.unpack_from
    sample_starts = []
    if bits_per_sample == 12:
        for _beam in range(num_beams):
            (count,) = unpack_count(payload, pos)
            pos += 12  # 3 x u16, plus 6 spare bytes
            sample_starts.append(pos)
            pos += (count + 1) // 2 * 3
    else:
        bytes_per_sample = bits_per_sample // 8
        for _beam in range(num_beams):
            (count,) = unpack_count(payload, pos)
            pos += 12  # 3 x u16, plus 6 spare bytes
            sample_starts.append(pos)
            pos += count * bytes_per_sample
    sample_starts = np.array(sample_starts, dtype=np.int64)
    header_bytes = np.frombuffer(payload, dtype=np.uint8)[(sample_starts - 12)[:, None] + np.arange(6)]
    headers = header_bytes.view('>u2').astype(np.int64)
    sample_counts = headers[:, 0]

    start_ranges = headers[:, 2]
    record = header
    beams = new_intensity_time_series_beam()
    beams['DetectRangeSample'] = start_ranges + headers[:, 1]
    record['Beams'] = beams

    # Each sample's beam number, and its range: its beam's start range plus
    # its position within the beam.
    total = int(sample_counts.sum())
    beam = np.repeat(np.arange(num_beams), sample_counts)
    position = np.arange(total) - np.repeat(np.cumsum(sample_counts) - sample_counts, sample_counts)
    samples = new_intensity_time_series_sample()
    samples['Beam'] = beam
    samples['RangeSample'] = start_ranges[beam] + position
    samples['Value'] = _gather_intensity_samples(payload, sample_starts, sample_counts, bits_per_sample)
    record['Samples'] = samples
    return record, pos - start


def _pack_intensity_samples(samples, sample_counts, bits_per_sample):
    """
    Pack a flat array of every beam's intensity samples into their wire
    bytes in one vectorized step: the inverse of
    _gather_intensity_samples(). 12-bit samples are packed two into
    every three bytes using gsf_enc.c's EncodeBRBIntensity() bit layout,
    with a beam whose sample count is odd padded out to a whole pair with
    a zero sample; every other width is written as big-endian unsigned
    integers.

    :param samples: a flat numpy array of every beam's samples, in beam
        order.
    :param sample_counts: a numpy int64 array of each beam's number of
        samples; these must total len(samples).
    :param bits_per_sample: 8, 12, 16, or 32.

    :return: a tuple of (wire, beam_bytes). wire is every beam's packed
        samples, as bytes, in beam order. beam_bytes is a numpy int64
        array of how many of those bytes belong to each beam.
    """
    if bits_per_sample != 12:
        beam_bytes = sample_counts * (bits_per_sample // 8)
        wire_dtype = np.dtype(_INTENSITY_SAMPLE_DTYPE[bits_per_sample // 8]).newbyteorder('>')
        return samples.astype(wire_dtype).tobytes(), beam_bytes

    beam_bytes = (sample_counts + 1) // 2 * 3
    values = samples.astype(np.uint16) & 0x0FFF
    odd = (sample_counts % 2) == 1
    if odd.any():
        values = np.insert(values, np.cumsum(sample_counts)[odd], 0)
    pairs = values.reshape(-1, 2)
    packed = np.empty((len(pairs), 3), dtype=np.uint8)
    packed[:, 0] = pairs[:, 0] >> 4
    packed[:, 1] = ((pairs[:, 0] & 0x0F) << 4) | (pairs[:, 1] >> 8)
    packed[:, 2] = pairs[:, 1] & 0xFF
    return packed.tobytes(), beam_bytes


def _encode_brb_intensity(record, sensor_id):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord
    (id 21), the per-beam receive-beam backscatter time series that can
    accompany a swath bathymetry ping. This is the inverse of
    _decode_brb_intensity(), and is ported from the reference gsflib C
    library's EncodeBRBIntensity() function in gsf_enc.c.

    The fixed header and the per-beam sample loop that follows it are the
    same for every sensor, but gsflib inserts an optional sensor-specific
    "imagery" preamble between them, whose size and layout depend on
    sensor_id, dispatched here exactly as _decode_brb_intensity() does.
    Every beam's samples are packed to wire bytes in one vectorized step
    (see _pack_intensity_samples()): 12-bit samples two-into-three-bytes
    using gsflib's own bit layout, and 8-, 16-, and 32-bit samples as
    plain big-endian values.

    :param record: a dictionary in the same shape _decode_brb_intensity()
        returns: the header fields new_intensity_time_series_header()
        creates, plus a 'Beams' table (a dictionary of column name ->
        array-like, or a pandas.DataFrame) with a 'DetectRangeSample'
        column, one row per beam, and a 'Samples' table with 'Beam',
        'RangeSample', and 'Value' columns, one row per sample, in beam
        order. Each beam's sample count, start range, and detect position
        are derived from these (see _intensity_beam_layout()); a beam with
        no samples is written with a start range of 0. A header field
        that is missing from the dictionary is written as zero.
    :param sensor_id: the vendor "_SPECIFIC" subrecord id for this ping,
        used to decide which sensor-imagery preamble format, if any,
        precedes the per-beam samples. This should be the same value
        used to decode or construct record, so the preamble that is
        written matches the fields record actually carries.

    :return: the encoded subrecord, as bytes, including its own 4-byte
        identifier word. As in gsf_enc.c, and unlike every other ping
        subrecord, that word's size field counts the identifier word
        itself, not just the bytes that follow it.

    :raises ValueError: if record's 'BitsPerSample' is not 8, 12, 16, or
        32, matching the reference encoder's GSF_MB_PING_RECORD_ENCODE_FAILED
        error for the same condition; or if the 'Samples' rows cannot be
        written as the file's per-beam runs (see
        _intensity_beam_layout()).
    """
    bits_per_sample = int(record.get('BitsPerSample', 16))
    if bits_per_sample not in (8, 12, 16, 32):
        raise ValueError("unsupported BitsPerSample for IntensityTimeSeries: %r" % (bits_per_sample,))

    body = struct.pack('>B', bits_per_sample)
    body += struct.pack('>I', int(record.get('AppliedCorrections', 0)))
    body += b'\x00' * 16  # spare

    if sensor_id == _SUBRECORD_KMALL_SPECIFIC:
        body += _encode_kmall_imagery_specific()
    elif sensor_id in _SUBRECORD_EM3_IMAGERY_IDS:
        body += _encode_em3_imagery_specific(record)
    elif sensor_id in _SUBRECORD_EM4_IMAGERY_IDS:
        body += _encode_em4_imagery_specific(record)
    elif sensor_id in _SUBRECORD_RESON_SIZE_SPARE_IMAGERY_IDS:
        body += _encode_reson_size_spare_imagery_specific(record)
    elif sensor_id in _SUBRECORD_RESON_8100_IMAGERY_IDS:
        body += _encode_reson8100_imagery_specific()
    elif sensor_id == _SUBRECORD_KLEIN_5410_BSS_SPECIFIC:
        body += _encode_klein5410bss_imagery_specific(record)
    elif sensor_id in _SUBRECORD_R2SONIC_IMAGERY_IDS:
        body += _encode_r2sonic_imagery_specific(record)
    # else: no sensor-imagery preamble precedes the per-beam samples for
    # this sensor_id (gsf_enc.c's switch default, sensor_size=0).

    sample_counts, start_ranges, detect_samples = _intensity_beam_layout(record)
    sample_table = record.get('Samples')
    samples = np.asarray(sample_table['Value'] if _table_length(sample_table) else ())

    wire, beam_bytes = _pack_intensity_samples(samples, sample_counts, bits_per_sample)
    beam_ends = np.cumsum(beam_bytes).tolist()
    parts = [body]
    beam_start = 0
    for count, detect, start_range, beam_end in zip(
            sample_counts.tolist(), detect_samples.tolist(), start_ranges.tolist(), beam_ends):
        parts.append(_INTENSITY_BEAM_HEADER.pack(count, detect, start_range))
        parts.append(b'\x00' * 6)  # spare
        parts.append(wire[beam_start:beam_end])
        beam_start = beam_end
    body = b''.join(parts)

    # Unlike every other ping subrecord, gsf_enc.c's EncodeBRBIntensity()
    # counts its own 4-byte identifier word in this subrecord's size field.
    header_word = ((_SUBRECORD_INTENSITY_SERIES_ARRAY & 0xFF) << 24) | (len(body) + 4)
    return struct.pack('>I', header_word) + body


def _gsf_timestamp(sec, nsec):
    """
    Combine a GSF-style timestamp, expressed as a count of whole seconds
    since the Unix epoch plus a nanosecond remainder within that second,
    into a single timezone-aware Python datetime in UTC.

    :param sec: whole seconds since the Unix epoch.
    :param nsec: the nanosecond remainder within that second.

    :return: a datetime.datetime in UTC.
    """
    return datetime.datetime.fromtimestamp(sec + nsec / 1.0e9, tz=datetime.timezone.utc)


def _decode_scale_factors(payload, pos):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS subrecord, the table
    of per-beam-array (multiplier, offset, compression flag) triples that
    lets a ping's beam arrays be stored on disk as scaled integers rather
    than as raw floating-point values. This is ported from the reference
    gsflib C library's DecodeScaleFactors() function in gsf_dec.c.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.

    :return: a tuple of (scale_table, bytes_consumed). scale_table is a
        dictionary mapping each beam-array subrecord id present in the
        table to a (multiplier, offset, compressionFlag) tuple.
        bytes_consumed is the number of bytes read from payload.
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
    Decode one scaled per-beam array subrecord (such as depth, across
    track, or beam angle) from its raw on-disk bytes into engineering
    units, using numpy to decode every beam's value at once rather than
    looping over beams in Python. Each raw stored integer is converted
    with value = raw_int / multiplier - offset. This mirrors the primary,
    uncompressed decode path shared by the per-width, per-signedness
    family of Decode*Array functions in the reference gsflib C library's
    gsf_dec.c.

    The on-disk width of each beam's value (1, 2, or 4 bytes) is not
    passed in directly; it is inferred from size / num_beams, the same
    way the reference decoder determines field width for this class of
    subrecord.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.
    :param size: the subrecord's declared size in bytes, used together
        with num_beams to infer how many bytes each beam's value
        occupies.
    :param num_beams: the number of beams in this ping, and therefore the
        number of values to decode.
    :param multiplier: this array's scale-factor multiplier, used to
        convert each raw stored integer back to engineering units.
    :param offset: this array's scale-factor offset, added back in (after
        dividing by multiplier) to recover the true value.
    :param signed: True if the raw stored integers are signed, False if
        they are unsigned.
    :param truncate_to_int: if True, the decoded values are truncated
        toward zero to integers after the scale factor is applied,
        matching fields that are conceptually integer-valued (such as a
        count or a flag) rather than a continuous physical measurement.

    :return: None if the per-beam width cannot be determined (size is not
        an exact multiple of num_beams, or the resulting width is not 1,
        2, or 4 bytes). Otherwise, a numpy array of length num_beams
        holding the decoded values, as floats, or as integers truncated
        toward zero if truncate_to_int is True.
    """
    if num_beams <= 0 or size % num_beams != 0:
        return None
    bytes_per_value = size // num_beams
    dtype = _ARRAY_DTYPE.get((bytes_per_value, signed))
    if dtype is None:
        return None

    # Dividing the zero-copy view of the wire integers yields the one new
    # float64 array needed; the offset is then removed in place.
    values = np.frombuffer(payload, dtype=dtype, count=num_beams, offset=pos) / multiplier
    values -= offset
    if truncate_to_int:
        values = np.trunc(values).astype(np.int64)
    return values


def _decode_quality_flags_array(payload, pos, num_beams, subrecord_size):
    """
    Decode a GSF_SWATH_BATHY_SUBRECORD_QUALITY_FLAGS_ARRAY subrecord: one
    2-bit quality flag per beam, with four beams packed into each byte
    (bits 7-6 hold the first beam in the byte, down to bits 1-0 for the
    fourth, most significant bits first). This is ported from the
    reference gsflib C library's DecodeQualityFlagsArray() function in
    gsf_dec.c, and is vectorized with numpy rather than decoding one beam
    at a time.

    If subrecord_size is too small to cover every beam, the reference
    decoder only reads sr_size * 4 beams' worth of flags and leaves the
    remaining beams at their pre-allocated value of zero; this function
    reproduces that behavior by returning zero for any beam beyond what
    subrecord_size actually covers.

    :param payload: the raw bytes of the ping record.
    :param pos: the byte offset within payload where this subrecord's
        data begins.
    :param num_beams: the number of beams in this ping, and therefore the
        length of the returned array.
    :param subrecord_size: the subrecord's declared size in bytes, which
        determines how many beams actually have an encoded flag.

    :return: a numpy uint8 array of length num_beams, with each beam's
        value in the range 0-3.
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
    scalar header fields, followed by the ping's variable subrecord
    stream (the scale-factor table, the per-beam arrays, the optional
    backscatter intensity time series, and at most one vendor
    sensor-specific subrecord). This is ported from the reference gsflib
    C library's gsfDecodeSwathBathymetryPing() function in gsf_dec.c, and
    is the most important decoder in this file, since the swath
    bathymetry ping is the record type that actually carries sounding
    data.

    :param payload: the raw bytes of the ping record.
    :param major_version: the major version number of the GSF file this
        ping came from (for example, 3 for a "GSF-v03.xx" file), as read
        from that file's own GSF_RECORD_HEADER record. It controls
        whether the Height_m, SEP_m, and GPSTideCorrector_m fields are
        present in the fixed header: gsflib only started writing those
        three fields starting with major version 3, so this function only
        attempts to decode them when major_version is greater than 2.
    :param scale_factors: a dictionary mapping beam-array subrecord id to
        (multiplier, offset, compressionFlag), passed in by the caller
        holding whatever scale factors this file last decoded. This
        function mutates it in place whenever the ping carries its own
        GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS subrecord, replacing its
        contents with the newly decoded table. Pings after the first one
        do not always repeat their scale factors, so the caller is
        expected to keep passing in the same dictionary across successive
        calls so that later pings can keep reusing it.
    :param decode_intensity: if True, fully decode the per-beam
        GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (the
        backscatter time series) into record['IntensityTimeSeries'].
        That subrecord can hold tens of thousands of samples per ping, so
        decoding it is opt-in and left False by default; today it is only
        requested by gsf.print_intensity_series(), not by the default
        gsf.print_records() path.

    :return: record, a dictionary built from new_swath_bathymetry_ping()'s
        template. The fixed ping scalar fields (for example 'PingTime',
        'Longitude_deg', 'Latitude_deg', 'NumberBeams', 'CenterBeam',
        'PingFlags', and 'Heading_deg') are stored flat, unprefixed, at
        the top level; Height_m, SEP_m, and GPSTideCorrector_m, which only
        GSF major version 3 and later store, keep the template's "not
        available" values for an older file. 'Beams' is a table (a
        dictionary of column name -> numpy array, one element per beam, in
        beam order) with one column for every label in
        BEAM_ARRAY_SUBRECORD_IDS. When the ping carries a beam array
        subrecord, its column holds the decoded array; otherwise the
        column is None, so `record['Beams'][label] is None` tests whether
        the ping carried that subrecord. A zero-length subrecord that
        another subrecord follows decodes to an empty array; one that ends
        the ping is skipped, as gsf_dec.c skips it. A beam flags subrecord whose size is not one
        byte per beam decodes to an array of the bytes it holds, and a
        note records the mismatch. The one case in which a subrecord is
        present but its column stays None is a scaled array that cannot
        be decoded, because no scale factors are available for it or its
        encoding is unsupported; a note then says why.
        'IntensityTimeSeries' is
        present, as a dictionary in the shape _decode_brb_intensity()
        returns (with its own 'Beams' and 'Samples' tables),
        only when decode_intensity is True and the ping carried that
        subrecord. 'SensorSpecificID' (an int) and 'SensorSpecific' (a
        dictionary, which may itself contain tables in the same
        dictionary-of-arrays form) are present together whenever the ping
        carried a vendor sensor-specific subrecord that could be
        decoded -- at most one such subrecord can appear per ping, per
        gsf.h's union gsfSensorSpecific, and the corresponding vendor
        family name can be looked up on demand via
        _SENSOR_SPECIFIC_SUBRECORD_NAMES[SensorSpecificID]. 'Notes' is
        always present, as a list of strings describing anything in the
        subrecord stream that this function did not decode (for example
        an unsupported encoding, beam-array data with no matching scale
        factors available, or an unrecognized subrecord id).
    """
    record = new_swath_bathymetry_ping()
    notes = record['Notes']

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

    beam_columns = record['Beams']
    sensor_id = None
    sensor_specific_id = None
    sensor_specific_record = None

    # Like gsf_dec.c's gsfDecodeSwathBathymetryPing(), stop once four or
    # fewer bytes remain. A subrecord in those last four bytes could only be
    # a zero-length one, which gsflib never writes and skips when reading,
    # so this decoder sees the same subrecords gsflib does.
    while len(payload) - pos > 4:
        word, = struct.unpack_from('>I', payload, pos)
        subrecord_id = (word >> 24) & 0xFF
        subrecord_size = word & 0x00FFFFFF
        pos += 4

        if subrecord_id == _SUBRECORD_SCALE_FACTORS:
            # A scale factors subrecord too short for the table it declares
            # (gsflib never writes one, but other tools can) is noted and
            # skipped, leaving the scale factors carried from earlier pings
            # in effect.
            try:
                table, _consumed = _decode_scale_factors(payload[:pos + subrecord_size], pos)
            except struct.error:
                notes.append("ScaleFactors (%d bytes) not decoded: too short for its table"
                             % subrecord_size)
            else:
                scale_factors.clear()
                scale_factors.update(table)

        elif subrecord_id == _SUBRECORD_BEAM_FLAGS_ARRAY:
            # Beam flags are one byte per beam. Every byte the subrecord
            # actually holds is kept, even when that is not one per beam,
            # so the column records that the subrecord was present. The
            # array is copied out of the payload so that it is writable.
            count = min(subrecord_size, len(payload) - pos)
            beam_columns['BeamFlags'] = np.frombuffer(
                payload, dtype=np.uint8, count=count, offset=pos).copy()
            if subrecord_size and subrecord_size != number_beams:
                notes.append("BeamFlags (%d bytes) does not match NumberBeams (%d)"
                             % (subrecord_size, number_beams))

        elif subrecord_id == _SUBRECORD_QUALITY_FLAGS_ARRAY:
            if subrecord_size == 0:
                beam_columns['QualityFlags'] = np.empty(0, dtype=np.uint8)
            else:
                beam_columns['QualityFlags'] = _decode_quality_flags_array(
                    payload, pos, number_beams, subrecord_size)

        elif subrecord_id in _PING_ARRAY_SUBRECORDS:
            _attr, label, signed = _PING_ARRAY_SUBRECORDS[subrecord_id]
            sf = scale_factors.get(subrecord_id)
            if subrecord_size == 0:
                # A zero-length array subrecord is present but holds no
                # values, so it decodes to an empty column, whether or not
                # scale factors are available for it.
                dtype = np.int64 if subrecord_id in _PING_ARRAY_INTEGER_SUBRECORDS else np.float64
                beam_columns[label] = np.empty(0, dtype=dtype)
            elif sf is None:
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
            # Unlike every other ping subrecord, gsf_enc.c's
            # EncodeBRBIntensity() counts its own 4-byte identifier word in
            # this subrecord's size field (gsf_dec.c never reads that size,
            # advancing by the bytes DecodeBRBIntensity() consumed instead).
            # A malformed size smaller than that word is treated as an empty
            # subrecord, so the decoder always moves forward.
            subrecord_size = max(subrecord_size - 4, 0)
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
                        intensity_record, _consumed = decoded
                        record['IntensityTimeSeries'] = intensity_record

        elif subrecord_id in _PING_SENSOR_SPECIFIC_CODECS:
            sensor_id = subrecord_id
            _family_label, decode_fn, _encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
            try:
                sensor_record, _consumed = decode_fn(payload, pos)
            except (struct.error, IndexError) as exc:
                notes.append("%s (%d, %d bytes) not decoded: %s" %
                             (_SENSOR_SPECIFIC_SUBRECORD_NAMES.get(subrecord_id, str(subrecord_id)),
                              subrecord_id, subrecord_size, exc))
            else:
                sensor_specific_id = subrecord_id
                sensor_specific_record = sensor_record

        elif subrecord_id in _SENSOR_SPECIFIC_SUBRECORD_NAMES:
            sensor_id = subrecord_id
            notes.append("%s (%d, %d bytes) not decoded" %
                         (_SENSOR_SPECIFIC_SUBRECORD_NAMES[subrecord_id], subrecord_id, subrecord_size))

        else:
            notes.append("subrecord id %d (%d bytes) not decoded" % (subrecord_id, subrecord_size))

        pos += subrecord_size

    if sensor_specific_id is not None:
        record['SensorSpecificID'] = sensor_specific_id
        record['SensorSpecific'] = sensor_specific_record

    return record


def new_single_beam_ping():
    """
    Return a new dictionary with every GSF_RECORD_SINGLE_BEAM_PING field
    name present, in the same shape _decode_single_beam_ping() returns:
    every field in _REQUIRED_SINGLE_BEAM_PING_FIELDS set to None as a
    placeholder that the caller must overwrite (gsf.h defines no "not
    available" value for them), 'PositioningSystemType' pre-set to 0, and
    'Notes' an empty list. _decode_single_beam_ping() builds its result
    starting from this same template, adding 'SensorSpecificID' and
    'SensorSpecific' only when the ping carries a sensor-specific
    subrecord (see that family's new_*_specific() template).

    :return: a dictionary with every single-beam-ping field name as a
        key, pre-filled with a default value as described above.
    """
    record = {name: None for name in _REQUIRED_SINGLE_BEAM_PING_FIELDS}
    record['PositioningSystemType'] = 0
    record['Notes'] = []
    return record


def _decode_single_beam_ping(payload):
    """
    Decode a GSF_RECORD_SINGLE_BEAM_PING payload: the fixed-format scalar
    fields, plus its one optional sensor-specific tail subrecord, decoded
    via the _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS registry. This is ported
    from the reference gsflib C library's gsfDecodeSinglebeam() function
    in gsf_dec.c.

    Unlike a swath bathymetry ping's sensor-specific subrecord stream, a
    single-beam ping carries at most one such subrecord, so this function
    does not need a loop: it reads the one 4-byte id+size word at offset
    38, if the payload is long enough to contain one, and dispatches to
    the matching decoder a single time. The reference decoder's obscure
    fallback behavior, where it will extract a trailing subrecord id from
    a subrecord whose declared size is exactly 0, is not reproduced here,
    since that fallback only matters for a subrecord that carries an id
    but no payload, and none of the vendor families registered here are
    ever encoded that way.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_SINGLE_BEAM_PING record is
    available.

    :param payload: the raw bytes of the single-beam ping record.

    :return: record, a dictionary built from new_single_beam_ping()'s
        template. The fixed ping scalar fields (for
        example 'PingTime', 'Longitude_deg', 'Latitude_deg', and
        'Depth_m') are stored flat, unprefixed, at the top level.
        'SensorSpecificID' (an int) and 'SensorSpecific' (a dictionary,
        with any tables it contains as dictionaries of column name ->
        numpy array) are present
        together only when the ping carried a sensor-specific tail
        subrecord that this function was able to decode. 'Notes' is
        always present, as a list of strings describing anything that
        was not decoded.
    """
    record = new_single_beam_ping()
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
    notes = record['Notes']
    if remaining > 4:
        word, = struct.unpack_from('>I', payload, 38)
        subrecord_id = (word >> 24) & 0xFF
        subrecord_size = word & 0x00FFFFFF
        if subrecord_id in _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS:
            _family_label, decode_fn, _encode_fn = _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS[subrecord_id]
            try:
                sensor_record, _consumed = decode_fn(payload, 42)
            except (struct.error, IndexError) as exc:
                notes.append("%s (%d, %d bytes) not decoded: %s" %
                             (_SINGLE_BEAM_SENSOR_SPECIFIC_NAMES.get(subrecord_id, str(subrecord_id)),
                              subrecord_id, subrecord_size, exc))
            else:
                record['SensorSpecificID'] = subrecord_id
                record['SensorSpecific'] = sensor_record
        elif subrecord_id in _SINGLE_BEAM_SENSOR_SPECIFIC_NAMES:
            notes.append("%s (%d, %d bytes) not decoded" %
                         (_SINGLE_BEAM_SENSOR_SPECIFIC_NAMES[subrecord_id], subrecord_id, subrecord_size))
        else:
            notes.append("subrecord id %d (%d bytes) not decoded" % (subrecord_id, subrecord_size))

    return record


def new_swath_bathy_summary():
    """
    Return a new dictionary with every GSF_RECORD_SWATH_BATHY_SUMMARY
    field name present, in the same shape _decode_swath_bathy_summary()
    returns, each pre-set to None: every field is required to write this
    record, and gsf.h defines no "not available" value for any of them.
    _decode_swath_bathy_summary() builds its result starting from this
    same template, so the two can never define a different set of field
    names.

    :return: a dictionary with 'StartTime', 'EndTime', 'MinLatitude_deg',
        'MinLongitude_deg', 'MaxLatitude_deg', 'MaxLongitude_deg',
        'MinDepth_m', and 'MaxDepth_m', each pre-set to None.
    """
    return {
        'StartTime': None, 'EndTime': None,
        'MinLatitude_deg': None, 'MinLongitude_deg': None,
        'MaxLatitude_deg': None, 'MaxLongitude_deg': None,
        'MinDepth_m': None, 'MaxDepth_m': None,
    }


def _decode_swath_bathy_summary(payload):
    """
    Decode a GSF_RECORD_SWATH_BATHY_SUMMARY payload: the start and end
    times, the geographic bounding box, and the depth range covered by a
    contiguous run of swath bathymetry pings. This is ported from the
    reference gsflib C library's gsfDecodeSwathBathySummary() function in
    gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_SWATH_BATHY_SUMMARY record is
    available.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_swath_bathy_summary()
        creates, with 'StartTime' and 'EndTime' (ISO 8601 strings, in
        UTC), 'MinLatitude_deg', 'MinLongitude_deg', 'MaxLatitude_deg',
        'MaxLongitude_deg', 'MinDepth_m', and 'MaxDepth_m'.
    """
    record = new_swath_bathy_summary()
    (start_sec, start_nsec, end_sec, end_nsec) = struct.unpack_from('>4I', payload, 0)
    record['StartTime'] = _gsf_timestamp(start_sec, start_nsec).isoformat()
    record['EndTime'] = _gsf_timestamp(end_sec, end_nsec).isoformat()

    (min_lat_raw, min_lon_raw, max_lat_raw, max_lon_raw,
     min_depth_raw, max_depth_raw) = struct.unpack_from('>6i', payload, 16)
    record['MinLatitude_deg'] = min_lat_raw / 1.0e7
    record['MinLongitude_deg'] = min_lon_raw / 1.0e7
    record['MaxLatitude_deg'] = max_lat_raw / 1.0e7
    record['MaxLongitude_deg'] = max_lon_raw / 1.0e7
    record['MinDepth_m'] = min_depth_raw / 100.0
    record['MaxDepth_m'] = max_depth_raw / 100.0
    return record


#: One sound velocity profile point as stored on the wire (gsf_dec.c's
#: gsfDecodeSoundVelocityProfile()): depth and sound speed, each in
#: hundredths.
_SVP_POINT_DTYPE = np.dtype([('depth', '>u4'), ('sound_speed', '>u4')])


def new_sound_velocity_profile():
    """
    Return a new dictionary with every GSF_RECORD_SOUND_VELOCITY_PROFILE
    field name present, in the same shape _decode_sound_velocity_profile()
    returns: the times and position pre-set to None (they have no
    "not available" value, and every one is required to write a profile),
    'NumberPoints' pre-set to 0, and an empty numpy array for each
    per-point field. _decode_sound_velocity_profile() builds its result
    starting from this same template, so the two can never define a
    different set of field names.

    :return: a dictionary with 'ObservationTime', 'ApplicationTime',
        'Longitude_deg', 'Latitude_deg' (each None), 'NumberPoints' (0),
        and 'Depth_m' and 'SoundSpeed_mPerSec' (each an empty float64
        array).
    """
    return {
        'ObservationTime': None, 'ApplicationTime': None,
        'Longitude_deg': None, 'Latitude_deg': None,
        'NumberPoints': 0,
        'Depth_m': np.empty(0), 'SoundSpeed_mPerSec': np.empty(0),
    }


def _decode_sound_velocity_profile(payload):
    """
    Decode a GSF_RECORD_SOUND_VELOCITY_PROFILE payload: the observation
    and application times, the position where the profile was collected,
    and the depth/sound-speed pairs that make up the profile itself. This
    is ported from the reference gsflib C library's
    gsfDecodeSoundVelocityProfile() function in gsf_dec.c. Every point is
    unpacked at once, as a zero-copy numpy view of the payload, and each
    field scaled with one vectorized operation.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_sound_velocity_profile()
        creates, with 'ObservationTime' and 'ApplicationTime' (ISO 8601
        strings, in UTC), 'Longitude_deg', 'Latitude_deg', 'NumberPoints',
        and 'Depth_m' and 'SoundSpeed_mPerSec' (each a float64 numpy
        array, one value per profile point).
    """
    record = new_sound_velocity_profile()
    (obs_sec, obs_nsec, app_sec, app_nsec) = struct.unpack_from('>4I', payload, 0)
    record['ObservationTime'] = _gsf_timestamp(obs_sec, obs_nsec).isoformat()
    record['ApplicationTime'] = _gsf_timestamp(app_sec, app_nsec).isoformat()

    (lon_raw, lat_raw, number_points) = struct.unpack_from('>2iI', payload, 16)
    record['Longitude_deg'] = lon_raw / 1.0e7
    record['Latitude_deg'] = lat_raw / 1.0e7
    record['NumberPoints'] = number_points

    points = np.frombuffer(payload, dtype=_SVP_POINT_DTYPE, count=number_points, offset=28)
    record['Depth_m'] = points['depth'] / 100.0
    record['SoundSpeed_mPerSec'] = points['sound_speed'] / 100.0
    return record


def new_name_value_parameters():
    """
    Return a new dictionary in the shape _decode_name_value_parameters()
    returns for a GSF_RECORD_PROCESSING_PARAMETERS or
    GSF_RECORD_SENSOR_PARAMETERS record: 'ParamTime' pre-set to None (it is
    required to write the record), and 'Parameters', an empty dictionary
    of parameter name -> value to fill in. _decode_name_value_parameters()
    builds its result starting from this same template, so the two can
    never define a different set of field names.

    :return: a dictionary with 'ParamTime' (None) and 'Parameters' (an
        empty dictionary).
    """
    return {'ParamTime': None, 'Parameters': {}}


def _decode_name_value_parameters(payload):
    """
    Decode a GSF_RECORD_PROCESSING_PARAMETERS or GSF_RECORD_SENSOR_PARAMETERS
    payload; both record types share the same on-disk format, a time
    stamp followed by a list of counted strings, each already formatted
    as "NAME=VALUE". This is ported from the reference gsflib C library's
    gsfDecodeProcessingParameters() and gsfDecodeSensorParameters()
    functions in gsf_dec.c, which are identical apart from which struct
    field they write their result into. Because each parameter is already
    a "NAME=VALUE" string, decoding it is just a matter of splitting on
    the '=' character.

    In practice, encoders commonly include a trailing NUL byte inside a
    parameter's counted size, left over from treating the parameter as a
    C string with its terminator baked into the on-disk length; this
    function strips that trailing NUL rather than letting it show up as a
    literal '\\x00' character in the decoded value.

    This function has been verified against real GSF files for
    GSF_RECORD_PROCESSING_PARAMETERS. It has not been tested against a
    verified GSF file for GSF_RECORD_SENSOR_PARAMETERS, since no sample
    data containing that record type is available, even though it is
    decoded by this same function.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_name_value_parameters()
        creates, with 'ParamTime' (an ISO 8601 string, in UTC) and
        'Parameters', a dictionary with one entry per decoded
        "NAME=VALUE" string, keyed by NAME, in file order. A parameter
        string with no '=' character is keyed by the whole string, with a
        value of None, so that it re-encodes unchanged.
    """
    record = new_name_value_parameters()
    (sec, nsec, number_parameters) = struct.unpack_from('>2IH', payload, 0)
    record['ParamTime'] = _gsf_timestamp(sec, nsec).isoformat()

    parameters = record['Parameters']
    pos = 10
    for _ in range(number_parameters):
        (size,) = struct.unpack_from('>h', payload, pos)
        pos += 2
        text = payload[pos:pos + size].decode('ascii', 'replace').rstrip('\x00')
        pos += size
        name, sep, value = text.partition('=')
        parameters[name if sep else text] = value if sep else None
    return record


def new_comment():
    """
    Return a new dictionary with every GSF_RECORD_COMMENT field name
    present, in the same shape _decode_comment() returns: 'CommentTime'
    pre-set to None (it is required to write the record) and 'Comment'
    pre-set to an empty string. _decode_comment() builds its result
    starting from this same template.

    :return: a dictionary with 'CommentTime' (None) and 'Comment' ('').
    """
    return {'CommentTime': None, 'Comment': ''}


def _decode_comment(payload):
    """
    Decode a GSF_RECORD_COMMENT payload: a time stamp and a single
    free-text comment string. This is ported from the reference gsflib C
    library's gsfDecodeComment() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_COMMENT record is available.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_comment() creates, with
        'CommentTime' (an ISO 8601 string, in UTC) and 'Comment' (the
        decoded comment text).
    """
    record = new_comment()
    (sec, nsec, length) = struct.unpack_from('>3I', payload, 0)
    record['CommentTime'] = _gsf_timestamp(sec, nsec).isoformat()
    record['Comment'] = payload[12:12 + length].decode('ascii', 'replace')
    return record


def new_history():
    """
    Return a new dictionary with every GSF_RECORD_HISTORY field name
    present, in the same shape _decode_history() returns: 'HistoryTime'
    pre-set to None (it is required to write the record), and each text
    field pre-set to an empty string. _decode_history() builds its result
    starting from this same template.

    :return: a dictionary with 'HistoryTime' (None), and 'HostName',
        'OperatorName', 'CommandLine', and 'Comment' (each '').
    """
    return {'HistoryTime': None, 'HostName': '', 'OperatorName': '', 'CommandLine': '', 'Comment': ''}


def _decode_history(payload):
    """
    Decode a GSF_RECORD_HISTORY payload: a time stamp plus the host name,
    operator name, command line, and comment text describing one
    processing step recorded in the file's history. This is ported from
    the reference gsflib C library's gsfDecodeHistory() function in
    gsf_dec.c.

    gsf_enc.c writes the host name, operator name, and command line with
    a trailing NUL counted in each one's size, as C strings; that NUL is
    stripped here, so the decoded strings are plain text and a decoded
    record re-encodes (via _encode_history(), which adds it back) to the
    same bytes.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_HISTORY record is available.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_history() creates, with
        'HistoryTime' (an ISO 8601 string, in UTC), 'HostName',
        'OperatorName', 'CommandLine', and 'Comment'.
    """
    record = new_history()
    (sec, nsec) = struct.unpack_from('>2I', payload, 0)
    record['HistoryTime'] = _gsf_timestamp(sec, nsec).isoformat()

    pos = 8
    for key in ('HostName', 'OperatorName', 'CommandLine', 'Comment'):
        (length,) = struct.unpack_from('>H', payload, pos)
        pos += 2
        record[key] = payload[pos:pos + length].decode('ascii', 'replace').rstrip('\x00')
        pos += length
    return record


def new_navigation_error():
    """
    Return a new dictionary with every GSF_RECORD_NAVIGATION_ERROR field
    name present, in the same shape _decode_navigation_error() returns,
    each pre-set to None: every field is required to write this record.
    _decode_navigation_error() builds its result starting from this same
    template.

    :return: a dictionary with 'NavErrorTime', 'RecordID',
        'LongitudeError_m', and 'LatitudeError_m', each pre-set to None.
    """
    return {'NavErrorTime': None, 'RecordID': None, 'LongitudeError_m': None, 'LatitudeError_m': None}


def _decode_navigation_error(payload):
    """
    Decode a GSF_RECORD_NAVIGATION_ERROR payload: a time stamp, the
    identifier of the record the error estimate applies to, and estimated
    longitude/latitude error magnitudes. This record type is obsolete in
    the GSF format, having been superseded by
    GSF_RECORD_HV_NAVIGATION_ERROR. This is ported from the reference
    gsflib C library's gsfDecodeNavigationError() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_NAVIGATION_ERROR record is
    available.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_navigation_error() creates,
        with 'NavErrorTime' (an ISO 8601 string, in UTC), 'RecordID',
        'LongitudeError_m', and 'LatitudeError_m'.
    """
    record = new_navigation_error()
    (sec, nsec, record_id, lon_err_raw, lat_err_raw) = struct.unpack_from('>3I2i', payload, 0)
    record['NavErrorTime'] = _gsf_timestamp(sec, nsec).isoformat()
    record['RecordID'] = record_id
    record['LongitudeError_m'] = lon_err_raw / 10.0
    record['LatitudeError_m'] = lat_err_raw / 10.0
    return record


def new_hv_navigation_error():
    """
    Return a new dictionary with every GSF_RECORD_HV_NAVIGATION_ERROR
    field name present, in the same shape _decode_hv_navigation_error()
    returns: 'PositionType' pre-set to an empty string (it is optional),
    and every other field pre-set to None (each is required to write the
    record). _decode_hv_navigation_error() builds its result starting
    from this same template.

    :return: a dictionary with 'NavErrorTime', 'RecordID',
        'HorizontalError_m', 'VerticalError_m', and 'SEPUncertainty_m'
        (each None), and 'PositionType' ('').
    """
    return {
        'NavErrorTime': None, 'RecordID': None,
        'HorizontalError_m': None, 'VerticalError_m': None, 'SEPUncertainty_m': None,
        'PositionType': '',
    }


def _decode_hv_navigation_error(payload):
    """
    Decode a GSF_RECORD_HV_NAVIGATION_ERROR payload: a time stamp, the
    identifier of the record the error estimate applies to, estimated
    horizontal and vertical position error magnitudes, an estimated
    separation (SEP) uncertainty, and the name of the positioning system
    the estimate came from. This is ported from the reference gsflib C
    library's gsfDecodeHVNavigationError() function in gsf_dec.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_HV_NAVIGATION_ERROR record is
    available.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_hv_navigation_error()
        creates, with 'NavErrorTime' (an ISO 8601 string, in UTC),
        'RecordID', 'HorizontalError_m', 'VerticalError_m',
        'SEPUncertainty_m', and 'PositionType'.
    """
    record = new_hv_navigation_error()
    (sec, nsec, record_id, horiz_err_raw, vert_err_raw, sep_unc_raw) = \
        struct.unpack_from('>3I2iH', payload, 0)
    record['NavErrorTime'] = _gsf_timestamp(sec, nsec).isoformat()
    record['RecordID'] = record_id
    record['HorizontalError_m'] = horiz_err_raw / 1000.0
    record['VerticalError_m'] = vert_err_raw / 1000.0
    record['SEPUncertainty_m'] = sep_unc_raw / 100.0
    # 2 spare bytes at 22-23, then the position type string.
    (length,) = struct.unpack_from('>H', payload, 24)
    record['PositionType'] = payload[26:26 + length].decode('ascii', 'replace')
    return record


#: One attitude measurement as stored on the wire (gsf_dec.c's
#: gsfDecodeAttitude()): a millisecond time offset from the record's base
#: time, then pitch, roll, heave, and heading, each in hundredths. Heading
#: is unsigned; the other three are signed.
_ATTITUDE_MEASUREMENT_DTYPE = np.dtype([
    ('time_offset_ms', '>u2'),
    ('pitch', '>i2'),
    ('roll', '>i2'),
    ('heave', '>i2'),
    ('heading', '>u2'),
])


#: The fixed start of every GSF_RECORD_ATTITUDE payload: base time (seconds,
#: nanoseconds) and the number of measurements that follow.
_ATTITUDE_HEADER_DTYPE = np.dtype([('sec', '>u4'), ('nsec', '>u4'), ('num_measurements', '>u2')])


def new_attitude():
    """
    Return a new dictionary with every GSF_RECORD_ATTITUDE field name
    present, in the same shape _decode_attitude() returns: a
    'NumMeasurements' count of 0, and an empty numpy array for each
    per-measurement field. _decode_attitude() builds its result starting
    from this same template, so the two can never define a different set
    of field names.

    :return: a dictionary with 'NumMeasurements' (0), 'Time' (an empty
        numpy datetime64[ns] array), and 'Pitch_deg', 'Roll_deg',
        'Heave_m', and 'Heading_deg' (each an empty float64 array).
    """
    return {
        'NumMeasurements': 0,
        'Time': np.empty(0, dtype='datetime64[ns]'),
        'Pitch_deg': np.empty(0), 'Roll_deg': np.empty(0),
        'Heave_m': np.empty(0), 'Heading_deg': np.empty(0),
    }


def _decode_attitude(payload):
    """
    Decode a GSF_RECORD_ATTITUDE payload: a base time stamp plus a series
    of attitude measurements, each carrying a small time offset from that
    base time together with pitch, roll, heave, and heading values. This
    is ported from the reference gsflib C library's gsfDecodeAttitude()
    function in gsf_dec.c.

    Every measurement is unpacked at once, as a zero-copy numpy view of
    the payload, and each field is scaled with a single vectorized
    operation rather than a per-measurement loop. Each measurement's time
    is computed exactly, in integer nanoseconds, as the base time plus
    that measurement's millisecond offset.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_attitude() creates, with
        'NumMeasurements' (an int), 'Time' (a numpy datetime64[ns] array
        of each measurement's own time, in UTC), and 'Pitch_deg',
        'Roll_deg', 'Heave_m', and 'Heading_deg' (each a float64 numpy
        array, one value per measurement).
    """
    (base_sec, base_nsec, num_measurements) = struct.unpack_from('>2IH', payload, 0)
    m = np.frombuffer(payload, dtype=_ATTITUDE_MEASUREMENT_DTYPE, count=num_measurements, offset=10)

    record = new_attitude()
    record['NumMeasurements'] = num_measurements
    # uint16 * int64 promotes to a new int64 array, which the base time is
    # then added to in place, and viewed (not copied) as datetime64[ns].
    time_ns = m['time_offset_ms'] * np.int64(1_000_000)
    time_ns += base_sec * 1_000_000_000 + base_nsec
    record['Time'] = time_ns.view('datetime64[ns]')
    record['Pitch_deg'] = m['pitch'] / 100.0
    record['Roll_deg'] = m['roll'] / 100.0
    record['Heave_m'] = m['heave'] / 100.0
    record['Heading_deg'] = m['heading'] / 100.0
    return record


def new_header():
    """
    Return a new dictionary in the shape _decode_header() returns for a
    GSF_RECORD_HEADER record: 'Version' pre-set to None, which
    _encode_header() writes as this library's own GSF_VERSION.
    _decode_header() builds its result starting from this same template.

    :return: a dictionary with one key, 'Version', pre-set to None.
    """
    return {'Version': None}


def _decode_header(payload):
    """
    Decode a GSF_RECORD_HEADER payload, which holds nothing but the
    NUL-padded GSF format version string written by the encoder that
    produced the file (e.g. "GSF-v03.11"). This is ported from the
    reference gsflib C library's gsfDecodeHeader() function in
    gsf_dec.c. This version string is what the rest of this file's
    decoders rely on, via _gsf_major_version(), to decide which
    version-conditional fields -- such as a swath bathymetry ping's
    Height_m, SEP_m, and GPSTideCorrector_m fields -- should be present.

    :param payload: the raw bytes of the record.

    :return: a dictionary, in the shape new_header() creates, with one
        key, 'Version', holding the decoded version string with its
        trailing NUL padding removed.
    """
    record = new_header()
    record['Version'] = payload[:GSF_VERSION_SIZE].split(b'\x00', 1)[0].decode('ascii', 'replace')
    return record


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
    # gsf.h marks these three obsolete (VerticalError_m and
    # HorizontalError_m replace them), but gsf_enc.c still encodes them,
    # always as two-byte unsigned arrays.
    11: (100.0, 0.0, 2, False),      # DepthError_m
    12: (100.0, 0.0, 2, False),      # AcrossTrackError_m
    13: (100.0, 0.0, 2, False),      # AlongTrackError_m
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
    Round a floating-point value to the nearest integer using the
    reference gsflib C library's own rounding convention: 0.501 is added
    before truncating toward zero for non-negative values, and 0.501 is
    subtracted before truncating toward zero for negative values. This
    differs from Python's built-in round() function, which uses banker's
    rounding, and is used throughout this file's encode functions
    wherever a scaled floating-point value must be converted to the
    integer that gsflib itself would write, matching the reference
    gsf_enc.c library's own `+/-0.501` truncating-cast idiom exactly, bit
    for bit.

    :param x: the floating-point value to round.

    :return: the rounded value, as a Python int.
    """
    return int(x + 0.501) if x >= 0.0 else int(x - 0.501)


def _gsf_epoch(time_value):
    """
    Convert a caller-supplied time value into the (seconds, nanoseconds)
    pair since the Unix epoch that GSF stores on the wire for its
    timestamp fields. This is the inverse of _gsf_timestamp(), which
    combines such a pair back into a single Python datetime.

    Three forms of input are accepted: a POSIX timestamp (an int or
    float number of seconds since the epoch), a datetime.datetime (a
    naive datetime, with no timezone attached, is assumed to already be
    in UTC), or an ISO 8601 string of the kind
    _gsf_timestamp(...).isoformat() produces, so that a time field taken
    directly from a dictionary returned by one of this file's decode
    functions can be passed straight back into an encoder unchanged.
    Whichever form is given, the whole-seconds component is truncated
    toward zero and the nanosecond remainder is derived from the
    fractional part using _gsf_round(); if that rounding pushes the
    nanosecond count negative, one second is borrowed from the seconds
    component to bring the nanosecond value back into the valid
    0-999,999,999 range.

    :param time_value: the time to convert: a POSIX timestamp, a
        datetime.datetime, or an ISO 8601 string.

    :return: a tuple of (sec, nsec): whole seconds since the Unix epoch,
        and the nanosecond remainder within that second.
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


def _require_fields(record, names, template_name):
    """
    Check that a record dictionary passed to an encoder supplies every
    field that encoder requires.

    :param record: the record dictionary.
    :param names: the required field names.
    :param template_name: the new_*() function that creates this record
        type's template, named in the error message.

    :raises ValueError: one or more of names is missing from record, or
        present but set to None.
    """
    missing = [k for k in names if record.get(k) is None]
    if missing:
        raise ValueError("record is missing required field(s): %s (see %s())"
                         % (', '.join(missing), template_name))


def _encode_header(record=None):
    """
    Encode a GSF_RECORD_HEADER payload: the NUL-padded GSF format
    version string written at the very start of every GSF file. This is
    ported from the reference gsflib C library's gsfEncodeHeader()
    function in gsf_enc.c.

    The reference encoder always stamps the library's own current
    version string into the header, ignoring any version value a caller
    might otherwise want to supply -- there is no way to make
    gsfEncodeHeader() write anything other than the version gsflib
    itself was built with. This function mirrors that behavior: a
    record's 'Version' exists only so tests can exercise the
    padding/truncation logic with a known string, and in ordinary use
    it is None (as new_header() creates it), which writes this library's
    own GSF_VERSION constant.

    :param record: a dictionary in the shape _decode_header() returns or
        new_header() creates. Optional; omitted, or with 'Version' None,
        GSF_VERSION is written.

    :return: the encoded, NUL-padded GSF_VERSION_SIZE-byte version
        field, as bytes.
    """
    version = (record or {}).get('Version') or GSF_VERSION
    encoded = version.encode('ascii')
    return encoded[:GSF_VERSION_SIZE].ljust(GSF_VERSION_SIZE, b'\x00')


def _encode_name_value_parameters(record):
    """
    Encode the on-disk format shared by GSF_RECORD_PROCESSING_PARAMETERS
    and GSF_RECORD_SENSOR_PARAMETERS. This is ported from the reference
    gsflib C library's gsfEncodeProcessingParameters() and
    gsfEncodeSensorParameters() functions in gsf_enc.c, which write an
    identical wire format. Each parameter is written as a NUL-terminated
    "NAME=VALUE" string, with its 2-byte size field counting that
    trailing NUL byte, matching what real encoders write and what
    _decode_name_value_parameters() expects to find (see that function's
    docstring for how the trailing NUL is stripped back out on decode). A
    parameter whose value is None is written as its bare NAME, with no
    '=', as _decode_name_value_parameters() decodes such a string.

    :param record: a dictionary in the shape
        _decode_name_value_parameters() returns or
        new_name_value_parameters() creates: 'ParamTime' (required; a
        POSIX timestamp, datetime.datetime, or ISO 8601 string) and
        'Parameters', a dictionary of {name: value} pairs, each value
        written with str() (or omitted, if None).

    :return: the encoded record payload, as bytes.

    :raises ValueError: 'ParamTime' is missing or None.
    """
    _require_fields(record, ('ParamTime',), 'new_name_value_parameters')
    params = record.get('Parameters') or {}
    sec, nsec = _gsf_epoch(record['ParamTime'])
    out = struct.pack('>2IH', sec, nsec, len(params))
    for name, value in params.items():
        text = (name if value is None else "%s=%s" % (name, value)).encode('ascii') + b'\x00'
        out += struct.pack('>h', len(text)) + text
    return out


def _encode_sound_velocity_profile(record):
    """
    Encode a GSF_RECORD_SOUND_VELOCITY_PROFILE payload: the observation
    and application time stamps, the position where the profile was
    collected, and the depth/sound-speed pairs that make up the profile
    itself. This is ported from the reference gsflib C library's
    gsfEncodeSoundVelocityProfile() function in gsf_enc.c.

    :param record: a dictionary in the shape
        _decode_sound_velocity_profile() returns or
        new_sound_velocity_profile() creates. 'ObservationTime' and
        'ApplicationTime' (each a POSIX timestamp, datetime.datetime, or
        ISO 8601 string), 'Latitude_deg', 'Longitude_deg', 'Depth_m', and
        'SoundSpeed_mPerSec' are all required. 'Depth_m' and
        'SoundSpeed_mPerSec' are array-likes of the same length, in meters
        and meters per second; each must be non-negative, since both are
        stored on disk as unsigned integer counts of hundredths.
        'NumberPoints', if present, is ignored: the point count written is
        the length of 'Depth_m'.

    :return: the encoded record payload, as bytes.

    :raises ValueError: a required field is missing, or 'Depth_m' and
        'SoundSpeed_mPerSec' are not the same length.
    """
    _require_fields(record, ('ObservationTime', 'ApplicationTime', 'Latitude_deg', 'Longitude_deg',
                             'Depth_m', 'SoundSpeed_mPerSec'), 'new_sound_velocity_profile')
    observation_time = record['ObservationTime']
    application_time = record['ApplicationTime']
    latitude_deg = record['Latitude_deg']
    longitude_deg = record['Longitude_deg']
    depth_m = record['Depth_m']
    sound_speed_mPerSec = record['SoundSpeed_mPerSec']
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


def _gsf_round_array(x):
    """
    Round each element of a floating-point numpy array to the nearest
    integer exactly as _gsf_round() rounds a single value: 0.501 is added
    before truncating toward zero for non-negative values, and subtracted
    before truncating toward zero for negative values, matching gsf_enc.c's
    `+/-0.501` truncating-cast idiom.

    :param x: a numpy array of floating-point values.

    :return: a numpy int64 array of the rounded values.
    """
    return np.where(x >= 0.0, x + 0.501, x - 0.501).astype(np.int64)


def _gsf_epoch_ns_array(times):
    """
    Convert an array-like of caller-supplied time values into a numpy
    int64 array of nanoseconds since the Unix epoch. A numpy datetime64
    array (as _decode_attitude() returns) and an array of POSIX
    timestamps are both converted with vectorized operations; any other
    sequence (of datetime.datetime objects or ISO 8601 strings) is
    converted one element at a time with _gsf_epoch(). POSIX timestamps
    are split into whole seconds and a rounded nanosecond remainder
    exactly as _gsf_epoch() splits a single one.

    :param times: an array-like of times: numpy datetime64 values, POSIX
        timestamps, datetime.datetime objects, or ISO 8601 strings.

    :return: a numpy int64 array of nanoseconds since the Unix epoch.
    """
    a = np.asarray(times)
    if a.dtype.kind == 'M':
        return a.astype('datetime64[ns]').view(np.int64)
    if a.dtype.kind in 'iuf':
        a = a.astype(np.float64, copy=False)
        sec = np.trunc(a)
        return sec.astype(np.int64) * 1_000_000_000 + _gsf_round_array((a - sec) * 1.0e9)
    return np.array([sec * 1_000_000_000 + nsec for sec, nsec in map(_gsf_epoch, a.tolist())],
                    dtype=np.int64)


def _encode_attitude(record):
    """
    Encode a GSF_RECORD_ATTITUDE payload: a base time stamp plus a
    series of attitude measurements, each stored as a small time offset
    from that base time together with pitch, roll, heave, and heading
    values. This is ported from the reference gsflib C library's
    gsfEncodeAttitude() function in gsf_enc.c.

    The first entry of 'Time' becomes the record's base time, and every
    measurement -- including the first one -- is stored as a
    millisecond offset from that base time. Because each offset is
    stored on disk as an unsigned 16-bit field, 'Time' must be
    non-decreasing and must span less than 65.536 seconds from its first
    entry to its last. Every measurement is scaled, rounded, and packed
    with vectorized numpy operations rather than a per-measurement loop.

    :param record: a dictionary in the shape _decode_attitude() returns
        or new_attitude() creates, so a decoded record (or the whole-file
        result of gsf.read_attitude(), split into spans under 65.536 s)
        can be passed straight back in. 'Time' (an array-like of numpy
        datetime64 values, POSIX timestamps, or datetime.datetime
        objects), 'Pitch_deg', 'Roll_deg', 'Heave_m', and 'Heading_deg'
        (each an array-like in degrees or meters, the same length as
        'Time') are all required. 'NumMeasurements', if present, is
        ignored: the count written is the length of 'Time'.

    :return: the encoded record payload, as bytes.

    :raises ValueError: a required field is missing; the arrays are not
        all the same length, or are empty; or a time offset or scaled
        value does not fit its on-disk field.
    """
    _require_fields(record, ('Time', 'Pitch_deg', 'Roll_deg', 'Heave_m', 'Heading_deg'), 'new_attitude')
    attitude_time = record['Time']
    pitch_deg = record['Pitch_deg']
    roll_deg = record['Roll_deg']
    heave_m = record['Heave_m']
    heading_deg = record['Heading_deg']
    n = len(attitude_time)
    if not (len(pitch_deg) == len(roll_deg) == len(heave_m) == len(heading_deg) == n):
        raise ValueError("attitude arrays must all be the same length")
    if n == 0:
        raise ValueError("an attitude record needs at least one measurement")

    time_ns = _gsf_epoch_ns_array(attitude_time)
    base_sec, base_nsec = divmod(int(time_ns[0]), 1_000_000_000)

    m = np.empty(n, dtype=_ATTITUDE_MEASUREMENT_DTYPE)
    fields = (
        ('time_offset_ms', (time_ns - time_ns[0]) / 1.0e6),
        ('pitch', np.asarray(pitch_deg, dtype=np.float64) * 100.0),
        ('roll', np.asarray(roll_deg, dtype=np.float64) * 100.0),
        ('heave', np.asarray(heave_m, dtype=np.float64) * 100.0),
        ('heading', np.asarray(heading_deg, dtype=np.float64) * 100.0),
    )
    for name, scaled in fields:
        rounded = _gsf_round_array(scaled)
        info = np.iinfo(_ATTITUDE_MEASUREMENT_DTYPE[name])
        if rounded.min() < info.min or rounded.max() > info.max:
            raise ValueError("attitude %s out of range for its %s on-disk field"
                             % (name, _ATTITUDE_MEASUREMENT_DTYPE[name].name))
        m[name] = rounded

    return struct.pack('>2IH', base_sec, base_nsec, n) + m.tobytes()


def _encode_swath_bathy_summary(record):
    """
    Encode a GSF_RECORD_SWATH_BATHY_SUMMARY payload: the start and end
    times, the geographic bounding box, and the depth range covered by a
    contiguous run of swath bathymetry pings. This is ported from the
    reference gsflib C library's gsfEncodeSwathBathySummary() function
    in gsf_enc.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_SWATH_BATHY_SUMMARY record is
    available.

    :param record: a dictionary in the shape
        _decode_swath_bathy_summary() returns or new_swath_bathy_summary()
        creates, with every field required: 'StartTime' and 'EndTime'
        (each a POSIX timestamp, datetime.datetime, or ISO 8601 string),
        'MinLatitude_deg', 'MinLongitude_deg', 'MaxLatitude_deg', and
        'MaxLongitude_deg' (decimal degrees), and 'MinDepth_m' and
        'MaxDepth_m' (meters).

    :return: the encoded record payload, as bytes.

    :raises ValueError: a required field is missing.
    """
    fields = tuple(new_swath_bathy_summary())
    _require_fields(record, fields, 'new_swath_bathy_summary')
    start_sec, start_nsec = _gsf_epoch(record['StartTime'])
    end_sec, end_nsec = _gsf_epoch(record['EndTime'])
    out = struct.pack('>4I', start_sec, start_nsec, end_sec, end_nsec)
    for key, scale in (
        ('MinLatitude_deg', 1.0e7), ('MinLongitude_deg', 1.0e7),
        ('MaxLatitude_deg', 1.0e7), ('MaxLongitude_deg', 1.0e7),
        ('MinDepth_m', 100.0), ('MaxDepth_m', 100.0),
    ):
        out += struct.pack('>i', _gsf_round(record[key] * scale))
    return out


def _encode_comment(record):
    """
    Encode a GSF_RECORD_COMMENT payload: a time stamp and a single
    free-text comment string. This is ported from the reference gsflib C
    library's gsfEncodeComment() function in gsf_enc.c.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_COMMENT record is available.

    :param record: a dictionary in the shape _decode_comment() returns or
        new_comment() creates: 'CommentTime' (required; a POSIX
        timestamp, datetime.datetime, or ISO 8601 string) and 'Comment',
        the comment text.

    :return: the encoded record payload, as bytes.

    :raises ValueError: 'CommentTime' is missing or None.
    """
    _require_fields(record, ('CommentTime',), 'new_comment')
    sec, nsec = _gsf_epoch(record['CommentTime'])
    text = record.get('Comment', '').encode('ascii')
    return struct.pack('>3I', sec, nsec, len(text)) + text


def _encode_history(record):
    """
    Encode a GSF_RECORD_HISTORY payload: a time stamp plus the host
    name, operator name, command line, and comment text describing one
    processing step in the file's history. This is ported from the
    reference gsflib C library's gsfEncodeHistory() function in
    gsf_enc.c.

    The host name, operator name, and command line are each written as
    a NUL-terminated string, with the 2-byte size field counting that
    trailing NUL byte. The comment is the odd one out: it is written with
    no NUL terminator, and its size field is simply the plain string
    length. This matches gsf_enc.c exactly. _decode_history() strips
    those NULs back off, so a decoded record re-encodes to the same
    bytes.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_HISTORY record is available.

    :param record: a dictionary in the shape _decode_history() returns or
        new_history() creates: 'HistoryTime' (required; a POSIX timestamp,
        datetime.datetime, or ISO 8601 string), and 'HostName',
        'OperatorName', 'CommandLine', and 'Comment' (each text; a missing
        one is written as an empty string).

    :return: the encoded record payload, as bytes.

    :raises ValueError: 'HistoryTime' is missing or None.
    """
    _require_fields(record, ('HistoryTime',), 'new_history')
    sec, nsec = _gsf_epoch(record['HistoryTime'])
    out = struct.pack('>2I', sec, nsec)
    for key in ('HostName', 'OperatorName', 'CommandLine'):
        text = record.get(key, '').encode('ascii') + b'\x00'
        out += struct.pack('>H', len(text)) + text
    text = record.get('Comment', '').encode('ascii')
    out += struct.pack('>H', len(text)) + text
    return out


def _encode_navigation_error(record):
    """
    Encode a GSF_RECORD_NAVIGATION_ERROR payload: a time stamp, the
    identifier of the record an error estimate applies to, and estimated
    longitude/latitude error magnitudes. This record type is obsolete in
    the GSF format, having been superseded by
    GSF_RECORD_HV_NAVIGATION_ERROR (see _encode_hv_navigation_error()).
    This is ported from the reference gsflib C library's
    gsfEncodeNavigationError() function in gsf_enc.c, with one
    deliberate deviation from that reference implementation.

    The reference encoder rounds both error fields with an unconditional
    "+ 0.501" that does not check the value's sign. For a negative error
    value this is a rounding bug: for example, a 'LongitudeError_m' of
    -1.29 m, using the reference's own formula and then truncating
    toward zero after scaling, encodes to -12 (in units of 1/10 m)
    instead of the correctly rounded -13. This function does not
    reproduce that bug: it uses the standard, sign-correct _gsf_round()
    convention that every other encoder in this module uses, which
    rounds negative values by subtracting 0.501 before truncating rather
    than adding it.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_NAVIGATION_ERROR record is
    available.

    :param record: a dictionary in the shape _decode_navigation_error()
        returns or new_navigation_error() creates, with every field
        required: 'NavErrorTime' (a POSIX timestamp, datetime.datetime, or
        ISO 8601 string), 'RecordID' (the identifier of the record this
        error estimate applies to), and 'LongitudeError_m' and
        'LatitudeError_m' (the estimated errors, in meters).

    :return: the encoded record payload, as bytes.

    :raises ValueError: a required field is missing.
    """
    _require_fields(record, tuple(new_navigation_error()), 'new_navigation_error')
    sec, nsec = _gsf_epoch(record['NavErrorTime'])
    out = struct.pack('>3I', sec, nsec, int(record['RecordID']))
    out += struct.pack('>i', _gsf_round(record['LongitudeError_m'] * 10.0))
    out += struct.pack('>i', _gsf_round(record['LatitudeError_m'] * 10.0))
    return out


def _encode_hv_navigation_error(record):
    """
    Encode a GSF_RECORD_HV_NAVIGATION_ERROR payload: a time stamp, the
    identifier of the record an error estimate applies to, estimated
    horizontal and vertical position error magnitudes, an estimated
    separation (SEP) uncertainty, and the name of the positioning system
    the estimate came from. This is ported from the reference gsflib C
    library's gsfEncodeHVNavigationError() function in gsf_enc.c.

    The reference encoder rounds the vertical error field using a plain
    "+/- 0.5" rather than the "+/- 0.501" convention used everywhere
    else in gsf_enc.c, including for the horizontal error in this same
    function. The two conventions are functionally equivalent except
    exactly on a 0.5 fractional boundary, so this function uses the
    standard _gsf_round() convention, with its 0.501 margin, for both
    fields rather than reproducing that one-field inconsistency.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_HV_NAVIGATION_ERROR record is
    available.

    :param record: a dictionary in the shape
        _decode_hv_navigation_error() returns or new_hv_navigation_error()
        creates: 'NavErrorTime' (a POSIX timestamp, datetime.datetime, or
        ISO 8601 string), 'RecordID' (the identifier of the record this
        error estimate applies to), and 'HorizontalError_m',
        'VerticalError_m', and 'SEPUncertainty_m' (meters) are required;
        'PositionType', the name of the positioning system the estimate
        came from, is optional and defaults to an empty string.

    :return: the encoded record payload, as bytes.

    :raises ValueError: a required field is missing.
    """
    _require_fields(record, ('NavErrorTime', 'RecordID', 'HorizontalError_m', 'VerticalError_m',
                             'SEPUncertainty_m'), 'new_hv_navigation_error')
    sec, nsec = _gsf_epoch(record['NavErrorTime'])
    out = struct.pack('>3I', sec, nsec, int(record['RecordID']))
    out += struct.pack('>i', _gsf_round(record['HorizontalError_m'] * 1000.0))
    out += struct.pack('>i', _gsf_round(record['VerticalError_m'] * 1000.0))
    out += struct.pack('>H', _gsf_round(record['SEPUncertainty_m'] * 100.0))
    out += b'\x00\x00'  # spare
    text = (record.get('PositionType') or '').encode('ascii')
    out += struct.pack('>H', len(text)) + text
    return out


#: Fields _encode_single_beam_ping()/_encode_swath_bathymetry_ping() require
#: `record` to supply (see each function's docstring).
_REQUIRED_SINGLE_BEAM_PING_FIELDS = (
    'PingTime', 'Longitude_deg', 'Latitude_deg', 'TideCorrector_m',
    'DepthCorrector_m', 'Heading_deg', 'Pitch_deg', 'Roll_deg', 'Heave_m',
    'Depth_m', 'SoundSpeedCorrection_m')


def _encode_single_beam_ping(record):
    """
    Encode a GSF_RECORD_SINGLE_BEAM_PING payload from one merged
    dictionary: the fixed-format scalar fields, plus, if present, one
    sensor-specific tail subrecord encoded via the
    _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS registry. This is ported from
    the reference gsflib C library's gsfEncodeSinglebeam() function in
    gsf_enc.c, and is the exact inverse of _decode_single_beam_ping(): a
    dictionary returned by that decoder can be passed straight back into
    this function unmodified.

    This function has not been tested against a verified GSF file, since
    no sample data containing a GSF_RECORD_SINGLE_BEAM_PING record is
    available.

    :param record: a dictionary shaped like the one
        _decode_single_beam_ping() returns. The eleven fixed scalar
        fields listed in _REQUIRED_SINGLE_BEAM_PING_FIELDS ('PingTime',
        'Longitude_deg', 'Latitude_deg', 'TideCorrector_m',
        'DepthCorrector_m', 'Heading_deg', 'Pitch_deg', 'Roll_deg',
        'Heave_m', 'Depth_m', and 'SoundSpeedCorrection_m') are all
        required. 'PositioningSystemType' is optional and defaults to 0.
        'SensorSpecificID' (an int) and 'SensorSpecific' (a dictionary,
        with any table values as dictionaries of column name ->
        array-like, or pandas.DataFrames) are optional and,
        together, describe the one sensor-specific tail subrecord to
        encode. 'Notes', if present, is ignored, since there is no wire
        slot for it.

    :return: the encoded record payload, as bytes.

    :raises ValueError: one or more of the required fields listed in
        _REQUIRED_SINGLE_BEAM_PING_FIELDS is missing from record (or
        present but set to None).
    :raises KeyError: record['SensorSpecificID'] is set to a subrecord
        id that has no encoder registered in
        _SINGLE_BEAM_SENSOR_SPECIFIC_CODECS.
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
        sensor_specific = dict(record.get('SensorSpecific') or {})
        sensor_specific['SubrecordID'] = subrecord_id
        out += encode_fn(sensor_specific)

    return out


def _encode_scale_factors(scale_factors):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS subrecord, including
    its own 4-byte subrecord id and size word, from a dictionary of
    per-beam-array (multiplier, offset, compression flag) triples. This
    is ported from the reference gsflib C library's EncodeScaleFactors()
    function in gsf_enc.c, and is the inverse of _decode_scale_factors().
    Entries are written out in ascending subrecord id order, regardless
    of the order in which they appear in the input dictionary.

    :param scale_factors: a dictionary mapping each beam-array subrecord
        id to a (multiplier, offset, compressionFlag) tuple, in the same
        shape _decode_scale_factors() returns.

    :return: the encoded subrecord, including its id+size header word,
        as bytes.
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
    Encode one scaled per-beam array subrecord, including its own 4-byte
    subrecord id and size word, from an array of engineering-unit
    values. Each value is converted to its raw on-disk integer as
    raw = round((value + offset) * multiplier), using _gsf_round()'s
    sign-correct rounding convention, and the whole array is converted
    at once with numpy rather than looping over beams in Python. This is
    ported from the reference gsflib C library's family of per-width,
    per-signedness Encode*Array functions in gsf_enc.c, and is the
    inverse of _decode_ping_array().

    :param subrecord_id: the beam-array subrecord id to write into the
        id+size header word.
    :param values: an array-like of engineering-unit values to encode,
        one per beam.
    :param multiplier: this array's scale-factor multiplier, applied to
        each value before rounding to an integer.
    :param offset: this array's scale-factor offset, added to each value
        before the multiplier is applied.
    :param signed: True to encode each value as a signed integer, False
        to encode it as unsigned.
    :param width: the on-disk width, in bytes (1, 2, or 4), of each
        beam's encoded value.

    :return: the encoded subrecord, including its id+size header word,
        as bytes.

    :raises ValueError: after scaling and rounding, at least one encoded
        value does not fit within the representable integer range of
        the requested width and signedness, rather than being silently
        wrapped or truncated.
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
    Check whether a given (multiplier, offset) pair would let
    _encode_ping_array() represent both min_v and max_v, the smallest
    and largest values in some beam array, within the integer range of
    a field of the given width and signedness, without raising that
    function's own out-of-range ValueError. This is not ported from the
    reference gsflib C library; it is an original helper written to
    support this library's own automatic scale-factor feature
    (auto_scale=True).

    This function reuses _gsf_round(), the exact rounding convention
    _encode_ping_array() itself applies to every scaled value, so a
    "yes" answer from this function is guaranteed to agree with what the
    real encoder would actually do: there is no separate, approximate
    copy of the rounding rule here that could drift out of sync with it.

    :param min_v: the smallest engineering-unit value that needs to be
        representable.
    :param max_v: the largest engineering-unit value that needs to be
        representable.
    :param multiplier: the candidate scale-factor multiplier being
        tested, the same value _encode_ping_array() would use to
        compute raw = round((value + offset) * multiplier) for every
        beam.
    :param offset: the candidate scale-factor offset being tested, used
        the same way.
    :param width: the field width, in bytes (1, 2, or 4).
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
    doesn't actually require a shift. This is not ported from the
    reference gsflib C library; it is an original helper written to
    support this library's own automatic scale-factor feature
    (auto_scale=True), solving for a valid scale factor given a target
    precision and an optional padding fraction.

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

    :param min_v: the smallest value in the range to represent (already
        padded by the caller if any hysteresis headroom is wanted; this
        function does not apply any padding of its own beyond the `pad`
        argument).
    :param max_v: the largest value in the range to represent, subject
        to the same caveat about padding.
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
    Choose the (multiplier, offset) scale factor to use for one ping's
    beam array, reusing the scale factor already active for this
    subrecord id whenever it still works, and only solving for a new one
    when it doesn't. This is the entry point that write_swath_bathymetry_ping()
    calls once per beam-array label when writing with auto_scale=True. It
    is not ported from the reference gsflib C library. It is this
    project's own algorithm for automatic scale-factor selection, loosely
    inspired by a much narrower feature gsflib itself has (gsflib's own
    auto-offset feature only adjusts a depth array's offset, in fixed
    100m "layers," to keep tide-corrected depths non-negative); this
    function instead solves for both the multiplier and the offset
    together, for any beam array, and adds hysteresis so that the scale
    factor does not change on every single ping. See the README's "Scale
    factors" section for a worked example of this feature in use.

    This function is designed to be cheap to call on every single ping:
    it does exactly one vectorized pass over `values` (a numpy min/max
    reduction) to find this ping's value range, then does a handful of
    scalar comparisons. There is no per-beam Python loop, and in the
    common case, where the currently active scale factor already covers
    this ping, no further arithmetic happens at all.

    The algorithm proceeds in up to four steps. First, it finds this
    ping's actual value range from `values`, ignoring any non-finite
    (NaN/Inf) entries, since GSF has no on-disk representation for "not
    a number" in a beam array; if every value is non-finite, there is
    nothing to fit, and the function returns `current` unchanged (or, if
    `current` is None, (target_multiplier, 0.0)). Second, it checks
    whether the scale factor already active for this subrecord id still
    covers this ping's range; if so, that scale factor is returned
    unchanged, and no new value is computed at all -- this fast path is
    what "the scale factors don't need to change with every ping" means
    in practice. Third, if the current scale factor doesn't cover this
    ping's range (or there is no current scale factor yet, because this
    is the first ping this subrecord id has been seen in), a new one is
    genuinely needed. Rather than solving for the tightest possible fit
    to just this ping's exact range, which the very next, slightly
    different ping might again exceed and so force another change
    immediately, the target range is padded outward first and a scale
    factor is solved for that padded range. This padding is the
    hysteresis: it trades a small amount of precision headroom for
    scale-factor stability across pings, the same trade gsflib's own
    depth-offset auto-offset feature makes with its fixed-size "layers,"
    but computed directly from the observed data instead of a fixed,
    depth-specific size. Fourth, because that padding is only a
    nice-to-have for stability and must never make representable data
    unrepresentable, the padded solution is checked against the field's
    actual capacity; in the rare case where a field is already near its
    capacity, padding outward can push the solve past what the field can
    hold even though the ping's actual, unpadded data would fit fine, so
    if that happens, the function retries with an exact, unpadded fit. If
    even that doesn't fit, this field genuinely cannot represent this
    ping's data at its declared width, and rather than trying to paper
    over that here, the function returns its best attempt and lets
    _encode_ping_array()'s own out-of-range error be the one place that
    failure is ever raised, exactly as it already is for a beam array
    written with a static, non-auto-scaled scale factor.

    :param values: this ping's beam array, in engineering units (for
        example depth in meters).
    :param current: the (multiplier, offset) pair currently active for
        this subrecord id, as previously returned by this function, or
        None if this is the first ping this subrecord id has been seen
        in for this file.
    :param width: the on-disk field width, in bytes (1, 2, or 4).
    :param signed: whether the on-disk integer field is signed.
    :param target_multiplier: the ideal, ceiling precision for this
        field (for example 1000.0 for 1mm depth resolution) -- see
        DEFAULT_PING_SCALE_FACTORS's docstring note on its dual role
        when writing with auto_scale=True.
    :param margin_fraction: the fraction of this ping's observed value
        range to pad outward by when a new scale factor needs to be
        solved for, as part of the hysteresis described above. See the
        module-level _AUTO_SCALE_MARGIN_FRACTION constant's docstring
        for more detail.
    :param min_margin_steps: a floor on the padding above, expressed as
        a fixed number of steps at the field's target precision, so that
        a ping whose observed value range is tiny or zero (for example a
        dead flat seafloor, or calm heave) still gets meaningful padding.
        See the module-level _AUTO_SCALE_MIN_MARGIN_STEPS constant's
        docstring for more detail.

    :return: the (multiplier, offset) pair to use for this ping -- either
        `current` unchanged, or a freshly solved pair.
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
    including its own 4-byte subrecord id and size word, packing four
    beams' 2-bit quality flags into each output byte. This is the
    inverse of _decode_quality_flags_array(). This is ported from the
    reference gsflib C library's EncodeQualityFlagsArray() function in
    gsf_enc.c, and is vectorized with numpy rather than encoding one beam
    at a time.

    :param values: an array-like of per-beam quality flags. Each value is
        masked down to its low 2 bits (range 0-3) before packing,
        matching the implicit truncation the reference C source performs.

    :return: the encoded subrecord as bytes, including its leading 4-byte
        subrecord id and size word.
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
    Return a new dictionary with every GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC
    scalar field name present, pre-set to 0 or 0.0 (or, for
    NumBytesPerTxSector and NumBytesPerClass, the fixed byte counts that
    _encode_kmall_specific() itself defaults to). This function exists so
    that a caller building a KMALL ping to write doesn't have to remember
    every field name by hand. gsf.h defines no GSF_NULL_* "not available"
    sentinel for these vendor-specific fields, unlike the ping scalars
    that new_swath_bathymetry_ping() fills in, so 0 is the only
    "not specified" marker available here -- if 0 is itself a plausible
    value for a field you care about, be sure to actually set it in the
    returned dictionary rather than relying on this default. Field
    meanings are documented on _decode_kmall_specific(); convert.md
    points to real sample values.

    Every field _decode_kmall_specific() returns is present, in the same
    order, and _decode_kmall_specific() builds its result starting from
    this same template. Three of them are present only for that reason:
    GSFKMALLVersion is always forced to 0 when the dictionary is encoded,
    reproducing a quirk of the reference gsflib C library's encoder (see
    _encode_kmall_specific()), and NumTxSectors and
    NumExtraDetectionClasses are always derived from the number of rows in
    record['TxSectors'] and record['ExtraDetectionClasses'] when encoding,
    so there is no point setting any of the three here.

    :return: a dictionary with every valid KMALL_SPECIFIC scalar field
        name as a key, pre-filled with a default value as described
        above.
    """
    return {
        'GSFKMALLVersion': 0, 'DgmType': 0, 'DgmVersion': 0, 'SystemID': 0,
        'EchoSounderID': 0, 'NumBytesCmnPart': 0, 'PingCnt': 0, 'RxFansPerPing': 0,
        'RxFanIndex': 0, 'SwathsPerPing': 0, 'SwathAlongPosition': 0, 'TxTransducerInd': 0,
        'RxTransducerInd': 0, 'NumRxTransducers': 0, 'AlgorithmType': 0,
        'NumBytesInfoData': 0, 'PingRate_Hz': 0.0, 'BeamSpacing': 0, 'DepthMode': 0,
        'SubDepthMode': 0, 'DistanceBtwSwath': 0, 'DetectionMode': 0, 'PulseForm': 0,
        'FrequencyMode_Hz': 0.0, 'FreqRangeLowLim_Hz': 0.0, 'FreqRangeHighLim_Hz': 0.0,
        'MaxTotalTxPulseLength_sec': 0.0, 'MaxEffTxPulseLength_sec': 0.0,
        'MaxEffTxBandWidth_Hz': 0.0, 'AbsCoeff_dBPerkm': 0.0, 'PortSectorEdge_deg': 0.0,
        'StarbSectorEdge_deg': 0.0, 'PortMeanCov_deg': 0.0, 'StarbMeanCov_deg': 0.0,
        'PortMeanCov_m': 0.0, 'StarbMeanCov_m': 0.0, 'ModeAndStabilisation': 0,
        'RuntimeFilter1': 0, 'RuntimeFilter2': 0, 'PipeTrackingStatus': 0,
        'TransmitArraySizeUsed_deg': 0.0, 'ReceiveArraySizeUsed_deg': 0.0,
        'TransmitPower_dB': 0.0, 'SLrampUpTimeRemaining': 0, 'YawAngle_deg': 0.0,
        'NumTxSectors': 0, 'NumBytesPerTxSector': 53, 'HeadingVessel_deg': 0.0,
        'SoundSpeedAtTxDepth_mPerSec': 0.0, 'TxTransducerDepth_m': 0.0,
        'ZWaterLevelReRefPoint_m': 0.0, 'XKmallToAll_m': 0.0, 'YKmallToAll_m': 0.0,
        'LatLongInfo': 0, 'PosSensorStatus': 0, 'AttitudeSensorStatus': 0,
        'Latitude_deg': 0.0, 'Longitude_deg': 0.0, 'EllipsoidHeightReRefPoint_m': 0.0,
        'NumBytesRxInfo': 0, 'NumSoundingsMaxMain': 0, 'NumSoundingsValidMain': 0,
        'NumBytesPerSounding': 0, 'WCSampleRate': 0.0, 'SeabedImageSampleRate': 0.0,
        'BSnormal_dB': 0.0, 'BSoblique_dB': 0.0, 'ExtraDetectionAlarmFlag': 0,
        'NumExtraDetections': 0, 'NumExtraDetectionClasses': 0, 'NumBytesPerClass': 35,
    }


def new_kmall_tx_sector():
    """
    Return a new dictionary with every per-transmit-sector field name
    present, pre-set to 0 or 0.0, representing one row of the
    'TxSectors' table. This function exists so a caller can build that
    table without having to remember every field name by hand, for
    example by assigning pd.DataFrame([new_kmall_tx_sector(), ...]) (or
    the equivalent dictionary of column name -> array) to
    record['SensorSpecific']['TxSectors'] before calling
    write_swath_bathymetry_ping(). See new_kmall_specific() for why 0,
    rather than a GSF_NULL_* sentinel, is the best available default
    here.

    :return: a dictionary with every valid per-transmit-sector field name
        as a key, pre-filled with a default value as described above.
    """
    return {
        'TxSectorNumb': 0, 'TxArrNumber': 0, 'TxSubArray': 0,
        'SectorTransmitDelay_sec': 0.0, 'TiltAngleReTx_deg': 0.0,
        'TxNominalSourceLevel_dB': 0.0, 'TxFocusRange_m': 0.0,
        'CentreFreq_Hz': 0.0, 'SignalBandWidth_Hz': 0.0,
        'TotalSignalLength_sec': 0.0, 'PulseShading': 0, 'SignalWaveForm': 0,
        'HighVoltageLevel_dB': 0.0, 'SectorTrackingCorr_dB': 0.0,
        'EffectiveSignalLength_sec': 0.0,
    }


def _encode_kmall_specific(record):
    """
    Encode a GSF_SWATH_BATHY_SUBRECORD_KMALL_SPECIFIC subrecord (id 156),
    including its own 4-byte subrecord id and size word. This is ported
    from the reference gsflib C library's EncodeKMALLSpecific() function
    in gsf_enc.c, and is the exact field-for-field inverse of
    _decode_kmall_specific(): the dictionary that function returns can be
    passed straight back into this one, unmodified, to re-encode it.

    This function reproduces a quirk of the reference encoder: gsf_enc.c
    always writes 0 for the gsfKMALLVersion field on the wire, regardless
    of whatever value `record` carries for it, and this function does
    the same. It also mirrors the duplicate port/starboard
    mean-coverage-in-degrees pair that _decode_kmall_specific() documents
    on the decode side (an apparent copy/paste artifact in the reference
    decoder): to keep byte alignment identical to what a real KMALL file
    contains, this function writes four bytes of zero padding where that
    duplicated, discarded pair would have been, rather than writing real
    data there.

    :param record: a dictionary of scalar KMALL_SPECIFIC fields, in the
        same shape _decode_kmall_specific() returns them; any field
        missing from `record` defaults to 0. It may also carry the
        optional 'TxSectors' and 'ExtraDetectionClasses' keys, each a
        table (a dictionary of column name -> array-like, or a
        pandas.DataFrame), of which at most 9 and 11 rows respectively
        are written (GSF_MAX_KMALL_SECTORS and
        GSF_MAX_KMALL_EXTRA_CLASSES) -- the exact tables
        _decode_kmall_specific() returns can be passed straight back in.
        The NumTxSectors and NumExtraDetectionClasses fields are always
        derived from the number of rows actually written, not read from
        `record`.

    :return: the encoded subrecord as bytes, including its leading 4-byte
        subrecord id and size word.
    """
    sector_rows = record.get('TxSectors')
    class_rows = record.get('ExtraDetectionClasses')
    num_sectors = min(_table_length(sector_rows), 9)  # gsf.h: GSF_MAX_KMALL_SECTORS
    num_classes = min(_table_length(class_rows), 11)  # gsf.h: GSF_MAX_KMALL_EXTRA_CLASSES
    g = record.get

    out = bytearray()
    # gsf_enc.c always writes 0 for gsfKMALLVersion, regardless of `record`.
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
    out += struct.pack('>H', num_sectors)
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

    for row in _table_rows(sector_rows, limit=num_sectors):
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
    out += struct.pack('>H', num_classes)
    out += struct.pack('>H', g('NumBytesPerClass', 35))
    out += b'\x00' * 32

    for row in _table_rows(class_rows, limit=num_classes):
        out += struct.pack('>H', int(row.get('NumExtraDetInClass', 0)))
        out += struct.pack('>B', int(row.get('AlarmFlag', 0)))
        out += b'\x00' * 32

    out += b'\x00' * 32

    header_word = (_SUBRECORD_KMALL_SPECIFIC << 24) | len(out)
    return struct.pack('>I', header_word) + bytes(out)


_PING_SENSOR_SPECIFIC_CODECS[_SUBRECORD_KMALL_SPECIFIC] = (
    "KMALL", _decode_kmall_specific, _encode_kmall_specific)


def _beam_array_subrecord_id(label):
    """
    Resolve a beam-array column label (for example 'Depth_m' or
    'AcrossTrack_m', as used as a key in record['Beams']) to the GSF
    beam-array subrecord id it corresponds to, by looking it up in
    BEAM_ARRAY_SUBRECORD_IDS. Unlike a plain dictionary lookup, an unknown
    label raises a KeyError whose message says what went wrong.

    :param label: the beam-array column label to resolve.

    :return: the matching GSF beam-array subrecord id, as an int.

    :raises KeyError: `label` is not a key of BEAM_ARRAY_SUBRECORD_IDS.
    """
    if label in BEAM_ARRAY_SUBRECORD_IDS:
        return BEAM_ARRAY_SUBRECORD_IDS[label]
    raise KeyError("no known ping array subrecord for beams column %r" % label)


def new_swath_bathymetry_ping():
    """
    Return a new dictionary with every GSF_RECORD_SWATH_BATHYMETRY_PING
    field name present, in the same shape _decode_swath_bathymetry_ping()
    returns: the four required fields (PingTime, Longitude_deg,
    Latitude_deg, and NumberBeams) set to None as a placeholder that the
    caller must overwrite; every optional scalar field pre-set to its
    GSF_NULL_* "not available" sentinel (or, for CenterBeam, PingFlags, and
    GPSTideCorrector_m, to 0 or 0.0, since gsf.h defines no sentinel for
    those three fields); 'Beams', a table (a dictionary of column name ->
    array) with every label in BEAM_ARRAY_SUBRECORD_IDS as a column, each
    set to None to mean that the ping carries no such subrecord; and
    'Notes', an empty list. _decode_swath_bathymetry_ping() builds its
    result starting from this same template, replacing a 'Beams' column
    with an array whenever the ping carries that subrecord, and adding
    'IntensityTimeSeries', 'SensorSpecificID', and 'SensorSpecific' only
    when the ping carries them.

    This function exists so a caller building a ping to write doesn't
    have to remember every field name or look up its correct "not
    available" sentinel by hand: it can populate the returned dictionary
    with whatever it actually knows, set the 'Beams' columns it has data
    for (or replace 'Beams' entirely with a dictionary of
    {column label: array-like} or a pandas.DataFrame; columns left None or
    left out are not written) and, optionally, add
    'SensorSpecificID' and 'SensorSpecific' keys for a vendor
    sensor-specific subrecord (see that family's new_*_specific()
    template), and pass the result straight to
    write_swath_bathymetry_ping(). Every scalar field the caller doesn't
    touch is then written as "not available", rather than as a misleading
    0 or 0.0. See convert.md's "Marking a field as not available" section
    for more detail.

    :return: a dictionary with every swath-bathymetry-ping field name as a
        key, pre-filled with a default value as described above.

    :raises ValueError: raised later, by write_swath_bathymetry_ping()
        (via _encode_swath_bathymetry_ping()), not by this function
        itself, if PingTime, Longitude_deg, Latitude_deg, or NumberBeams
        is still None when the dictionary this function returns is
        passed to write_swath_bathymetry_ping().
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
        'Beams': dict.fromkeys(BEAM_ARRAY_SUBRECORD_IDS),
        'Notes': [],
    }



def _encode_swath_bathymetry_ping(record, scale_factors=None, major_version=3):
    """
    Encode a GSF_RECORD_SWATH_BATHYMETRY_PING payload: the fixed-format
    scalar fields, a GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS subrecord,
    one beam-array subrecord per entry in record['Beams'], and, if
    given, the vendor sensor-specific subrecord. This is the exact
    inverse of _decode_swath_bathymetry_ping(): `record` uses the same
    dictionary shape that function returns, so a ping decoded by that
    function can be re-encoded directly and unmodified by this one
    (PingTime accepts the same ISO 8601 string that
    _decode_swath_bathymetry_ping() produces -- see _gsf_epoch()).

    :param record: a dictionary describing the ping to encode.
        PingTime, Longitude_deg, Latitude_deg, and NumberBeams are
        required: they must be present in `record` and not None.
        Building this dictionary with new_swath_bathymetry_ping()
        is recommended over assembling it by hand, since that function
        pre-fills every valid scalar key name, so there is nothing to
        look up and nothing to get wrong. CenterBeam, PingFlags, and, at
        major_version greater than 2, GPSTideCorrector_m default to 0 or
        0.0 if absent from `record`, since no GSF_NULL_* sentinel is
        defined for these three fields. Every other scalar key
        (TideCorrector_m, DepthCorrector_m, Heading_deg, Pitch_deg,
        Roll_deg, Heave_m, Course_deg, Speed_kn, and, at major_version
        greater than 2, Height_m and SEP_m) defaults to its GSF_NULL_*
        sentinel (for example GSF_NULL_SPEED, 99.0 knots) if absent from
        `record`, per gsf.h's convention -- not to 0 or 0.0, since 0 is
        itself a valid measured value for most of these fields. Pass the
        value explicitly, whether 0.0 or otherwise, whenever it is
        known; omit the key only when the value is genuinely
        unavailable.

        record['Beams'], if present, is a dictionary of {column label:
        array-like} or a pandas.DataFrame, for example {'Depth_m': [...],
        'AcrossTrack_m': [...]}. Every array in it must have length
        NumberBeams. A column whose value is None, or an empty array, is
        not written, since gsflib never writes a zero-length subrecord. So
        the 'Beams' table from
        new_swath_bathymetry_ping() or from a decoded ping can be passed
        in as is. Only labels in BEAM_ARRAY_SUBRECORD_IDS can be encoded,
        and a non-empty scaled array also needs an entry in
        DEFAULT_PING_SCALE_FACTORS or in `scale_factors`. Omit
        record['Beams'], or pass an empty dictionary, for a ping with no
        beam arrays.

        record['SensorSpecificID'] and record['SensorSpecific'], if
        present, together describe one vendor sensor-specific subrecord,
        for any vendor family registered in _PING_SENSOR_SPECIFIC_CODECS
        (including KMALL_SPECIFIC, id 156). 'SensorSpecific' is a flat
        dictionary of that family's scalar field names, as returned,
        unprefixed, by _decode_swath_bathymetry_ping(), plus, for
        families with nested per-element arrays (EM3, EM3Raw, EM4, and
        KMALL), any table entries keyed by their table name (for example
        'TxSectors'), each a dictionary of column name -> array-like or
        a pandas.DataFrame -- the exact tables
        _decode_swath_bathymetry_ping() returns can be passed straight
        back in, with no conversion required.

        record['IntensityTimeSeries'], if present, is a dictionary in
        the same shape _decode_brb_intensity() returns: the fields
        new_intensity_time_series_header() creates, plus a 'Beams' table
        of each beam's 'DetectRangeSample', and a 'Samples' table of every
        sample's 'Beam', 'RangeSample', and 'Value', in beam order. It is
        encoded
        as a GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord,
        written after the sensor-specific subrecord, matching the
        reference gsflib C library's own subrecord ordering. Whichever
        vendor sensor-imagery preamble, if any, is written into this
        subrecord is chosen from record['SensorSpecificID'], not from
        anything inside record['IntensityTimeSeries'] itself.

        record['Notes'], if present, is ignored: there is no wire slot
        for record-level free-text notes in this subrecord.

    :param scale_factors: an optional override of
        DEFAULT_PING_SCALE_FACTORS, in the same shape (a dictionary
        mapping subrecord id to (multiplier, offset, field width in
        bytes, signed)), used to choose the on-disk encoding for each
        beam array in record['Beams']. Defaults to
        DEFAULT_PING_SCALE_FACTORS when not given.
    :param major_version: the major version number of the GSF file being
        written (for example 3 for a "GSF-v03.xx" file). It controls
        whether the Height_m, SEP_m, and GPSTideCorrector_m fields are
        written in the fixed header: they are only written when
        major_version is greater than 2, matching the version at which
        the reference gsflib C library itself started writing them.

    :return: the encoded ping payload as bytes, ready to be wrapped in a
        GSF_RECORD_HEADER and written to a file.

    :raises ValueError: raised when PingTime, Longitude_deg,
        Latitude_deg, or NumberBeams is missing or None in `record`, or
        when a beam-array or sensor-specific value cannot be represented
        within its on-disk field's width (for example because the value
        is out of range for the multiplier/offset/width in effect for
        that field).
    :raises KeyError: raised when a column label in record['Beams'] has
        no beam-array subrecord id that _beam_array_subrecord_id() can
        resolve it to, or when record['SensorSpecificID'] names a
        subrecord id with no encoder registered in
        _PING_SENSOR_SPECIFIC_CODECS.
    """
    missing = [k for k in ('PingTime', 'Longitude_deg', 'Latitude_deg', 'NumberBeams')
               if record.get(k) is None]
    if missing:
        raise ValueError(
            "record is missing required field(s): %s (see "
            "new_swath_bathymetry_ping())" % ', '.join(missing))

    g = record.get
    number_beams = int(record['NumberBeams'])
    # Empty columns are left out along with absent ones: gsflib never writes
    # a zero-length subrecord, and gsf_dec.c misreads one that is followed by
    # another subrecord.
    beams = {label: values for label, values in _present_columns(record.get('Beams')).items()
             if len(values)}

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

        subrecord_id = BEAM_ARRAY_SUBRECORD_IDS[label]
        multiplier, offset, width, signed = sf_table[subrecord_id]
        used_scale_factors[subrecord_id] = (float(multiplier), float(offset), width << 4)
        array_subrecords += _encode_ping_array(subrecord_id, values, multiplier, offset, signed, width)

    # gsf_dec.c rejects a scale factors table with no entries, so a ping
    # without scaled arrays gets no scale factors subrecord at all.
    if used_scale_factors:
        out += _encode_scale_factors(used_scale_factors)
    out += array_subrecords

    subrecord_id = record.get('SensorSpecificID')
    if subrecord_id is not None:
        if subrecord_id not in _PING_SENSOR_SPECIFIC_CODECS:
            raise KeyError("no encoder registered for sensor-specific subrecord id %d" % subrecord_id)
        _family_label, _decode_fn, encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
        sensor_specific = dict(record.get('SensorSpecific') or {})
        sensor_specific['SubrecordID'] = subrecord_id
        out += encode_fn(sensor_specific)

    intensity_record = record.get('IntensityTimeSeries')
    if intensity_record is not None:
        out += _encode_brb_intensity(intensity_record, subrecord_id)

    return out


def _gsf_major_version(version_string, default=3):
    """
    Parse the major version number out of a GSF_RECORD_HEADER version
    string, for example "GSF-v03.09" parses to 3, falling back to
    `default` if the string can't be parsed, for example because no
    header has been decoded yet.

    :param version_string: the version string to parse, as decoded from
        a GSF_RECORD_HEADER record, or None/empty if no header has been
        decoded yet.
    :param default: the major version number to return when
        `version_string` is empty or cannot be parsed.

    :return: the parsed major version number as an int, or `default`.
    """
    if version_string:
        try:
            return int(version_string.split('-v', 1)[1].split('.', 1)[0])
        except (IndexError, ValueError):
            pass
    return default


def resolve_record_type(value):
    """
    Resolve a record type given as a RecordType enum member, an int
    record id, or a name string into a RecordType enum member. Name
    strings may be given in short form (for example "COMMENT") or full
    form (for example "GSF_RECORD_COMMENT"), and are matched
    case-insensitively.

    :param value: the record type to resolve: None, a RecordType member,
        an int record id, or a name string in either short or full form.

    :return: None if `value` is None, otherwise the matching RecordType
        enum member.

    :raises ValueError: raised when `value` is a string that names no
        known record type. The error message lists every valid name.
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


def _decode_record(record_id, payload, major_version, scale_factors, decode_intensity=False):
    """
    Decode one record's payload with the decoder for its record type. The
    type -> decoder mapping is spelled out explicitly here, rather than
    looked up in a table, so it is visible in one place; this is the one
    dispatch that gsf.iter_records(), gsf.print_records(), and
    gsf.print_intensity_series() all share.

    :param record_id: the record's recordID (a RecordType value).
    :param payload: the raw bytes of the record's payload.
    :param major_version: the file's GSF major version, as returned by
        _gsf_major_version(); used only for swath bathymetry pings.
    :param scale_factors: the scale-factor dictionary carried across a
        file's swath bathymetry pings (see _decode_swath_bathymetry_ping(),
        which updates it in place); used only for those pings.
    :param decode_intensity: passed through to
        _decode_swath_bathymetry_ping().

    :return: the record as that record type's decoder returns it, a
        single dictionary in the shape of that type's new_*() template;
        or None for a recordID with no decoder.

    :raises struct.error, IndexError, ValueError: the payload is too short
        or otherwise malformed for its record type.
    """
    if record_id == RecordType.GSF_RECORD_HEADER:
        return _decode_header(payload)
    if record_id == RecordType.GSF_RECORD_SWATH_BATHY_SUMMARY:
        return _decode_swath_bathy_summary(payload)
    if record_id == RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING:
        return _decode_swath_bathymetry_ping(payload, major_version, scale_factors, decode_intensity)
    if record_id == RecordType.GSF_RECORD_SOUND_VELOCITY_PROFILE:
        return _decode_sound_velocity_profile(payload)
    if record_id in (RecordType.GSF_RECORD_PROCESSING_PARAMETERS, RecordType.GSF_RECORD_SENSOR_PARAMETERS):
        # GSF_RECORD_SENSOR_PARAMETERS: untested against a verified GSF
        # file -- no sample data containing that record type.
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


class gsf():
    """
    The main class of this library, representing one Generic Sensor Format
    (GSF) file. A single `gsf` instance is used both to read an existing
    file (indexing its records, decoding them, and printing them for
    inspection) and to write a new one (encoding and appending records),
    depending on which of its methods a caller invokes.

    This class is modeled after the `kmall` class in kmall.py: it is a
    lightweight sequential reader built on Python's `struct` module, so it
    does not require a compiled GSF library to be installed, and it builds
    a pandas-based index of the file's records to make lookups fast.
    """

    def __init__(self, filename=None, auto_scale=False):
        """
        Create a new gsf instance bound to `filename`, without opening the
        file yet. The file is only actually opened on first use, either by
        OpenFiletoRead() (called directly, or indirectly by index_file()
        and the other read-side methods) or by one of the write_* methods.

        :param filename: the path to the GSF file this instance will read
            from or write to.
        :param auto_scale: the default value used for the `auto_scale`
            parameter of write_swath_bathymetry_ping() on every call made
            through this instance, unless a given call overrides it
            explicitly. When True, write_swath_bathymetry_ping()
            automatically picks each beam array's scale factor
            (multiplier and offset) from that ping's actual data, reusing
            the previous ping's scale factor whenever it still fits
            rather than recomputing one for every ping -- see
            _pick_ping_scale_factor()'s docstring for the full algorithm,
            and the README's "Scale factors" section for a worked
            example. Defaults to False, meaning writes keep using
            DEFAULT_PING_SCALE_FACTORS unless a caller opts in.
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
        """
        Open the GSF file for reading and store the resulting file handle
        on self.FID. The file to open is self.filename if it was set when
        this instance was created; otherwise `inputfilename` is used. If
        neither is available, a message is printed and the process exits
        with status 1.

        :param inputfilename: the path to open, used only when this
            instance was created without a filename.
        """
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
        """
        Open the GSF file for writing and store the resulting file handle
        on self.FID. The file to open is self.filename if it was set when
        this instance was created; otherwise `inputfilename` is used. If
        neither is available, a message is printed and the process exits
        with status 1. The file is opened in binary write mode, so an
        existing file at that path is truncated.

        :param inputfilename: the path to open, used only when this
            instance was created without a filename.
        """
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
        """
        Close this instance's underlying file handle, if one is currently
        open. Does nothing if the file has never been opened.
        """
        if self.FID is not None:
            self.FID.close()

    ###########################################################
    # Low level record framing
    ###########################################################

    def read_record_header(self):
        """
        Read the eight-byte record framing (the four-byte size field
        followed by the four-byte data identifier word) for the record at
        the current file position, then leave the file positioned at the
        start of that record's payload, immediately after the data
        identifier word. This does not read or decode the record's
        payload itself.

        This corresponds to the framing that the reference gsflib C
        library's gsfUnpackStream() function, in gsf.c, unpacks from the
        start of each record.

        :return: None if the file is at a clean end of file, meaning no
            bytes remain to be read. Otherwise, a tuple of (dataSize,
            readSize, gsfDataID) where dataSize is the value of the
            on-disk record's "size" field, i.e. the number of bytes in
            the record's payload only. This does not include the
            four-byte checksum word even when the checksum flag is set,
            because gsfUnpackStream() reads that word separately, ahead
            of the payload. readSize is dataSize plus four additional
            bytes if the checksum flag is set, i.e. the total number of
            bytes remaining to be read or skipped, after this framing, to
            reach the start of the next record.

        :raises GSFPartialRecordAtEndOfFileError: fewer than eight bytes
            remain in the file for this record's framing, or the record's
            declared size would extend past the end of the file.
        :raises GSFRecordSizeError: the record's declared size is eight
            bytes or smaller, or larger than GSF_MAX_RECORD_SIZE.
        :raises GSFUnrecognizedRecordIDError: the record's recordID is
            not a value between 1 and NUM_REC_TYPES - 1, inclusive.
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

    def _walk_record_framing(self):
        """
        Walk every record's eight-byte framing from the start of the file,
        returning each record's byte offset, size field, and packed data
        identifier word, and validating them exactly as
        read_record_header() does. The file is memory-mapped and each
        record's framing unpacked straight from memory, following only
        the size fields; every check is then applied to all records at
        once, and the error for the first invalid record, in file order,
        is raised.

        :return: a tuple of three numpy int64 arrays, one element per
            record: (offsets, data_sizes, id_words).

        :raises GSFPartialRecordAtEndOfFileError: fewer than eight bytes
            remain for a record's framing, or a record's declared size
            would extend past the end of the file.
        :raises GSFRecordSizeError: a record's declared size is eight
            bytes or smaller, or larger than GSF_MAX_RECORD_SIZE.
        :raises GSFUnrecognizedRecordIDError: a record's recordID is not
            a value between 1 and NUM_REC_TYPES - 1, inclusive.
        """
        offsets = []
        data_sizes = []
        id_words = []
        partial_at = None
        if self.file_size:
            unpack = _RECORD_FRAMING.unpack_from
            n = self.file_size
            with mmap.mmap(self.FID.fileno(), 0, access=mmap.ACCESS_READ) as data:
                pos = 0
                while pos < n:
                    if pos + GSF_RECORD_FRAMING_SIZE > n:
                        partial_at = pos
                        break
                    size, word = unpack(data, pos)
                    offsets.append(pos)
                    data_sizes.append(size)
                    id_words.append(word)
                    pos += GSF_RECORD_FRAMING_SIZE + size + (4 if word & 0x80000000 else 0)

        offsets = np.array(offsets, dtype=np.int64)
        data_sizes = np.array(data_sizes, dtype=np.int64)
        id_words = np.array(id_words, dtype=np.int64)

        read_sizes = data_sizes + np.where(id_words & 0x80000000, 4, 0)
        record_ids = id_words & 0x003FFFFF
        bad_size = (read_sizes <= 8) | (read_sizes > GSF_MAX_RECORD_SIZE)
        bad_id = (record_ids < 1) | (record_ids >= NUM_REC_TYPES)
        past_end = offsets + GSF_RECORD_FRAMING_SIZE + read_sizes > self.file_size
        bad = np.flatnonzero(bad_size | bad_id | past_end)
        if len(bad):
            i = bad[0]
            offset = int(offsets[i])
            if bad_size[i]:
                raise GSFRecordSizeError(
                    "Record at byte offset %d in %s has an invalid size (%d bytes)"
                    % (offset, self.filename, read_sizes[i]))
            if bad_id[i]:
                raise GSFUnrecognizedRecordIDError(
                    "Record at byte offset %d in %s has an unrecognized recordID (%d)"
                    % (offset, self.filename, record_ids[i]))
            raise GSFPartialRecordAtEndOfFileError(
                "Record at byte offset %d in %s declares %d bytes, which "
                "extends past the end of the file" % (offset, self.filename, read_sizes[i]))
        if partial_at is not None:
            raise GSFPartialRecordAtEndOfFileError(
                "Partial record header at byte offset %d in %s" % (partial_at, self.filename))

        return offsets, data_sizes, id_words

    def index_file(self):
        """
        Index this instance's GSF file: walk every record in the file
        from the beginning, recording each record's type, byte offset,
        and size, without decoding any record's payload. If the file is
        not already open, this opens it for reading; if it is already
        open, this closes and reopens it first, to force any pending
        writes to be flushed and to restart from the beginning of the
        file.

        Builds self.Index, a pandas DataFrame with one row per record and
        the columns RecordType, RecordID, ByteOffset, RecordSize,
        TotalBytes, and ChecksumFlag. Every other read-side method on this
        class relies on self.Index for fast lookups. This mirrors
        kmall.py's index_file().
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

        offsets, data_sizes, id_words = self._walk_record_framing()

        checksum = (id_words & 0x80000000) != 0
        record_ids = (id_words & 0x003FFFFF).astype(np.int64)
        read_sizes = data_sizes + np.where(checksum, 4, 0)
        total = GSF_RECORD_FRAMING_SIZE + read_sizes

        header_rows = np.flatnonzero(record_ids == RecordType.GSF_RECORD_HEADER)
        if len(header_rows):
            first = header_rows[0]
            start = int(offsets[first]) + GSF_RECORD_FRAMING_SIZE + (4 if checksum[first] else 0)
            self.FID.seek(start)
            version_bytes = self.FID.read(min(GSF_VERSION_SIZE, int(data_sizes[first])))
            self.gsfVersion = version_bytes.split(b'\x00', 1)[0].decode('ascii', 'replace')

        self.Index = pd.DataFrame({
            'RecordType': _RECORD_TYPE_NAMES[record_ids],
            'RecordID': record_ids,
            'ByteOffset': offsets,
            'RecordSize': data_sizes,
            'TotalBytes': total,
            'ChecksumFlag': checksum,
        })
        self.Index['RecordType'] = self.Index['RecordType'].astype('category')

        if self.verbose:
            for name, offset, size in zip(self.Index['RecordType'], offsets.tolist(), total.tolist()):
                print("RECORD_TYPE: %s,\tOFFSET: %d,\tSIZE: %d" % (name, offset, size))

        unreadBytes = self.file_size - int(self.Index['TotalBytes'].sum())
        if unreadBytes != 0:
            print()
            print("   *** WARNING! %d bytes were not accounted for in this file! ***" % unreadBytes)
            print()

        if self.verbose >= 2:
            print(self.Index)

    ###########################################################
    # Whole-file readers
    ###########################################################

    def iter_records(self, record_type=None, decode_intensity=False):
        """
        Iterate over the file's records in file order, decoding each one,
        optionally restricted to a single record type. Records are located
        through self.Index (the file is indexed first if it has not been
        yet), so each record is read with a single seek and read, and
        records of other types are skipped without being read at all. Swath
        bathymetry pings are decoded with the file's own GSF major version
        and with scale factors carried from one ping to the next, as
        _decode_swath_bathymetry_ping() requires. Index the file again
        (index_file()) after writing to it, so the index covers the new
        records.

        To read every attitude measurement in a file, read_attitude() is
        far faster than iterating over its attitude records one by one.

        :param record_type: an optional record type to restrict the
            iteration to: a RecordType, its integer recordID, or a name
            string, either short (e.g. "SWATH_BATHYMETRY_PING") or full
            (e.g. "GSF_RECORD_SWATH_BATHYMETRY_PING"), matched
            case-insensitively. Every record is returned if this is
            omitted.
        :param decode_intensity: if True, each swath bathymetry ping's
            intensity time series is decoded too (see
            _decode_swath_bathymetry_ping()).

        :return: a generator of (record_type, byte_offset, record) tuples:
            the record's RecordType, the byte offset of its framing within
            the file, and the record as decoded by that type's decoder: a
            single dictionary in the shape of that type's new_*()
            template, which can be passed straight back to the matching
            write_*() method.

        :raises ValueError: record_type is a string that does not name any
            known record type.
        :raises struct.error, IndexError, ValueError: a record's payload is
            too short or otherwise malformed for its record type.
        """
        record_type = resolve_record_type(record_type)
        if self.Index is None:
            self.index_file()
        if self.FID is None or self.FID.closed:
            self.OpenFiletoRead()
        index = self.Index
        if record_type is not None:
            index = index[index['RecordID'] == record_type]
        major_version = _gsf_major_version(self.gsfVersion)
        scale_factors = {}
        fid = self.FID
        for record_id, offset, size, checksum in zip(
                index['RecordID'].tolist(), index['ByteOffset'].tolist(),
                index['RecordSize'].tolist(), index['ChecksumFlag'].tolist()):
            fid.seek(offset + GSF_RECORD_FRAMING_SIZE + (4 if checksum else 0))
            record = _decode_record(record_id, fid.read(size), major_version, scale_factors, decode_intensity)
            yield RecordType(record_id), offset, record

    #: At most this many attitude measurements are gathered from the file
    #: per vectorized step in read_attitude(), bounding its temporary index
    #: arrays on very large files.
    _ATTITUDE_GATHER_CHUNK = 1_000_000

    def read_attitude(self):
        """
        Read every GSF_RECORD_ATTITUDE record in the file at once, and
        return all of their measurements together, in file order, as one
        dictionary of numpy arrays: the same shape _decode_attitude()
        returns for a single record, but covering the whole file. This
        indexes the file first if it has not been indexed yet.

        Rather than decoding one record at a time, this memory-maps the
        file and gathers every attitude record's header, and then every
        measurement, with vectorized indexing, so its cost barely depends
        on how many records the measurements are spread across: files
        that store one measurement per record (as some writers do at
        100 Hz) read as quickly as files that batch them.

        :return: a dictionary, in the shape new_attitude() creates, with
            'NumMeasurements' (the total over all attitude records),
            'Time' (a numpy datetime64[ns] array of each measurement's
            time, in UTC), and 'Pitch_deg', 'Roll_deg', 'Heave_m', and
            'Heading_deg' (each a float64 numpy array), one element per
            measurement.

        :raises ValueError: an attitude record's payload is too short for
            the number of measurements it declares.
        """
        if self.Index is None:
            self.index_file()
        rows = self.Index[self.Index['RecordID'] == RecordType.GSF_RECORD_ATTITUDE]
        payload_starts = (rows['ByteOffset'].to_numpy() + GSF_RECORD_FRAMING_SIZE
                          + np.where(rows['ChecksumFlag'].to_numpy(), 4, 0))
        payload_sizes = rows['RecordSize'].to_numpy()

        record = new_attitude()
        if len(rows) == 0:
            return record

        header_size = _ATTITUDE_HEADER_DTYPE.itemsize
        measurement_size = _ATTITUDE_MEASUREMENT_DTYPE.itemsize
        with mmap.mmap(self.FID.fileno(), 0, access=mmap.ACCESS_READ) as data:
            file_bytes = np.frombuffer(data, dtype=np.uint8)
            try:
                headers = file_bytes[payload_starts[:, None] + np.arange(header_size)] \
                    .view(_ATTITUDE_HEADER_DTYPE).reshape(-1)
                counts = headers['num_measurements'].astype(np.int64)
                short = np.flatnonzero(payload_sizes < header_size + counts * measurement_size)
                if len(short):
                    i = short[0]
                    raise ValueError(
                        "attitude record at byte offset %d in %s declares %d measurements "
                        "but holds only %d payload bytes"
                        % (rows['ByteOffset'].iloc[i], self.filename, counts[i], payload_sizes[i]))

                total = int(counts.sum())
                # Each measurement's byte offset in the file: its record's
                # first measurement, plus 10 bytes per earlier measurement in
                # the same record.
                first_in_record = np.cumsum(counts) - counts
                record_of = np.repeat(np.arange(len(counts)), counts)
                measurement_starts = (payload_starts + header_size)[record_of] \
                    + (np.arange(total) - first_in_record[record_of]) * measurement_size

                measurements = np.empty(total, dtype=_ATTITUDE_MEASUREMENT_DTYPE)
                for lo in range(0, total, self._ATTITUDE_GATHER_CHUNK):
                    hi = min(lo + self._ATTITUDE_GATHER_CHUNK, total)
                    measurements[lo:hi] = file_bytes[
                        measurement_starts[lo:hi, None] + np.arange(measurement_size)] \
                        .view(_ATTITUDE_MEASUREMENT_DTYPE).reshape(-1)
            finally:
                # Release the view before the mapping closes.
                del file_bytes

        base_ns = headers['sec'].astype(np.int64) * 1_000_000_000 + headers['nsec']
        time_ns = measurements['time_offset_ms'] * np.int64(1_000_000)
        time_ns += base_ns[record_of]
        record['NumMeasurements'] = total
        record['Time'] = time_ns.view('datetime64[ns]')
        record['Pitch_deg'] = measurements['pitch'] / 100.0
        record['Roll_deg'] = measurements['roll'] / 100.0
        record['Heave_m'] = measurements['heave'] / 100.0
        record['Heading_deg'] = measurements['heading'] / 100.0
        return record

    ###########################################################
    # Reporting
    ###########################################################

    def report_record_types(self):
        """
        Print, and also return, a summary of this file's index: for each
        record type present in the file, how many records of that type
        there are and how many bytes they occupy in total, as a minimum,
        and as a maximum, together with what percentage of the total file
        size that record type accounts for. Indexes the file first, by
        calling index_file(), if it has not already been indexed. This
        mirrors kmall.py's report_packet_types(), and is what the -V
        command line flag invokes.

        :return: summary, a pandas DataFrame with one row per record type
            present in the file (sorted by total bytes, descending) and
            the columns Count, Total Bytes, Min Bytes, Max Bytes, and %
            of File.
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
    def _table_to_string(table, index_name):
        """
        A print_records() helper that renders a table (a dictionary of
        column name -> equal-length array) as aligned text, one row per
        element, via a pandas.DataFrame built only for display.

        :param table: the table to render.
        :param index_name: the label for the row-number column.

        :return: the rendered table, as a string.
        """
        frame = pd.DataFrame(table)
        frame.index.name = index_name
        return frame.to_string()

    @staticmethod
    def _print_array_record(record, table_label, index_name):
        """
        A print_records() helper that prints one decoded record other
        than a ping: each scalar field as a "key : value" line, then any
        equal-length numpy arrays (an attitude record's measurements, or a
        sound velocity profile's points) together as one table, one row
        per element. The table is built only here, for display; the
        decoders themselves return plain numpy arrays.

        :param record: the decoded record dictionary.
        :param table_label: the heading printed above the table, or None
            for a record with no arrays.
        :param index_name: the label for the table's row-number column, or
            None for a record with no arrays.
        """
        scalars = {k: v for k, v in record.items() if not isinstance(v, np.ndarray)}
        table = {k: v for k, v in record.items() if isinstance(v, np.ndarray)}
        if scalars:
            width = max(len(k) for k in scalars)
            for k, v in scalars.items():
                print("  %-*s : %s" % (width, k, v))
        if _table_length(table):
            print("-- %s --" % table_label)
            print(gsf._table_to_string(table, index_name))

    @staticmethod
    def _print_ping_record(record):
        """
        A print_records() helper that prints one decoded ping record to
        stdout as readable text. It handles the two record types
        (GSF_RECORD_SWATH_BATHYMETRY_PING and GSF_RECORD_SINGLE_BEAM_PING)
        whose decoded dictionaries nest tables and sensor-specific
        subrecords, which _print_array_record() does not handle.

        This prints, in order: every flat scalar field in `record` (i.e.
        every entry other than 'Beams', 'IntensityTimeSeries',
        'SensorSpecificID', 'SensorSpecific', and 'Notes') as a "key :
        value" line; then each string in `record['Notes']`, if present;
        then `record['Beams']`, if present and non-empty, as a table with
        one row per beam; then `record['IntensityTimeSeries']`, if
        present, as its own scalar fields (the sample encoding and any
        vendor imagery-preamble fields) followed by its nested 'Beams'
        table, one row per beam's backscatter sample series; and finally,
        if `record['SensorSpecific']` is present, that vendor
        sensor-specific subrecord's own scalar fields under a header
        naming its resolved family (looked up from
        `record['SensorSpecificID']` via
        _SENSOR_SPECIFIC_SUBRECORD_NAMES), followed by any of its own
        per-element tables.

        :param record: a decoded ping record, in the merged-dictionary
            shape returned by _decode_swath_bathymetry_ping() or
            _decode_single_beam_ping().
        """
        flat = {k: v for k, v in record.items()
                if k not in ('Beams', 'IntensityTimeSeries', 'SensorSpecificID', 'SensorSpecific', 'Notes')}
        if flat:
            width = max(len(k) for k in flat)
            for k, v in flat.items():
                print("  %-*s : %s" % (width, k, v))
        for note in record.get('Notes', []):
            print("  # %s" % note)
        # Columns with one value per beam print as one table. A column that
        # is present but has some other length (an empty array from a
        # zero-length subrecord, or beam flags of an unexpected size)
        # cannot share that table, so each one is noted on its own line.
        beams = _present_columns(record.get('Beams'))
        number_beams = record.get('NumberBeams')
        per_beam = {k: v for k, v in beams.items() if len(v) == number_beams}
        for label, values in beams.items():
            if label not in per_beam:
                print("  # %s: %d values for %s beams" % (label, len(values), number_beams))
        if _table_length(per_beam):
            print("-- Beams --")
            print(gsf._table_to_string(per_beam, 'Beam'))
        intensity_record = record.get('IntensityTimeSeries')
        if intensity_record:
            scalar_items = {k: v for k, v in intensity_record.items() if k not in ('Beams', 'Samples')}
            if scalar_items:
                print("-- IntensityTimeSeries --")
                width = max(len(k) for k in scalar_items)
                for k, v in scalar_items.items():
                    print("  %-*s : %s" % (width, k, v))
            intensity_beams = intensity_record.get('Beams')
            if _table_length(intensity_beams):
                # Shown with each beam's own samples alongside its row.
                display = dict(intensity_beams)
                display['Samples'] = [s.tolist() for s in _split_intensity_samples(intensity_record)]
                print("-- IntensityTimeSeries.Beams --")
                print(gsf._table_to_string(display, 'Beam'))
        sensor_specific = record.get('SensorSpecific')
        if sensor_specific:
            sensor_id = record.get('SensorSpecificID')
            family_name = _SENSOR_SPECIFIC_SUBRECORD_NAMES.get(sensor_id, str(sensor_id))
            scalar_items = {k: v for k, v in sensor_specific.items() if not isinstance(v, dict)}
            if scalar_items:
                print("-- SensorSpecific (%s, id=%s) --" % (family_name, sensor_id))
                width = max(len(k) for k in scalar_items)
                for k, v in scalar_items.items():
                    print("  %-*s : %s" % (width, k, v))
            for k, v in sensor_specific.items():
                if isinstance(v, dict) and _table_length(v):
                    print("-- %s --" % k)
                    print(gsf._table_to_string(v, k))

    def print_records(self, record_type=None):
        """
        A debugging utility that walks the file from the beginning and
        prints each record to stdout as readable text. This is what the
        -p command line flag invokes. If the file is not already open,
        this opens it for reading; if it is already open, this closes and
        reopens it first, to force any pending writes to be flushed and
        to restart from the beginning of the file.

        For every record type that has a field-level decoder available
        (currently every record type except the obsolete or rare ones,
        and the beam-array subrecords, noted in the "Field-level record
        decoding" section above), this prints the record's scalar fields
        as "key : value" lines, and prints any per-beam, per-point, or
        per-measurement data as a table with one row per beam, point, or
        measurement. The two ping record types are printed by
        _print_ping_record(), since their decoders return a different
        shape from every other record type's decoder. A record type with
        no available decoder falls back to printing its raw payload as
        ASCII text, with each non-printable byte shown as a '.'.

        :param record_type: an optional record type to restrict the
            output to: a RecordType, its integer recordID, or a name
            string, either short (e.g. "COMMENT") or full (e.g.
            "GSF_RECORD_COMMENT"), matched case-insensitively. Every
            record in the file is printed if this is omitted.

        :raises ValueError: `record_type` is a string that does not name
            any known record type.
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
                    rid = data_id.recordID
                    try:
                        decoded = _decode_record(rid, payload, major_version, scale_factors)
                    except (struct.error, IndexError, ValueError) as exc:
                        print("  # decode failed (%s); showing raw text" % exc)
                    if decoded is None:
                        text = ''.join(chr(b) if 32 <= b < 127 else '.' for b in payload)
                        print(text)
                    elif rid in (RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING,
                                 RecordType.GSF_RECORD_SINGLE_BEAM_PING):
                        self._print_ping_record(decoded)
                    elif rid == RecordType.GSF_RECORD_ATTITUDE:
                        self._print_array_record(decoded, 'Measurements', 'Measurement')
                    elif rid == RecordType.GSF_RECORD_SOUND_VELOCITY_PROFILE:
                        self._print_array_record(decoded, 'Profile', 'Point')
                    elif rid in (RecordType.GSF_RECORD_PROCESSING_PARAMETERS,
                                 RecordType.GSF_RECORD_SENSOR_PARAMETERS):
                        # Each parameter on its own "NAME : VALUE" line.
                        flat = {'ParamTime': decoded['ParamTime']}
                        flat.update((k, '' if v is None else v) for k, v in decoded['Parameters'].items())
                        self._print_array_record(flat, None, None)
                    else:
                        self._print_array_record(decoded, None, None)
                    print()

            self.FID.seek(offset + GSF_RECORD_FRAMING_SIZE + readSize, 0)

    def print_intensity_series(self):
        """
        A debugging utility that walks the file's
        GSF_RECORD_SWATH_BATHYMETRY_PING records and prints each ping's
        per-beam backscatter intensity time series to stdout as CSV. This
        is what the -I command line flag invokes. If the file is not
        already open, this opens it for reading; if it is already open,
        this closes and reopens it first, to force any pending writes to
        be flushed and to restart from the beginning of the file.

        For each ping that carries one, this decodes the
        GSF_SWATH_BATHY_SUBRECORD_INTENSITY_SERIES_ARRAY subrecord (via
        _decode_brb_intensity()) and prints one CSV row per beam, giving
        the beam index, the sample count, the bottom-detect sample index,
        the start-range sample index, and then every raw sample value in
        that beam's time series.

        This subrecord is decoded for every sensor family gsflib defines
        an imagery-specific preamble for (KMALL, the EM3 series, the EM4
        series, the Reson 7125/T-series/8100 family, the Klein 5410 BSS,
        and R2Sonic), as well as for every sensor that has no such
        preamble at all. A ping that carries no intensity series
        subrecord, or whose bits-per-sample value does not resolve to a
        supported sample width, is noted and skipped rather than guessed
        at.
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
                        record = _decode_record(
                            data_id.recordID, payload, major_version, scale_factors, decode_intensity=True)
                    except (struct.error, IndexError, ValueError) as exc:
                        print("# ping offset=%d: decode failed (%s)" % (offset, exc))
                    else:
                        intensity_record = record.get('IntensityTimeSeries')
                        if intensity_record is None:
                            note = next((n for n in record.get('Notes', []) if 'IntensityTimeSeries' in n), None)
                            print("# ping offset=%d ping_time=%s: %s" %
                                  (offset, record.get('PingTime', '?'),
                                   note or "no intensity time series subrecord"))
                        else:
                            print("# ping offset=%d ping_time=%s" % (offset, record.get('PingTime', '?')))
                            print("# Beam,SampleCount,DetectSample,StartRangeSamples,Sample0,Sample1,...")
                            counts, start_ranges, detects = _intensity_beam_layout(intensity_record)
                            for beam, (count, detect, start_range, samples) in enumerate(zip(
                                    counts.tolist(), detects.tolist(), start_ranges.tolist(),
                                    _split_intensity_samples(intensity_record))):
                                fields = [str(beam), str(count), str(detect), str(start_range)]
                                fields.extend(str(v) for v in samples.tolist())
                                print(",".join(fields))

            self.FID.seek(offset + GSF_RECORD_FRAMING_SIZE + readSize, 0)

    ###########################################################
    # Writing
    ###########################################################

    def write_record(self, record_id, payload, checksum=False):
        """
        Write one complete, framed GSF record to the file, appending it
        after whatever has already been written: a four-byte size field,
        then a four-byte data identifier word (optionally preceded, within
        that same field's byte count, by a four-byte checksum), then the
        payload itself. If the file is not already open for writing, this
        opens it first, via OpenFiletoWrite(). This is the single
        low-level primitive that every write_* method on this class uses
        to actually append a record's bytes to the file, and is the
        inverse of read_record_header() together with the payload reads
        performed elsewhere in this class. This corresponds to the
        framing that the reference gsflib C library's gsfWrite() and
        gsfPackStream() functions, in gsf.c, write to a record's start,
        using the same encoding documented in the gsfDataID class
        docstring above.

        :param record_id: the type of record being written, as a
            RecordType or as its integer recordID.
        :param payload: the fully assembled record body, as bytes, such
            as the bytes returned by one of this file's _encode_*
            functions.
        :param checksum: if True, the four-byte checksum computed by
            gsf_checksum(payload) is written immediately before `payload`,
            and the checksum flag bit is set in the data identifier word;
            if False, no checksum is written and the flag bit is left
            unset.
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

    def write_header(self, record=None):
        """
        Write the GSF_RECORD_HEADER record, which is normally the first
        record in any file this library writes. The version string that
        is encoded into the record is also stored on the instance (as
        self.gsfVersion), so that write_swath_bathymetry_ping() can later
        decide, based on that version, whether to include fields that
        only exist in newer GSF versions. See _encode_header() for how
        the version string is encoded. The reference gsflib C library's
        own encoder always stamps its own current version when it writes
        this record, so a record's 'Version' exists mainly to support
        testing.

        :param record: a dictionary in the shape new_header() creates.
            Optional; omitted, or with 'Version' None, this library's own
            current GSF_VERSION is written.
        """
        version = (record or {}).get('Version') or GSF_VERSION
        self.write_record(RecordType.GSF_RECORD_HEADER, _encode_header({'Version': version}))
        self.gsfVersion = version

    def write_processing_parameters(self, record):
        """
        Write a GSF_RECORD_PROCESSING_PARAMETERS record, which records
        the processing parameters that were in effect when this file's
        data was generated, as a set of named text values. See
        _encode_name_value_parameters() for how the record is encoded
        onto the wire.

        :param record: a dictionary in the shape new_name_value_parameters()
            creates: 'ParamTime', the timestamp associated with this set
            of parameters, and 'Parameters', a dictionary mapping
            parameter name strings to their values.
        """
        self.write_record(
            RecordType.GSF_RECORD_PROCESSING_PARAMETERS, _encode_name_value_parameters(record))

    def write_sensor_parameters(self, record):
        """
        Write a GSF_RECORD_SENSOR_PARAMETERS record, which records a set
        of sensor configuration parameters as named text values, using
        the same wire format as GSF_RECORD_PROCESSING_PARAMETERS. See
        _encode_name_value_parameters() for how the record is encoded
        onto the wire.

        This method is untested against a verified GSF file, since no
        sample data containing a GSF_RECORD_SENSOR_PARAMETERS record is
        available; unlike write_processing_parameters(), whose record
        type does appear in sample data, this method's round-trip test
        checks only that writing and then reading back this method's own
        output is self-consistent.

        :param record: a dictionary in the shape new_name_value_parameters()
            creates: 'ParamTime', the timestamp associated with this set
            of parameters, and 'Parameters', a dictionary mapping
            parameter name strings to their values.
        """
        self.write_record(
            RecordType.GSF_RECORD_SENSOR_PARAMETERS, _encode_name_value_parameters(record))

    def write_sound_velocity_profile(self, record):
        """
        Write a GSF_RECORD_SOUND_VELOCITY_PROFILE record, which records a
        sound speed profile (a series of depth/sound-speed pairs) along
        with the position and times associated with it. See
        _encode_sound_velocity_profile() for the fields the record needs
        and how they are encoded onto the wire.

        :param record: a dictionary in the shape
            new_sound_velocity_profile() creates or
            _decode_sound_velocity_profile() returns.
        """
        self.write_record(
            RecordType.GSF_RECORD_SOUND_VELOCITY_PROFILE, _encode_sound_velocity_profile(record))

    def write_attitude(self, record):
        """
        Write a GSF_RECORD_ATTITUDE record, which records one or more
        vessel attitude measurements (pitch, roll, heave, and heading).
        See _encode_attitude() for the fields the record needs and how
        they are encoded onto the wire.

        :param record: a dictionary in the shape new_attitude() creates or
            _decode_attitude() returns: 'Time', 'Pitch_deg', 'Roll_deg',
            'Heave_m', and 'Heading_deg', each an array-like with one
            element per measurement.
        """
        self.write_record(RecordType.GSF_RECORD_ATTITUDE, _encode_attitude(record))

    def write_swath_bathymetry_ping(self, record, scale_factors=None, auto_scale=None):
        """
        Write a GSF_RECORD_SWATH_BATHYMETRY_PING record, which holds one
        multibeam ping's worth of beam arrays (depth, across-track
        distance, beam flags, and so on) along with the ping's sensor-
        specific metadata. See _encode_swath_bathymetry_ping() for the
        expected shape of `record` and `scale_factors`, which is the same
        dict shape _decode_swath_bathymetry_ping() returns, so a ping
        decoded from another file can be passed straight back in. This
        method uses self.gsfVersion (set by write_header(), which must be
        called first) to decide whether to include the height, estimated
        surface error, and GPS tide corrector fields, which only exist
        when the file's major version number is greater than 2 -- true
        for every GSF_VERSION this codebase writes.

        When `auto_scale` resolves to True, `scale_factors` is computed
        automatically for every beam array present in `record['Beams']`,
        instead of falling back to the static defaults in
        DEFAULT_PING_SCALE_FACTORS. Each array's (multiplier, offset)
        pair is chosen by _pick_ping_scale_factor() from that array's
        actual values in this ping, reusing whatever scale factor was
        already active for that subrecord id on this gsf instance (held
        in self._auto_scale_factors) whenever it still fits that data, so
        that a file's scale factors only change when the data actually
        requires it, rather than on every single ping. BeamFlags and
        QualityFlags are excluded from this automatic selection, since
        those arrays are not scaled at all. See the README's "Scale
        factors" section for a worked example of when and why this
        differs from the static defaults, and _pick_ping_scale_factor()'s
        own docstring for the full selection algorithm. Passing an
        explicit `scale_factors` override together with `auto_scale=True`
        is not supported, since the two are two different answers to the
        same question of what scale factors this ping should use.

        :param record: the ping to write, as a dictionary in the same
            shape _decode_swath_bathymetry_ping() returns, including its
            'Beams' entry (a dictionary of beam arrays).
        :param scale_factors: an explicit override of the (multiplier,
            offset, width, signed) scale factor to use for each beam
            array's subrecord id, in the same shape
            _encode_swath_bathymetry_ping() expects. If left as None and
            `auto_scale` does not resolve to True, DEFAULT_PING_SCALE_FACTORS
            is used instead.
        :param auto_scale: if True, or if left as None while
            self.auto_scale is True, `scale_factors` is computed
            automatically for this ping as described above, instead of
            using an explicit override or the static defaults.

        :raises ValueError: `auto_scale` resolves to True while
            `scale_factors` is also given explicitly, since the two
            cannot both be honored at once.
        """
        if auto_scale is None:
            auto_scale = self.auto_scale
        if auto_scale and scale_factors is not None:
            raise ValueError(
                "auto_scale=True and an explicit scale_factors= override are mutually exclusive")

        beams = _present_columns(record.get('Beams'))

        if auto_scale:
            # Resolve one (multiplier, offset, width, signed) scale
            # factor per beam array actually present in this ping,
            # building the same shape of dict write_swath_bathymetry_ping()
            # would otherwise accept as an explicit scale_factors=
            # override -- BeamFlags/QualityFlags are excluded since they
            # aren't scaled at all (see _beam_array_subrecord_id()), and
            # so are empty arrays, which are written without a scale
            # factor.
            scale_factors = {}
            for label in beams:
                if label in ('BeamFlags', 'QualityFlags') or len(beams[label]) == 0:
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

    def write_swath_bathy_summary(self, record):
        """
        Write a GSF_RECORD_SWATH_BATHY_SUMMARY record, which summarizes
        the time span, geographic bounding box, and depth range covered
        by a collection of swath bathymetry pings. See
        _encode_swath_bathy_summary() for the fields the record needs and
        how they are encoded onto the wire.

        This method is untested against a verified GSF file, since no
        sample data containing a GSF_RECORD_SWATH_BATHY_SUMMARY record is
        available.

        :param record: a dictionary in the shape new_swath_bathy_summary()
            creates or _decode_swath_bathy_summary() returns.
        """
        self.write_record(RecordType.GSF_RECORD_SWATH_BATHY_SUMMARY, _encode_swath_bathy_summary(record))

    def write_comment(self, record):
        """
        Write a GSF_RECORD_COMMENT record, which holds a free-text
        comment string along with a timestamp. See _encode_comment() for
        the fields the record needs and how they are encoded onto the
        wire.

        This method is untested against a verified GSF file, since no
        sample data containing a GSF_RECORD_COMMENT record is available.

        :param record: a dictionary in the shape new_comment() creates or
            _decode_comment() returns.
        """
        self.write_record(RecordType.GSF_RECORD_COMMENT, _encode_comment(record))

    def write_history(self, record):
        """
        Write a GSF_RECORD_HISTORY record, which logs one processing
        step applied to the file: who ran it, on what host, with what
        command line, and any free-text comment about it. See
        _encode_history() for the fields the record needs and how they
        are encoded onto the wire.

        This method is untested against a verified GSF file, since no
        sample data containing a GSF_RECORD_HISTORY record is available.

        :param record: a dictionary in the shape new_history() creates or
            _decode_history() returns.
        """
        self.write_record(RecordType.GSF_RECORD_HISTORY, _encode_history(record))

    def write_navigation_error(self, record):
        """
        Write a GSF_RECORD_NAVIGATION_ERROR record, which records the
        estimated longitude and latitude error associated with a ping.
        This record type is obsolete in the GSF format; prefer
        write_hv_navigation_error() for new data. See
        _encode_navigation_error() for the fields the record needs and how
        they are encoded onto the wire.

        This method is untested against a verified GSF file, since no
        sample data containing a GSF_RECORD_NAVIGATION_ERROR record is
        available.

        :param record: a dictionary in the shape new_navigation_error()
            creates or _decode_navigation_error() returns.
        """
        self.write_record(RecordType.GSF_RECORD_NAVIGATION_ERROR, _encode_navigation_error(record))

    def write_hv_navigation_error(self, record):
        """
        Write a GSF_RECORD_HV_NAVIGATION_ERROR record, which records the
        estimated horizontal and vertical positioning error associated
        with a ping, superseding the older GSF_RECORD_NAVIGATION_ERROR
        record type. See _encode_hv_navigation_error() for the fields the
        record needs and how they are encoded onto the wire.

        This method is untested against a verified GSF file, since no
        sample data containing a GSF_RECORD_HV_NAVIGATION_ERROR record is
        available.

        :param record: a dictionary in the shape new_hv_navigation_error()
            creates or _decode_hv_navigation_error() returns.
        """
        self.write_record(RecordType.GSF_RECORD_HV_NAVIGATION_ERROR, _encode_hv_navigation_error(record))

    def write_single_beam_ping(self, record):
        """
        Write a GSF_RECORD_SINGLE_BEAM_PING record, which holds one
        single-beam echosounder ping's worth of depth and sensor-specific
        metadata. See _encode_single_beam_ping() for the expected shape
        of `record`, which is the same dict shape
        _decode_single_beam_ping() returns, so a ping decoded from
        another file can be passed straight back in.

        This method is untested against a verified GSF file, since no
        sample data containing a GSF_RECORD_SINGLE_BEAM_PING record is
        available.

        :param record: the ping to write, as a dictionary in the same
            shape _decode_single_beam_ping() returns.
        """
        self.write_record(
            RecordType.GSF_RECORD_SINGLE_BEAM_PING,
            _encode_single_beam_ping(record))


###########################################################
# Command line interface
###########################################################

def _record_type_help_text():
    """
    Build a reference listing of every GSF_RECORD_* type, in its short
    form (with the "GSF_RECORD_" prefix removed), alongside its
    description as looked up in RECORD_TYPE_DESCRIPTIONS. The resulting
    text is used as the epilog of the command line tool's help output,
    both as the reference for what values the -p argument accepts and as
    part of what -h prints.

    :return: the formatted, multi-line reference text, as a string.
    """
    lines = ["record types (short or full name accepted for -p, e.g. -p COMMENT):", ""]
    short_names = [rt.name.replace('GSF_RECORD_', '') for rt in RecordType]
    width = max(len(name) for name in short_names)
    for rt, short in zip(RecordType, short_names):
        lines.append("  %-*s  %s" % (width, short, RECORD_TYPE_DESCRIPTIONS.get(rt, "")))
    return "\n".join(lines)


def main(args=None):
    """
    Command line entry point for the gsfu.py script, and the function
    setup.py's console_scripts entry point invokes. It parses command
    line arguments and, depending on which flags are given, either
    prints usage and exits (no -f filename given), prints each ping's
    per-beam backscatter intensity time series as CSV (-I), prints
    records to stdout as ASCII text for debugging, optionally restricted
    to one record type (-p), or indexes the file and prints either a
    summary of its record types (-V) or a plain record count.

    :param args: the command line arguments to parse, as a list of
        strings (not including the program name), such as
        sys.argv[1:]. If left as None, sys.argv[1:] is used.

    :return: an integer process exit code: 0 on success, or 1 if no
        filename was given, or if printing records failed with a
        ValueError.
    """
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
