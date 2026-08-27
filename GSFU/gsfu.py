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
"""
import argparse
import os
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum

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
# Command line interface
###########################################################

def main(args=None):
    """ Command line script entry point. """
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="A python script (and class) for indexing, reading, "
                     "and writing Generic Sensor Format (GSF) data files.")
    parser.add_argument('-f', action='store', dest='gsf_filename',
                         help="The path and filename to parse.")
    parser.add_argument('-V', action='store_true', dest='verify',
                         default=False,
                         help="Index the file and print a summary of its record types "
                              "(count and bytes consumed by each).")
    parser.add_argument('-v', action='count', dest='verbose', default=0,
                         help="Increasingly verbose output (e.g. -v -vv), for debugging use -vv")
    parsed = parser.parse_args(args)

    if parsed.gsf_filename is None:
        parser.print_help()
        return 1

    G = gsf(parsed.gsf_filename)
    G.verbose = parsed.verbose
    G.index_file()

    if parsed.verify:
        G.report_record_types()
    else:
        print("Indexed %d records in %s" % (len(G.Index), parsed.gsf_filename))

    return 0


if __name__ == '__main__':
    sys.exit(main())
