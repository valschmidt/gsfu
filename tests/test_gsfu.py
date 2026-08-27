#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for GSFU.gsfu.

Two kinds of coverage:

  * Synthetic-record tests build minimal, hand-crafted GSF byte streams
    (using BytesIO-backed files) to independently verify the record framing
    math (size field / data identifier bit-packing / checksum handling /
    error conditions) against the encoding documented in gsflib's gsf.h and
    gsf.c (gsfUnpackStream), without depending on any real data file.

  * Real-file tests run against the sample .gsf files in ../data/GSF (real
    Kongsberg EM712 GSF files) and check invariants that must hold for any
    valid GSF file: every byte in the file is accounted for by exactly one
    record, the first record is always GSF_RECORD_HEADER, and every
    recordID seen is a recognized GSF_RECORD_* type.
"""
import struct
from pathlib import Path

import pytest

from GSFU.gsfu import (
    GSF_RECORD_FRAMING_SIZE,
    GSF_VERSION_SIZE,
    GSFPartialRecordAtEndOfFileError,
    GSFRecordSizeError,
    GSFUnrecognizedRecordIDError,
    NUM_REC_TYPES,
    RecordType,
    gsf,
    gsf_checksum,
    main,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "GSF"
SAMPLE_FILES = sorted(DATA_DIR.glob("*.gsf")) if DATA_DIR.is_dir() else []

# Use the smallest sample file for tests that need real data but should stay
# fast; the full set is exercised (offset accounting only) in a parametrized
# test below.
SMALL_SAMPLE = min(SAMPLE_FILES, key=lambda p: p.stat().st_size) if SAMPLE_FILES else None

requires_sample_data = pytest.mark.skipif(
    not SAMPLE_FILES, reason="no sample .gsf files found in data/GSF")


# ---------------------------------------------------------------------------
# Helpers for building synthetic GSF byte streams
# ---------------------------------------------------------------------------

def _pack_record(record_id, payload, checksum_flag=False, reserved=0):
    """
    Build the on-disk bytes for a single GSF record, per gsf.c's
    gsfUnpackStream() framing: [4-byte size][4-byte data ID][payload],
    where payload is prefixed with a 4-byte checksum if checksum_flag is
    set.

    Per gsf.c: the on-disk "size" field holds only len(payload) -- it does
    NOT include the 4-byte checksum word. gsfUnpackStream() separately adds
    4 to dataSize (as "readSize") to account for the checksum when the
    checksum flag is set.
    """
    did = (int(checksum_flag) << 31) | ((reserved & 0x1FF) << 22) | (record_id & 0x003FFFFF)
    size = len(payload)

    if checksum_flag:
        checksum = gsf_checksum(payload)
        body = struct.pack('>I', checksum) + payload
    else:
        body = payload

    return struct.pack('>II', size, did) + body


def _write_records(path, records):
    with open(path, "wb") as f:
        for rec in records:
            f.write(rec)


def _header_record():
    """ A minimal, valid GSF_RECORD_HEADER record. """
    version = b"GSF-v03.09\x00\x00"
    assert len(version) == GSF_VERSION_SIZE
    return _pack_record(RecordType.GSF_RECORD_HEADER, version)


def _comment_record(text=b"hello world"):
    """
    A minimal GSF_RECORD_COMMENT-ish payload (contents don't matter to the
    indexer, which does not decode payloads).

    Per gsf.c's gsfUnpackStream(), readSize (== dataSize when unchecksummed)
    must be > 8 bytes or the record is rejected as GSF_RECORD_SIZE_ERROR;
    real gsfComment records comfortably clear this (an 8-byte timespec + a
    4-byte length precede the text alone), so callers here must pass text
    of more than 8 bytes too.
    """
    assert len(text) > 8, "synthetic comment payload must be > 8 bytes (see gsf.c GSF_RECORD_SIZE_ERROR)"
    return _pack_record(RecordType.GSF_RECORD_COMMENT, text)


# ---------------------------------------------------------------------------
# Synthetic-record tests: data identifier bit-packing
# ---------------------------------------------------------------------------

class TestDataIdentifierBitPacking:
    """
    Verify the checksumFlag / reserved / recordID bit extraction against
    gsf.h's documented layout of the packed 32-bit data identifier word:

        1098 7654 3210 9876 5432 1098 7654 3210
        1000 0000 0000 0000 0000 0000 0000 0000   checksumFlag (bit 31)
        0111 1111 1100 0000 0000 0000 0000 0000   reserved (bits 22-30)
        0000 0000 0011 1111 1111 1111 1111 1111   recordID (bits 0-21)
    """

    def test_plain_record_id_no_checksum(self, tmp_path):
        record = _pack_record(RecordType.GSF_RECORD_COMMENT, b"1234567890", checksum_flag=False)
        path = tmp_path / "synthetic.gsf"
        _write_records(path, [record])

        G = gsf(str(path))
        G.OpenFiletoRead()
        dataSize, readSize, data_id = G.read_record_header()

        assert dataSize == 10
        assert readSize == 10
        assert data_id.checksumFlag is False
        assert data_id.reserved == 0
        assert data_id.recordID == RecordType.GSF_RECORD_COMMENT

    def test_checksum_flag_and_reserved_bits(self, tmp_path):
        record = _pack_record(
            RecordType.GSF_RECORD_ATTITUDE, b"payload!", checksum_flag=True, reserved=0x1AB)
        path = tmp_path / "synthetic.gsf"
        _write_records(path, [record])

        G = gsf(str(path))
        G.OpenFiletoRead()
        dataSize, readSize, data_id = G.read_record_header()

        # dataSize is the payload size (checksum not included), readSize
        # additionally accounts for the 4-byte checksum word.
        assert dataSize == 8
        assert readSize == 12
        assert data_id.checksumFlag is True
        assert data_id.reserved == 0x1AB
        assert data_id.recordID == RecordType.GSF_RECORD_ATTITUDE

    def test_registry_number_bits_included_in_record_id(self, tmp_path):
        # recordID packs "bits 00-11 => data type number, bits 12-22 =>
        # registry number" into a single field; a non-zero registry number
        # pushes recordID above the range of known top-level record types
        # (1..NUM_REC_TYPES-1), and should be rejected as unrecognized here
        # since this indexer only tracks the standard record types.
        record_id_with_registry = RecordType.GSF_RECORD_COMMENT | (1 << 12)
        record = _pack_record(record_id_with_registry, b"xxxxxxxxxxxx")
        path = tmp_path / "synthetic.gsf"
        _write_records(path, [record])

        G = gsf(str(path))
        G.OpenFiletoRead()
        with pytest.raises(GSFUnrecognizedRecordIDError):
            G.read_record_header()


class TestGsfChecksum:
    def test_checksum_is_byte_wise_sum(self):
        assert gsf_checksum(b"\x00\x00\x00") == 0
        assert gsf_checksum(b"\x01\x02\x03") == 6
        assert gsf_checksum(bytes([255, 255])) == 510

    def test_checksum_wraps_modulo_32(self):
        # 2**32 identical 0xFF bytes would overflow a 32-bit sum; verify the
        # wrap happens at 2**32, matching the C library's unsigned 32-bit
        # accumulator.
        data = bytes([0xFF]) * ((1 << 32) // 0xFF + 1)
        expected = sum(data) % (1 << 32)
        assert gsf_checksum(data) == expected


# ---------------------------------------------------------------------------
# Synthetic-record tests: error conditions
# ---------------------------------------------------------------------------

class TestErrorConditions:
    def test_unrecognized_record_id_raises(self, tmp_path):
        record = _pack_record(NUM_REC_TYPES + 5, b"data012345")
        path = tmp_path / "bad.gsf"
        _write_records(path, [record])

        G = gsf(str(path))
        G.OpenFiletoRead()
        with pytest.raises(GSFUnrecognizedRecordIDError):
            G.read_record_header()

    def test_record_id_zero_raises(self, tmp_path):
        # 0 is reserved for GSF_NEXT_RECORD (a request, not a stored type)
        # and is not itself a valid on-disk recordID.
        record = _pack_record(0, b"data012345")
        path = tmp_path / "bad.gsf"
        _write_records(path, [record])

        G = gsf(str(path))
        G.OpenFiletoRead()
        with pytest.raises(GSFUnrecognizedRecordIDError):
            G.read_record_header()

    def test_zero_length_payload_raises_size_error(self, tmp_path):
        # readSize must be > 8 per gsf.c's gsfUnpackStream (a zero-length
        # payload is rejected as a record size error).
        record = _pack_record(RecordType.GSF_RECORD_COMMENT, b"")
        path = tmp_path / "bad.gsf"
        _write_records(path, [record])

        G = gsf(str(path))
        G.OpenFiletoRead()
        with pytest.raises(GSFRecordSizeError):
            G.read_record_header()

    def test_truncated_record_header_raises_partial_record(self, tmp_path):
        path = tmp_path / "truncated.gsf"
        with open(path, "wb") as f:
            f.write(struct.pack('>I', 100))  # only 4 of the required 8 bytes

        G = gsf(str(path))
        G.OpenFiletoRead()
        with pytest.raises(GSFPartialRecordAtEndOfFileError):
            G.read_record_header()

    def test_declared_size_extends_past_eof_raises_partial_record(self, tmp_path):
        did = RecordType.GSF_RECORD_COMMENT  # no checksum, no registry/reserved bits
        path = tmp_path / "truncated.gsf"
        with open(path, "wb") as f:
            # Declare a much larger payload than actually follows.
            f.write(struct.pack('>II', 10_000, did))
            f.write(b"short")

        G = gsf(str(path))
        G.OpenFiletoRead()
        with pytest.raises(GSFPartialRecordAtEndOfFileError):
            G.read_record_header()

    def test_clean_eof_returns_none(self, tmp_path):
        path = tmp_path / "empty_after_header.gsf"
        _write_records(path, [_header_record()])

        G = gsf(str(path))
        G.index_file()
        assert G.read_record_header() is None

    def test_missing_file_raises(self, tmp_path):
        G = gsf(str(tmp_path / "does_not_exist.gsf"))
        with pytest.raises(FileNotFoundError):
            G.OpenFiletoRead()


# ---------------------------------------------------------------------------
# Synthetic-record tests: index_file() / report_record_types()
# ---------------------------------------------------------------------------

class TestIndexFileSynthetic:
    def test_index_contains_one_row_per_record_in_order(self, tmp_path):
        records = [_header_record(), _comment_record(b"aaaaaaaaaa"), _comment_record(b"bbbbbbbbbbbb")]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.index_file()

        assert len(G.Index) == 3
        assert list(G.Index['RecordType']) == [
            'GSF_RECORD_HEADER', 'GSF_RECORD_COMMENT', 'GSF_RECORD_COMMENT']
        assert list(G.Index['ByteOffset']) == [0, len(records[0]), len(records[0]) + len(records[1])]

    def test_index_accounts_for_every_byte(self, tmp_path):
        records = [_header_record(), _comment_record(b"x" * 50), _comment_record(b"y" * 9)]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.index_file()

        assert int(G.Index['TotalBytes'].sum()) == path.stat().st_size
        last = G.Index.iloc[-1]
        assert last['ByteOffset'] + last['TotalBytes'] == path.stat().st_size

    def test_gsf_version_captured_from_header(self, tmp_path):
        records = [_header_record(), _comment_record()]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.index_file()

        assert G.gsfVersion == "GSF-v03.09"

    def test_checksummed_record_indexed_correctly(self, tmp_path):
        payload = b"checksum me"
        records = [
            _header_record(),
            _pack_record(RecordType.GSF_RECORD_COMMENT, payload, checksum_flag=True),
        ]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.index_file()

        checksummed_row = G.Index.iloc[1]
        assert bool(checksummed_row['ChecksumFlag']) is True
        # RecordSize (the on-disk "size" field) excludes the 4-byte checksum.
        assert checksummed_row['RecordSize'] == len(payload)
        # TotalBytes (actual on-disk footprint) does include it: framing(8) + checksum(4) + payload.
        assert checksummed_row['TotalBytes'] == GSF_RECORD_FRAMING_SIZE + 4 + len(payload)
        assert int(G.Index['TotalBytes'].sum()) == path.stat().st_size

    def test_report_record_types_returns_expected_summary(self, tmp_path, capsys):
        records = [_header_record(), _comment_record(b"aaaaaaaaaa"), _comment_record(b"bbbbbbbbbbbb")]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        summary = G.report_record_types()

        assert summary.loc['GSF_RECORD_COMMENT', 'Count'] == 2
        assert summary.loc['GSF_RECORD_HEADER', 'Count'] == 1
        assert int(summary['Count'].sum()) == 3
        assert int(summary['Total Bytes'].sum()) == path.stat().st_size

        captured = capsys.readouterr()
        assert "GSF_RECORD_COMMENT" in captured.out
        assert "GSF Version: GSF-v03.09" in captured.out

    def test_no_unaccounted_bytes_warning_printed_for_valid_file(self, tmp_path, capsys):
        records = [_header_record(), _comment_record()]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.index_file()

        captured = capsys.readouterr()
        assert "WARNING" not in captured.out


# ---------------------------------------------------------------------------
# Real-file tests
# ---------------------------------------------------------------------------

@requires_sample_data
class TestIndexFileRealData:
    @pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
    def test_index_accounts_for_every_byte(self, path):
        G = gsf(str(path))
        G.index_file()

        assert int(G.Index['TotalBytes'].sum()) == path.stat().st_size
        last = G.Index.iloc[-1]
        assert last['ByteOffset'] + last['TotalBytes'] == path.stat().st_size

    @pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
    def test_all_record_types_recognized(self, path):
        G = gsf(str(path))
        G.index_file()

        unknown = G.Index[G.Index['RecordType'].astype(str).str.startswith("UNKNOWN")]
        assert len(unknown) == 0

    def test_first_record_is_header_at_offset_zero(self):
        G = gsf(str(SMALL_SAMPLE))
        G.index_file()

        first = G.Index.iloc[0]
        assert first['RecordType'] == 'GSF_RECORD_HEADER'
        assert first['RecordID'] == RecordType.GSF_RECORD_HEADER
        assert first['ByteOffset'] == 0

    def test_gsf_version_parsed(self):
        G = gsf(str(SMALL_SAMPLE))
        G.index_file()
        assert G.gsfVersion.startswith("GSF-v")

    def test_report_record_types_columns_and_totals(self):
        G = gsf(str(SMALL_SAMPLE))
        summary = G.report_record_types()

        for column in ('Count', 'Total Bytes', 'Min Bytes', 'Max Bytes', '% of File'):
            assert column in summary.columns

        assert int(summary['Count'].sum()) == len(G.Index)
        assert int(summary['Total Bytes'].sum()) == G.file_size

    def test_reindexing_reopens_and_reproduces_same_index(self):
        G = gsf(str(SMALL_SAMPLE))
        G.index_file()
        first_index = G.Index.copy()

        G.index_file()  # should reopen (closeFile + OpenFiletoRead) cleanly

        assert list(G.Index['RecordType']) == list(first_index['RecordType'])
        assert list(G.Index['ByteOffset']) == list(first_index['ByteOffset'])


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

@requires_sample_data
class TestCLI:
    def test_dash_V_prints_summary(self, capsys):
        rc = main(['-f', str(SMALL_SAMPLE), '-V'])
        captured = capsys.readouterr()

        assert rc == 0
        assert "GSF_RECORD_SWATH_BATHYMETRY_PING" in captured.out
        assert "GSF Version" in captured.out

    def test_without_dash_V_just_indexes(self, capsys):
        rc = main(['-f', str(SMALL_SAMPLE)])
        captured = capsys.readouterr()

        assert rc == 0
        assert "Indexed" in captured.out

    def test_missing_filename_prints_help_and_errors(self, capsys):
        rc = main([])
        captured = capsys.readouterr()

        assert rc == 1
        assert "usage" in captured.out.lower()
