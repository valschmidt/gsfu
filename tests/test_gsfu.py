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
    _SENSOR_SPECIFIC_SUBRECORD_NAMES,
    _decode_brb_intensity,
    _decode_name_value_parameters,
    _decode_swath_bathymetry_ping,
    gsf,
    gsf_checksum,
    main,
    resolve_record_type,
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
# resolve_record_type()
# ---------------------------------------------------------------------------

class TestResolveRecordType:
    def test_none_passes_through(self):
        assert resolve_record_type(None) is None

    def test_record_type_passes_through(self):
        assert resolve_record_type(RecordType.GSF_RECORD_COMMENT) == RecordType.GSF_RECORD_COMMENT

    def test_int_resolves_to_record_type(self):
        assert resolve_record_type(int(RecordType.GSF_RECORD_COMMENT)) == RecordType.GSF_RECORD_COMMENT

    def test_short_name_resolves_without_prefix(self):
        assert resolve_record_type("COMMENT") == RecordType.GSF_RECORD_COMMENT

    def test_short_name_is_case_insensitive(self):
        assert resolve_record_type("comment") == RecordType.GSF_RECORD_COMMENT
        assert resolve_record_type("Comment") == RecordType.GSF_RECORD_COMMENT

    def test_full_name_still_accepted(self):
        assert resolve_record_type("GSF_RECORD_COMMENT") == RecordType.GSF_RECORD_COMMENT

    def test_unknown_name_raises_with_valid_names_listed(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_record_type("BOGUS")
        assert "Unknown record type: BOGUS" in str(excinfo.value)
        for rt in RecordType:
            assert rt.name in str(excinfo.value)


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
# print_records() -- ASCII debug dump
# ---------------------------------------------------------------------------

class TestPrintRecordsSynthetic:
    def test_prints_every_record_by_default(self, tmp_path, capsys):
        records = [_header_record(), _comment_record(b"aaaaaaaaaa"), _comment_record(b"bbbbbbbbbbbb")]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.print_records()

        captured = capsys.readouterr()
        assert captured.out.count("=== GSF_RECORD_HEADER") == 1
        assert captured.out.count("=== GSF_RECORD_COMMENT") == 2

    def test_decoded_comment_prints_key_value_pairs(self, tmp_path, capsys):
        text = b"hello world!"
        payload = struct.pack('>3I', 1700000000, 0, len(text)) + text
        records = [_header_record(), _pack_record(RecordType.GSF_RECORD_COMMENT, payload)]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.print_records(record_type=RecordType.GSF_RECORD_COMMENT)

        captured = capsys.readouterr()
        assert "CommentTime" in captured.out
        assert "Comment" in captured.out
        assert "hello world!" in captured.out

    def test_undecodable_payload_falls_back_to_raw_text(self, tmp_path, capsys):
        # A COMMENT record too short to hold its own time+length fields
        # (needs 12 bytes minimum, but must still clear the top-level
        # framing's own >8-byte minimum) triggers the decode-failure fallback.
        text = b"1234567890"  # 10 bytes: > 8 (framing minimum), < 12 (decoder minimum)
        records = [_header_record(), _pack_record(RecordType.GSF_RECORD_COMMENT, text)]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.print_records(record_type=RecordType.GSF_RECORD_COMMENT)

        captured = capsys.readouterr()
        assert "decode failed" in captured.out
        assert text.decode("ascii") in captured.out

    def test_non_printable_bytes_rendered_as_dots(self, tmp_path, capsys):
        payload = bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # all non-printable, len > 8
        records = [_header_record(), _pack_record(RecordType.GSF_RECORD_COMMENT, payload)]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.print_records(record_type=RecordType.GSF_RECORD_COMMENT)

        captured = capsys.readouterr()
        assert "." * len(payload) in captured.out

    def test_filters_to_requested_record_type_only(self, tmp_path, capsys):
        records = [_header_record(), _comment_record(b"aaaaaaaaaa")]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.print_records(record_type=RecordType.GSF_RECORD_COMMENT)

        captured = capsys.readouterr()
        assert "GSF_RECORD_HEADER" not in captured.out
        assert "GSF_RECORD_COMMENT" in captured.out

    def test_checksummed_record_payload_printed_without_checksum_bytes(self, tmp_path, capsys):
        payload = b"checksum me"
        records = [
            _header_record(),
            _pack_record(RecordType.GSF_RECORD_COMMENT, payload, checksum_flag=True),
        ]
        path = tmp_path / "synthetic.gsf"
        _write_records(path, records)

        G = gsf(str(path))
        G.print_records(record_type=RecordType.GSF_RECORD_COMMENT)

        captured = capsys.readouterr()
        assert payload.decode("ascii") in captured.out


@requires_sample_data
class TestPrintRecordsRealData:
    def test_header_record_prints_gsf_version_string(self, capsys):
        G = gsf(str(SMALL_SAMPLE))
        G.print_records(record_type=RecordType.GSF_RECORD_HEADER)

        captured = capsys.readouterr()
        assert "GSF-v" in captured.out

    def test_no_filter_covers_every_indexed_record(self, capsys):
        G = gsf(str(SMALL_SAMPLE))
        G.index_file()
        expected_counts = G.Index['RecordType'].value_counts().to_dict()

        G2 = gsf(str(SMALL_SAMPLE))
        G2.print_records()
        captured = capsys.readouterr()

        for record_type, count in expected_counts.items():
            assert captured.out.count("=== %s " % record_type) == count

    @pytest.mark.parametrize("rt", list(RecordType), ids=lambda rt: rt.name)
    def test_every_record_type_decodes_without_falling_back(self, rt, capsys):
        # Every GSF_RECORD_* type has a field-level decoder (see
        # _decode_record); none of them should hit the "decode failed"
        # exception fallback against real, well-formed data. Types absent
        # from this particular sample file simply produce no output.
        G = gsf(str(SMALL_SAMPLE))
        G.print_records(record_type=rt)

        captured = capsys.readouterr()
        assert "decode failed" not in captured.out

    def test_swath_bathymetry_ping_table_has_expected_columns(self, capsys):
        G = gsf(str(SMALL_SAMPLE))
        G.print_records(record_type=RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING)

        captured = capsys.readouterr()
        for column in ('Depth_m', 'AcrossTrack_m', 'AlongTrack_m', 'TravelTime_s', 'BeamAngle_deg'):
            assert column in captured.out
        assert 'PingTime' in captured.out
        assert 'NumberBeams' in captured.out

    def test_processing_parameters_have_no_embedded_nul_bytes(self, capsys):
        G = gsf(str(SMALL_SAMPLE))
        G.print_records(record_type=RecordType.GSF_RECORD_PROCESSING_PARAMETERS)

        captured = capsys.readouterr()
        assert "PLATFORM_TYPE" in captured.out
        assert '\x00' not in captured.out

    def test_kmall_specific_prints_expected_sections(self, capsys):
        G = gsf(str(SMALL_SAMPLE))
        G.print_records(record_type=RecordType.GSF_RECORD_SWATH_BATHYMETRY_PING)

        captured = capsys.readouterr()
        assert "decode failed" not in captured.out
        assert "KMALL.EchoSounderID" in captured.out
        assert "-- TxSectors --" in captured.out
        assert "IntensityTimeSeries (21," in captured.out  # noted, not decoded via -p

    @pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda p: p.name)
    def test_kmall_specific_decodes_every_ping_without_error(self, path):
        # These sample files are all EM712 (KMALL_SPECIFIC, subrecord id
        # 156); every ping should decode its vendor-specific subrecord
        # cleanly, matching a real echo sounder id and a TxSectors table
        # with exactly NumTxSectors rows. Decodes directly (bypassing
        # print_records()'s table rendering) to keep this fast across all
        # sample files, including the largest ones.
        G = gsf(str(path))
        G.index_file()

        ping_offsets = G.Index.loc[
            G.Index['RecordType'] == 'GSF_RECORD_SWATH_BATHYMETRY_PING', 'ByteOffset']

        scale_factors = {}
        for offset in ping_offsets:
            G.FID.seek(int(offset))
            dataSize, _readSize, data_id = G.read_record_header()
            if data_id.checksumFlag:
                G.FID.seek(4, 1)
            payload = G.FID.read(dataSize)

            scalars, tables, notes = _decode_swath_bathymetry_ping(
                payload, major_version=3, scale_factors=scale_factors)

            assert all("IntensityTimeSeries" in n for n in notes)
            assert scalars['KMALL.EchoSounderID'] == 712
            assert len(tables['TxSectors']) == scalars['KMALL.NumTxSectors']

    def test_kmall_specific_num_tx_sectors_matches_tx_sectors_table_rows(self):
        G = gsf(str(SMALL_SAMPLE))
        G.index_file()
        G.OpenFiletoRead()
        first_ping_offset = int(
            G.Index.loc[G.Index['RecordType'] == 'GSF_RECORD_SWATH_BATHYMETRY_PING', 'ByteOffset'].iloc[0])

        G.FID.seek(first_ping_offset)
        dataSize, _readSize, data_id = G.read_record_header()
        payload = G.FID.read(dataSize)

        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert all("IntensityTimeSeries" in n for n in notes)
        assert scalars['KMALL.EchoSounderID'] == 712
        assert len(tables['TxSectors']) == scalars['KMALL.NumTxSectors']


class TestDecodeSwathBathymetryPingSynthetic:
    """
    Whitebox tests for _decode_swath_bathymetry_ping() against a hand-built
    payload (fixed header + a scale factors subrecord + one beam array
    subrecord), independent of any real file, to pin down the scale/offset
    arithmetic and subrecord framing exactly.
    """

    @staticmethod
    def _fixed_header(number_beams):
        # major_version=2 fixed-header layout (42 bytes; skips the
        # height/SEP/GPS-tide-corrector extension used at major_version>2).
        return struct.pack(
            '>2I2i4HhiH3h2H',
            1700000000, 0,       # ping_time sec, nsec
            -1571234567,          # longitude raw (-157.1234567 deg)
            187654321,             # latitude raw (18.7654321 deg)
            number_beams, number_beams // 2, 0, 0,  # number_beams, center_beam, ping_flags, reserved
            -50,                  # tide_corrector raw (-0.50 m)
            244,                   # depth_corrector raw (2.44 m)
            35872,                 # heading raw (358.72 deg)
            -358, -411, 55,        # pitch, roll, heave raw
            0, 0,                  # course, speed raw
        )

    @staticmethod
    def _scale_factors_subrecord(entries):
        # entries: dict subrecordID -> (multiplier, offset)
        body = struct.pack('>I', len(entries))
        for subrecord_id, (multiplier, offset) in entries.items():
            body += struct.pack('>I', (subrecord_id & 0xFF) << 24)
            body += struct.pack('>I', int(multiplier))
            body += struct.pack('>i', int(offset))
        word = (100 << 24) | len(body)
        return struct.pack('>I', word) + body

    @staticmethod
    def _array_subrecord(subrecord_id, values, fmt):
        body = b"".join(struct.pack(fmt, v) for v in values)
        word = ((subrecord_id & 0xFF) << 24) | len(body)
        return struct.pack('>I', word) + body

    def test_fixed_header_scalars(self):
        payload = self._fixed_header(3) + self._scale_factors_subrecord({1: (100.0, 0)}) \
            + self._array_subrecord(1, [1000, 1050, 995], '>H')

        scalars, beams, notes = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert scalars['NumberBeams'] == 3
        assert scalars['CenterBeam'] == 1
        assert scalars['TideCorrector_m'] == pytest.approx(-0.50)
        assert scalars['DepthCorrector_m'] == pytest.approx(2.44)
        assert scalars['Heading_deg'] == pytest.approx(358.72)
        assert scalars['Pitch_deg'] == pytest.approx(-3.58)
        assert scalars['Roll_deg'] == pytest.approx(-4.11)
        assert scalars['Heave_m'] == pytest.approx(0.55)
        assert scalars['Longitude_deg'] == pytest.approx(-157.1234567)
        assert scalars['Latitude_deg'] == pytest.approx(18.7654321)
        assert notes == []

    def test_depth_array_decoded_with_scale_and_offset(self):
        payload = self._fixed_header(3) + self._scale_factors_subrecord({1: (100.0, 0)}) \
            + self._array_subrecord(1, [1000, 1050, 995], '>H')

        _scalars, tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})
        beams = tables['Beams']

        assert list(beams['Depth_m']) == pytest.approx([10.0, 10.5, 9.95])
        assert beams.index.name == 'Beam'
        assert list(beams.index) == [0, 1, 2]

    def test_signed_array_and_nonzero_offset_applied(self):
        # across_track (id 2) is signed; multiplier=10, offset=5 =>
        # value = raw/10 - 5. raw=-30 -> -8.0; raw=100 -> 5.0.
        payload = self._fixed_header(2) + self._scale_factors_subrecord({2: (10.0, 5)}) \
            + self._array_subrecord(2, [-30, 100], '>h')

        _scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert list(tables['Beams']['AcrossTrack_m']) == pytest.approx([-8.0, 5.0])
        assert notes == []

    def test_missing_scale_factors_reported_as_note_not_crash(self):
        # A DEPTH_ARRAY subrecord with no preceding SCALE_FACTORS (and none
        # cached from an earlier ping) can't be scaled; it should be
        # reported via `notes`, not raise or silently fabricate a column.
        payload = self._fixed_header(2) + self._array_subrecord(1, [100, 200], '>H')

        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert 'Beams' not in tables
        assert len(notes) == 1
        assert "no scale factors available" in notes[0]

    def test_unrecognized_subrecord_reported_as_note(self):
        # A subrecord id with no entry in _PING_ARRAY_SUBRECORDS, no known
        # vendor "_SPECIFIC" name, and not SCALE_FACTORS/BEAM_FLAGS/
        # INTENSITY_SERIES -- id 154 is unused/reserved in gsf.h, so it can
        # never collide with a real subrecord -- is skipped and reported by
        # bare numeric id, not decoded.
        payload = self._fixed_header(1) + self._array_subrecord(154, [0, 1, 2, 3], '>B')

        _scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert 'Beams' not in tables
        assert len(notes) == 1
        assert "subrecord id 154 (4 bytes) not decoded" in notes[0]

    def test_known_vendor_specific_subrecord_reported_by_name(self):
        # A vendor "_SPECIFIC" subrecord with no field-level decoder here
        # (id 133 = EM710_SPECIFIC) is still reported by its proper name,
        # not a bare numeric id.
        payload = self._fixed_header(1) + self._array_subrecord(133, [0, 1, 2, 3], '>B')

        _scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert 'Beams' not in tables
        assert len(notes) == 1
        assert "EM710_SPECIFIC (133, 4 bytes) not decoded" in notes[0]

    def test_scale_factors_persist_across_calls_via_shared_cache(self):
        # Mirrors gsflib's behavior: a ping need not repeat scale factors
        # that haven't changed since an earlier ping in the same file: the
        # caller-supplied `scale_factors` dict carries them forward.
        shared_cache = {}
        first_ping = self._fixed_header(2) + self._scale_factors_subrecord({1: (100.0, 0)}) \
            + self._array_subrecord(1, [1000, 1050], '>H')
        _decode_swath_bathymetry_ping(first_ping, major_version=2, scale_factors=shared_cache)

        second_ping = self._fixed_header(2) + self._array_subrecord(1, [2000, 500], '>H')
        _scalars, tables, notes = _decode_swath_bathymetry_ping(
            second_ping, major_version=2, scale_factors=shared_cache)

        assert notes == []
        assert list(tables['Beams']['Depth_m']) == pytest.approx([20.0, 5.0])


class TestDecodeNameValueParametersSynthetic:
    """
    Whitebox tests for _decode_name_value_parameters() (used for both
    GSF_RECORD_PROCESSING_PARAMETERS and GSF_RECORD_SENSOR_PARAMETERS).
    """

    @staticmethod
    def _payload(param_strings):
        # param_time (8 bytes) + number_parameters (2 bytes) + per-param:
        # size (2 bytes, signed) + that many bytes of text.
        body = struct.pack('>2I', 1700000000, 0) + struct.pack('>H', len(param_strings))
        for text in param_strings:
            encoded = text.encode('ascii')
            body += struct.pack('>h', len(encoded)) + encoded
        return body

    def test_plain_name_value_pair(self):
        payload = self._payload([b"PLATFORM_TYPE=SURFACE_SHIP".decode()])
        scalars, _tables, _notes = _decode_name_value_parameters(payload)
        assert scalars['PLATFORM_TYPE'] == 'SURFACE_SHIP'

    def test_trailing_nul_byte_stripped_from_value(self):
        # Some encoders count a trailing C-string NUL terminator as part of
        # a parameter's size; it must not appear in the decoded value.
        payload = self._payload(["ROLL_COMPENSATED=NO \x00"])
        scalars, _tables, _notes = _decode_name_value_parameters(payload)
        assert scalars['ROLL_COMPENSATED'] == 'NO '
        assert '\x00' not in scalars['ROLL_COMPENSATED']

    def test_trailing_space_before_nul_is_preserved(self):
        # Only the NUL is stripped -- padding the encoder itself wrote
        # (e.g. a trailing space) is left alone.
        payload = self._payload(["HEAVE_COMPENSATED=YES\x00"])
        scalars, _tables, _notes = _decode_name_value_parameters(payload)
        assert scalars['HEAVE_COMPENSATED'] == 'YES'

    def test_parameter_without_equals_sign_keyed_by_full_text(self):
        payload = self._payload(["FREEFORM_NOTE\x00"])
        scalars, _tables, _notes = _decode_name_value_parameters(payload)
        assert scalars['FREEFORM_NOTE'] == ''


class TestSensorSpecificSubrecordNames:
    def test_kmall_specific_named(self):
        assert _SENSOR_SPECIFIC_SUBRECORD_NAMES[156] == "KMALL_SPECIFIC"

    def test_em710_specific_named(self):
        assert _SENSOR_SPECIFIC_SUBRECORD_NAMES[133] == "EM710_SPECIFIC"

    def test_id_154_is_absent(self):
        # gsf.h has no GSF_SWATH_BATHY_SUBRECORD_* define for 154 (a gap
        # between R2SONIC_2020_SPECIFIC=153 and RESON_TSERIES_SPECIFIC=155).
        assert 154 not in _SENSOR_SPECIFIC_SUBRECORD_NAMES

    def test_covers_ids_102_through_157_except_154(self):
        expected = set(range(102, 158)) - {154}
        assert set(_SENSOR_SPECIFIC_SUBRECORD_NAMES) == expected


class TestDecodeBRBIntensitySynthetic:
    """
    Whitebox tests for _decode_brb_intensity() -- the per-beam backscatter
    time series decoder -- built around a hand-crafted KMALL-sensor payload
    (the only sensor-imagery format currently decoded).
    """

    #: gsf_dec.c's DecodeBRBIntensity() header: bits_per_sample(1) +
    #: applied_corrections(4) + spare(16) = 21 bytes, followed here by
    #: DecodeKMALLImagerySpecific()'s fixed 64 spare bytes.
    _KMALL_PREAMBLE_SIZE = 21 + 64

    @staticmethod
    def _preamble(bits_per_sample, applied_corrections=0):
        return struct.pack('>B', bits_per_sample) + struct.pack('>I', applied_corrections) \
            + b"\x00" * 16 + b"\x00" * 64

    @staticmethod
    def _beam(sample_count, detect_sample, start_range_samples, samples, fmt):
        header = struct.pack('>3H', sample_count, detect_sample, start_range_samples) + b"\x00" * 6
        return header + b"".join(struct.pack(fmt, v) for v in samples)

    def test_unsupported_sensor_id_returns_none(self):
        payload = self._preamble(8) + self._beam(1, 0, 0, [42], '>B')
        assert _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=999) is None

    def test_zero_beams_returns_none(self):
        payload = self._preamble(8)
        assert _decode_brb_intensity(payload, 0, num_beams=0, sensor_id=156) is None

    def test_8_bit_samples_decoded_per_beam(self):
        payload = self._preamble(8) \
            + self._beam(3, 1, 100, [10, 20, 30], '>B') \
            + self._beam(2, 0, 50, [200, 201], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=2, sensor_id=156)

        assert header['BitsPerSample'] == 8
        assert len(beam_rows) == 2
        assert beam_rows[0] == {
            'SampleCount': 3, 'DetectSample': 1, 'StartRangeSamples': 100, 'Samples': [10, 20, 30]}
        assert beam_rows[1] == {
            'SampleCount': 2, 'DetectSample': 0, 'StartRangeSamples': 50, 'Samples': [200, 201]}
        assert consumed == len(payload)

    def test_16_bit_samples_decoded_per_beam(self):
        payload = self._preamble(16) + self._beam(2, 5, 10, [1000, 65000], '>H')

        _header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=156)

        assert beam_rows[0]['Samples'] == [1000, 65000]
        assert consumed == len(payload)

    def test_12_bit_packed_samples_decoded(self):
        # Two 12-bit samples pack into 3 bytes: sample1 = (b0<<4)|(b1>>4),
        # sample2 = ((b1&0x0F)<<8)|b2. Choose sample1=0xABC, sample2=0x123:
        # b0 = 0xAB, b1 = (0xC<<4)|(0x1) = 0xC1, b2 = 0x23.
        packed = bytes([0xAB, 0xC1, 0x23])
        payload = self._preamble(12) \
            + struct.pack('>3H', 2, 0, 0) + b"\x00" * 6 + packed

        _header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=156)

        assert beam_rows[0]['Samples'] == [0xABC, 0x123]
        assert consumed == len(payload)

    def test_12_bit_odd_sample_count_drops_trailing_half_sample(self):
        # sample_count=1 with 12-bit packing: only the first sample of the
        # pair is emitted (mirrors gsf_dec.c's "if (j+1 < sample_count)" guard).
        packed = bytes([0xAB, 0xC0, 0x00])  # sample1 = 0xABC; sample2 unused
        payload = self._preamble(12) \
            + struct.pack('>3H', 1, 0, 0) + b"\x00" * 6 + packed

        _header, beam_rows, _consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=156)

        assert beam_rows[0]['Samples'] == [0xABC]


@requires_sample_data
class TestPrintIntensitySeriesRealData:
    def test_runs_without_error_and_prints_csv_rows(self, capsys):
        G = gsf(str(SMALL_SAMPLE))
        G.print_intensity_series()

        captured = capsys.readouterr()
        assert "decode failed" not in captured.out
        assert captured.out.count("# ping offset=") > 0
        # At least one beam row was printed with a sample count that
        # matches the number of trailing sample values in that same row.
        data_lines = [line for line in captured.out.splitlines() if line and not line.startswith("#")]
        assert data_lines
        beam, sample_count, detect_sample, start_range, *samples = data_lines[0].split(",")
        assert int(sample_count) == len(samples)

    def test_every_sample_row_matches_its_declared_sample_count(self, capsys):
        G = gsf(str(SMALL_SAMPLE))
        G.print_intensity_series()

        captured = capsys.readouterr()
        for line in captured.out.splitlines():
            if not line or line.startswith("#"):
                continue
            _beam, sample_count, _detect, _start, *samples = line.split(",")
            assert int(sample_count) == len(samples)


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

    def test_dash_p_with_no_value_prints_every_record(self, capsys):
        rc = main(['-f', str(SMALL_SAMPLE), '-p'])
        captured = capsys.readouterr()

        assert rc == 0
        assert "=== GSF_RECORD_HEADER" in captured.out
        assert "=== GSF_RECORD_ATTITUDE" in captured.out

    def test_dash_p_with_short_type_name_filters_output(self, capsys):
        rc = main(['-f', str(SMALL_SAMPLE), '-p', 'HEADER'])
        captured = capsys.readouterr()

        assert rc == 0
        assert "=== GSF_RECORD_HEADER" in captured.out
        assert "=== GSF_RECORD_ATTITUDE" not in captured.out

    def test_dash_p_with_full_type_name_filters_output(self, capsys):
        rc = main(['-f', str(SMALL_SAMPLE), '-p', 'GSF_RECORD_HEADER'])
        captured = capsys.readouterr()

        assert rc == 0
        assert "=== GSF_RECORD_HEADER" in captured.out
        assert "=== GSF_RECORD_ATTITUDE" not in captured.out

    def test_dash_p_with_unknown_type_name_errors(self, capsys):
        rc = main(['-f', str(SMALL_SAMPLE), '-p', 'BOGUS'])
        captured = capsys.readouterr()

        assert rc == 1
        assert "Unknown record type" in captured.out
        assert "GSF_RECORD_HEADER" in captured.out  # listed among valid types

    def test_dash_I_prints_intensity_csv(self, capsys):
        rc = main(['-f', str(SMALL_SAMPLE), '-I'])
        captured = capsys.readouterr()

        assert rc == 0
        assert "# ping offset=" in captured.out
        data_lines = [line for line in captured.out.splitlines() if line and not line.startswith("#")]
        assert data_lines

    def test_help_lists_every_record_type(self, capsys):
        with pytest.raises(SystemExit):
            main(['-h'])
        captured = capsys.readouterr()

        for rt in RecordType:
            assert rt.name.replace('GSF_RECORD_', '') in captured.out
