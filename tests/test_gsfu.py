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
    _SUBRECORD_EM3_IMAGERY_IDS,
    _SUBRECORD_EM4_IMAGERY_IDS,
    _SUBRECORD_KLEIN_5410_BSS_SPECIFIC,
    _SUBRECORD_R2SONIC_IMAGERY_IDS,
    _SUBRECORD_RESON_8100_IMAGERY_IDS,
    _SUBRECORD_RESON_SIZE_SPARE_IMAGERY_IDS,
    _decode_bdb_specific,
    _decode_brb_intensity,
    _decode_cmp_sass_specific,
    _decode_delta_t_specific,
    _decode_echotrac_specific,
    _decode_em3_run_time,
    _decode_em3_specific,
    _decode_em3raw_specific,
    _decode_em4_specific,
    _decode_em_pu_status,
    _decode_em_run_time,
    _encode_em_pu_status,
    _encode_em_run_time,
    _decode_elac_mkii_specific,
    _decode_em12_specific,
    _decode_em100_specific,
    _decode_em121a_specific,
    _decode_em950_specific,
    _decode_geoswath_plus_specific,
    _decode_klein5410bss_specific,
    _decode_mgd77_specific,
    _decode_name_value_parameters,
    _decode_noshdb_specific,
    _decode_quality_flags_array,
    _decode_r2sonic_specific,
    _decode_reson7125_specific,
    _decode_reson8100_specific,
    _decode_reson_tseries_specific,
    _decode_sass_specific,
    _decode_sb_amp_specific,
    _decode_seabat8101_specific,
    _decode_seabat_ii_specific,
    _decode_seabat_specific,
    _decode_seabeam_2112_specific,
    _decode_seabeam_specific,
    _decode_seamap_specific,
    _decode_single_beam_ping,
    _decode_swath_bathymetry_ping,
    _encode_swath_bathymetry_ping,
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
        assert "SensorSpecific (KMALL_SPECIFIC, id=156)" in captured.out
        assert "EchoSounderID" in captured.out
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

            record = _decode_swath_bathymetry_ping(
                payload, major_version=3, scale_factors=scale_factors)

            assert all("IntensityTimeSeries" in n for n in record['Notes'])
            assert record['SensorSpecific']['EchoSounderID'] == 712
            assert len(record['SensorSpecific']['TxSectors']) == record['SensorSpecific']['NumTxSectors']

    def test_kmall_specific_num_tx_sectors_matches_tx_sectors_table_rows(self):
        G = gsf(str(SMALL_SAMPLE))
        G.index_file()
        G.OpenFiletoRead()
        first_ping_offset = int(
            G.Index.loc[G.Index['RecordType'] == 'GSF_RECORD_SWATH_BATHYMETRY_PING', 'ByteOffset'].iloc[0])

        G.FID.seek(first_ping_offset)
        dataSize, _readSize, data_id = G.read_record_header()
        payload = G.FID.read(dataSize)

        record = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert all("IntensityTimeSeries" in n for n in record['Notes'])
        assert record['SensorSpecific']['EchoSounderID'] == 712
        assert len(record['SensorSpecific']['TxSectors']) == record['SensorSpecific']['NumTxSectors']

    def test_real_kmall_ping_round_trips_through_encode_with_tx_sectors_intact(self):
        # A real decoded record (its 'SensorSpecific' table values already
        # pandas.DataFrames) must be re-encodable via
        # _encode_swath_bathymetry_ping() with no repackaging -- exactly
        # the top-level decode-encode-decode round trip this redesign was
        # meant to make work cleanly, since 'TxSectors' is a DataFrame on
        # both sides now with no list[dict] conversion step anywhere.
        G = gsf(str(SMALL_SAMPLE))
        G.index_file()
        G.OpenFiletoRead()
        first_ping_offset = int(
            G.Index.loc[G.Index['RecordType'] == 'GSF_RECORD_SWATH_BATHYMETRY_PING', 'ByteOffset'].iloc[0])

        G.FID.seek(first_ping_offset)
        dataSize, _readSize, _data_id = G.read_record_header()
        payload = G.FID.read(dataSize)

        record = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})
        original_tx_sectors = record['SensorSpecific']['TxSectors']

        re_encoded = _encode_swath_bathymetry_ping(record, major_version=3)
        re_decoded = _decode_swath_bathymetry_ping(re_encoded, major_version=3, scale_factors={})

        assert re_decoded['SensorSpecificID'] == 156
        assert re_decoded['SensorSpecific']['EchoSounderID'] == record['SensorSpecific']['EchoSounderID']
        re_tx_sectors = re_decoded['SensorSpecific']['TxSectors']
        assert len(re_tx_sectors) == len(original_tx_sectors)
        assert list(re_tx_sectors['CentreFreq_Hz']) == pytest.approx(list(original_tx_sectors['CentreFreq_Hz']))
        assert list(re_decoded['Beams']['Depth_m']) == pytest.approx(list(record['Beams']['Depth_m']), abs=0.001)


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

        record = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert record['NumberBeams'] == 3
        assert record['CenterBeam'] == 1
        assert record['TideCorrector_m'] == pytest.approx(-0.50)
        assert record['DepthCorrector_m'] == pytest.approx(2.44)
        assert record['Heading_deg'] == pytest.approx(358.72)
        assert record['Pitch_deg'] == pytest.approx(-3.58)
        assert record['Roll_deg'] == pytest.approx(-4.11)
        assert record['Heave_m'] == pytest.approx(0.55)
        assert record['Longitude_deg'] == pytest.approx(-157.1234567)
        assert record['Latitude_deg'] == pytest.approx(18.7654321)
        assert record['Notes'] == []

    def test_depth_array_decoded_with_scale_and_offset(self):
        payload = self._fixed_header(3) + self._scale_factors_subrecord({1: (100.0, 0)}) \
            + self._array_subrecord(1, [1000, 1050, 995], '>H')

        record = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})
        beams = record['Beams']

        assert list(beams['Depth_m']) == pytest.approx([10.0, 10.5, 9.95])
        assert beams.index.name == 'Beam'
        assert list(beams.index) == [0, 1, 2]

    def test_signed_array_and_nonzero_offset_applied(self):
        # across_track (id 2) is signed; multiplier=10, offset=5 =>
        # value = raw/10 - 5. raw=-30 -> -8.0; raw=100 -> 5.0.
        payload = self._fixed_header(2) + self._scale_factors_subrecord({2: (10.0, 5)}) \
            + self._array_subrecord(2, [-30, 100], '>h')

        record = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert list(record['Beams']['AcrossTrack_m']) == pytest.approx([-8.0, 5.0])
        assert record['Notes'] == []

    def test_missing_scale_factors_reported_as_note_not_crash(self):
        # A DEPTH_ARRAY subrecord with no preceding SCALE_FACTORS (and none
        # cached from an earlier ping) can't be scaled; it should be
        # reported via `record['Notes']`, not raise or silently fabricate a
        # column.
        payload = self._fixed_header(2) + self._array_subrecord(1, [100, 200], '>H')

        record = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert 'Beams' not in record
        assert len(record['Notes']) == 1
        assert "no scale factors available" in record['Notes'][0]

    def test_unrecognized_subrecord_reported_as_note(self):
        # A subrecord id with no entry in _PING_ARRAY_SUBRECORDS, no known
        # vendor "_SPECIFIC" name, and not SCALE_FACTORS/BEAM_FLAGS/
        # INTENSITY_SERIES -- id 154 is unused/reserved in gsf.h, so it can
        # never collide with a real subrecord -- is skipped and reported by
        # bare numeric id, not decoded.
        payload = self._fixed_header(1) + self._array_subrecord(154, [0, 1, 2, 3], '>B')

        record = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert 'Beams' not in record
        assert len(record['Notes']) == 1
        assert "subrecord id 154 (4 bytes) not decoded" in record['Notes'][0]

    def test_known_vendor_specific_subrecord_reported_by_name(self):
        # A vendor "_SPECIFIC" subrecord with no field-level decoder here
        # (id 133 = EM710_SPECIFIC) is still reported by its proper name,
        # not a bare numeric id.
        payload = self._fixed_header(1) + self._array_subrecord(133, [0, 1, 2, 3], '>B')

        record = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})

        assert 'Beams' not in record
        assert len(record['Notes']) == 1
        assert "EM710_SPECIFIC (133, 4 bytes) not decoded" in record['Notes'][0]

    def test_scale_factors_persist_across_calls_via_shared_cache(self):
        # Mirrors gsflib's behavior: a ping need not repeat scale factors
        # that haven't changed since an earlier ping in the same file: the
        # caller-supplied `scale_factors` dict carries them forward.
        shared_cache = {}
        first_ping = self._fixed_header(2) + self._scale_factors_subrecord({1: (100.0, 0)}) \
            + self._array_subrecord(1, [1000, 1050], '>H')
        _decode_swath_bathymetry_ping(first_ping, major_version=2, scale_factors=shared_cache)

        second_ping = self._fixed_header(2) + self._array_subrecord(1, [2000, 500], '>H')
        record = _decode_swath_bathymetry_ping(
            second_ping, major_version=2, scale_factors=shared_cache)

        assert record['Notes'] == []
        assert list(record['Beams']['Depth_m']) == pytest.approx([20.0, 5.0])


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


class TestDecodeElacMkIISpecific:
    """
    Whitebox test for _decode_elac_mkii_specific() -- the reference
    example for the ping-level sensor-specific subrecord registry
    (_PING_SENSOR_SPECIFIC_CODECS). Every other family follows this same
    decode_fn(payload, pos) -> (fields, tables, bytes_consumed) shape.
    """

    def test_decode(self):
        payload = struct.pack('>BHHHBBH', 5, 42, 1500, 200, 10, 12, 0)
        fields, tables, consumed = _decode_elac_mkii_specific(payload, 0)

        assert fields == {
            'Mode': 5, 'PingNumber': 42, 'SoundVelocity_mps': 1500,
            'PulseLength_hundredth_ms': 200, 'ReceiverGainStbd_dB': 10,
            'ReceiverGainPort_dB': 12, 'Reserved': 0,
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeSeabeamSpecific:
    def test_decode(self):
        payload = struct.pack('>H', 1234)
        fields, tables, consumed = _decode_seabeam_specific(payload, 0)

        assert fields == {'EclipseTime_tenths_s': 1234}
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeEM12Specific:
    def test_decode(self):
        payload = struct.pack('>HBBHB', 100, 1, 50, 15003, 2) + b'\x00' * 32
        fields, tables, consumed = _decode_em12_specific(payload, 0)

        assert fields == {
            'PingNumber': 100, 'Resolution': 1, 'PingQuality': 50,
            'SoundVelocity_mps': pytest.approx(1500.3), 'Mode': 2,
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeEM100Specific:
    def test_decode(self):
        payload = struct.pack('>hhBBBBBH', -123, 45, 1, 2, 3, 4, 5, 999)
        fields, tables, consumed = _decode_em100_specific(payload, 0)

        assert fields == {
            'ShipPitch_deg': pytest.approx(-1.23), 'TransducerPitch_deg': pytest.approx(0.45),
            'Mode': 1, 'Power': 2, 'Attenuation': 3, 'TVG': 4, 'PulseLength': 5, 'Counter': 999,
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeCmpSassSpecific:
    def test_decode(self):
        payload = struct.pack('>HH', 49005, 25)
        fields, tables, consumed = _decode_cmp_sass_specific(payload, 0)

        assert fields == {
            'SurfaceSoundVelocity_ftps': pytest.approx(4900.5), 'Heave_ftps': pytest.approx(2.5),
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeEm950Specific:
    """ Shared by ids 105 (EM950) and 111 (EM1000). """

    def test_decode(self):
        payload = struct.pack('>HBbhhH', 42, 3, -5, -110, 220, 15005)
        fields, tables, consumed = _decode_em950_specific(payload, 0)

        assert fields == {
            'PingNumber': 42, 'Mode': 3, 'PingQuality': -5,
            'ShipPitch_deg': pytest.approx(-1.1), 'TransducerPitch_deg': pytest.approx(2.2),
            'SurfaceVelocity_mps': pytest.approx(1500.5),
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeEm121aSpecific:
    """ Shared by ids 106 (EM121A) and 107 (EM121). """

    def test_decode(self):
        payload = struct.pack('>HBBBBBBBH', 7, 1, 32, 3, 2, 8, 0, 0, 15005)
        fields, tables, consumed = _decode_em121a_specific(payload, 0)

        assert fields == {
            'PingNumber': 7, 'Mode': 1, 'ValidBeams': 32, 'PulseLength': 3,
            'BeamWidth': 2, 'TxPower': 8, 'TxStatus': 0, 'RxStatus': 0,
            'SurfaceVelocity_mps': pytest.approx(1500.5),
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeSeamapSpecific:
    def test_decode(self):
        vals = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        payload = struct.pack('>11H', *vals)
        fields, tables, consumed = _decode_seamap_specific(payload, 0)

        assert fields == {
            'PortTransmitter0': pytest.approx(1.0), 'PortTransmitter1': pytest.approx(1.1),
            'StbdTransmitter0': pytest.approx(1.2), 'StbdTransmitter1': pytest.approx(1.3),
            'PortGain': pytest.approx(1.4), 'StbdGain': pytest.approx(1.5),
            'PortPulseLength': pytest.approx(1.6), 'StbdPulseLength': pytest.approx(1.7),
            'PressureDepth': pytest.approx(1.8), 'Altitude': pytest.approx(1.9),
            'Temperature': pytest.approx(2.0),
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeSeabatSpecific:
    def test_decode(self):
        payload = struct.pack('>HHBBBB', 42, 15005, 3, 100, 5, 6)
        fields, tables, consumed = _decode_seabat_specific(payload, 0)

        assert fields == {
            'PingNumber': 42, 'SurfaceVelocity_mps': pytest.approx(1500.5),
            'Mode': 3, 'SonarRange_m': 100, 'TransmitPower': 5, 'ReceiveGain': 6,
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeSBAmpSpecific:
    def test_decode(self):
        payload = struct.pack('>BBBBIh', 12, 30, 15, 50, 123456, -100)
        fields, tables, consumed = _decode_sb_amp_specific(payload, 0)

        assert fields == {
            'Hour': 12, 'Minute': 30, 'Second': 15, 'Hundredths': 50,
            'BlockNumber': 123456, 'AvgGateDepth': -100,
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeSeabatIISpecific:
    def test_decode(self):
        payload = struct.pack('>HHHHHHBB4x', 100, 14905, 7, 200, 10, 20, 15, 25)
        fields, tables, consumed = _decode_seabat_ii_specific(payload, 0)

        assert fields == {
            'PingNumber': 100, 'SurfaceVelocity_mps': pytest.approx(1490.5),
            'Mode': 7, 'SonarRange_m': 200, 'TransmitPower': 10, 'ReceiveGain': 20,
            'ForeAftBW_deg': pytest.approx(1.5), 'AthwartBW_deg': pytest.approx(2.5),
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeSeabeam2112Specific:
    def test_decode(self):
        # surface_velocity encoded as (raw + 130000) / 100.0 = 1490.0 -> raw = 19000
        payload = struct.pack('>BHBBBBB4s2x', 5, 19000, ord('V'), 10, 3, 2, 2, b"WMTB")
        fields, tables, consumed = _decode_seabeam_2112_specific(payload, 0)

        assert fields == {
            'Mode': 5, 'SurfaceVelocity_mps': pytest.approx(1490.0),
            'SsvSource': ord('V'), 'PingGain_dB': 10, 'PulseWidth_ms': 3,
            'TransmitterAttenuation_dB': 2, 'NumberAlgorithms': 2,
            'AlgorithmOrder': "WMTB",
        }
        assert tables == {}
        assert consumed == len(payload)

    def test_decode_strips_trailing_nul_padding(self):
        payload = struct.pack('>BHBBBBB4s2x', 0, 0, 0, 0, 0, 0, 1, b"B\x00\x00\x00")
        fields, _tables, _consumed = _decode_seabeam_2112_specific(payload, 0)
        assert fields['AlgorithmOrder'] == "B"


class TestDecodeSeabat8101Specific:
    def test_decode(self):
        payload = struct.pack('>HHHHHHHBBBBHHHHB4x',
                               5, 15001, 1, 100, 3, 20, 200, 4, 6, 15, 25, 0, 0, 0, 0, 7)
        fields, tables, consumed = _decode_seabat8101_specific(payload, 0)

        assert fields == {
            'PingNumber': 5, 'SurfaceVelocity_mps': pytest.approx(1500.1),
            'Mode': 1, 'Range_m': 100, 'Power': 3, 'Gain': 20,
            'PulseWidth_us': 200, 'TvgSpreading': 4, 'TvgAbsorption': 6,
            'ForeAftBW_deg': pytest.approx(1.5), 'AthwartBW_deg': pytest.approx(2.5),
            'RangeFiltMin': 0, 'RangeFiltMax': 0, 'DepthFiltMin': 0, 'DepthFiltMax': 0,
            'Projector': 7,
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeReson8100Specific:
    def test_decode(self):
        payload = struct.pack('>HIIHHHHHHHHHHBBBBBhHHHHBHH2x',
                               10, 5, 12345, 8111, 200, 15001, 40000, 1000, 1, 100, 3, 20, 200,
                               4, 6, 15, 25, 9, -100, 0, 0, 0, 0, 1, 250, 2500)
        fields, tables, consumed = _decode_reson8100_specific(payload, 0)

        assert fields['Latency_ms'] == 10
        assert fields['PingNumber'] == 5
        assert fields['SonarID'] == 12345
        assert fields['SonarModel'] == 8111
        assert fields['SurfaceVelocity_mps'] == pytest.approx(1500.1)
        assert fields['ForeAftBW_deg'] == pytest.approx(1.5)
        assert fields['AthwartBW_deg'] == pytest.approx(2.5)
        assert fields['ProjectorAngle'] == -100
        assert fields['FiltersActive'] == 1
        assert fields['BeamSpacing_deg'] == pytest.approx(0.25)
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeGeoswathPlusSpecific:
    def test_decode(self):
        payload = struct.pack('>HHHHHIHHHHHHHHHHHHHHHHHH32x',
                               0, 1, 5410, 12345, 2, 7, 3, 4, 5, 6, 7, 8,
                               30000, 30002, 100, 15000, 150, 250, 10, 2, 1, 3, 123, 125)
        fields, tables, consumed = _decode_geoswath_plus_specific(payload, 0)

        assert fields['DataSource'] == 0
        assert fields['Side'] == 1
        assert fields['ModelNumber'] == 5410
        assert fields['Frequency_Hz'] == pytest.approx(123450.0)
        assert fields['PingNumber'] == 7
        assert fields['MeanSV_mps'] == pytest.approx(1500.0)
        assert fields['SurfaceVelocity_mps'] == pytest.approx(1500.1)
        assert fields['SampleRate_Hz'] == pytest.approx(150000.0)
        assert fields['PulseLength_us'] == pytest.approx(150.0)
        assert fields['RangeUncertainty_m'] == pytest.approx(0.123)
        assert fields['AngleUncertainty_deg'] == pytest.approx(1.25)
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeKlein5410bssSpecific:
    def test_decode(self):
        payload = struct.pack('>HHHIIIIIIIIIIHHI32x',
                               0, 1, 5410, 455000000, 100000000, 9, 1000, 900, 0, 200,
                               3300, 25000, 1500123, 1, 0, 7)
        fields, tables, consumed = _decode_klein5410bss_specific(payload, 0)

        assert fields['DataSource'] == 0
        assert fields['Side'] == 1
        assert fields['ModelNumber'] == 5410
        assert fields['AcousticFrequency_Hz'] == pytest.approx(455000.0)
        assert fields['SamplingFrequency_Hz'] == pytest.approx(100000.0)
        assert fields['PingNumber'] == 9
        assert fields['NumSamples'] == 1000
        assert fields['NumRaaSamples'] == 900
        assert fields['Range'] == 200
        assert fields['FishDepth_V'] == pytest.approx(3.3)
        assert fields['FishAltitude_m'] == pytest.approx(25.0)
        assert fields['SoundSpeed_mps'] == pytest.approx(1500.123)
        assert fields['TxWaveform'] == 1
        assert fields['Altimeter'] == 0
        assert fields['RawDataConfig'] == 7
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeSassSpecific:
    """ Shared by ids 108 (SASS) and 112 (TypeIII SeaBeam) -- identical wire format. """

    def test_decode(self):
        payload = struct.pack('>6H', 1, 60, 61, 2, 100, 5)
        fields, tables, consumed = _decode_sass_specific(payload, 0)

        assert fields == {
            'LeftmostBeam': 1, 'RightmostBeam': 60, 'TotalBeams': 61,
            'NavMode': 2, 'PingNumber': 100, 'MissionNumber': 5,
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeDeltaTSpecific:
    def test_decode(self):
        payload = b"DT4\x00" + struct.pack('>B', 3) + struct.pack('>H', 1234) \
            + struct.pack('>2I', 1700000000, 250000000) \
            + struct.pack('>H', 500) \
            + struct.pack('>H', 120) \
            + struct.pack('>H', 12000) \
            + struct.pack('>H', 25) \
            + struct.pack('>H', 100) \
            + struct.pack('>H', 260) \
            + struct.pack('>H', 15000) \
            + struct.pack('>H', 1) \
            + struct.pack('>H', 170) \
            + struct.pack('>H', 500) \
            + struct.pack('>I', 99999) \
            + struct.pack('>B', 1) \
            + struct.pack('>H', 123) \
            + struct.pack('>H', 456) \
            + struct.pack('>B', 1) \
            + struct.pack('>B', 3) \
            + struct.pack('>B', 5) \
            + struct.pack('>H', 10) \
            + struct.pack('>B', 7) \
            + struct.pack('>I', 1234) \
            + struct.pack('>B', 9) \
            + struct.pack('>I', 100) \
            + struct.pack('>B', 25) \
            + struct.pack('>B', 15) \
            + b"\x00" * 32

        fields, tables, consumed = _decode_delta_t_specific(payload, 0)

        assert fields['DecodeFileType'] == "DT4"
        assert fields['Version'] == 3
        assert fields['PingByteSize'] == 1234
        assert fields['SamplesPerBeam'] == 500
        assert fields['SectorSize_deg'] == pytest.approx(120.0)
        assert fields['StartAngle_deg'] == pytest.approx(-60.0)  # (12000/100) - 180
        assert fields['AngleIncrement_deg'] == pytest.approx(0.25)
        assert fields['AcousticRange_m'] == 100
        assert fields['AcousticFrequency_kHz'] == 260
        assert fields['SoundVelocity_mps'] == pytest.approx(1500.0)
        assert fields['RangeResolution_cm'] == pytest.approx(1.0)
        assert fields['ProfileTiltAngle_deg'] == pytest.approx(-10.0)  # 170 - 180
        assert fields['RepetitionRate_ms'] == pytest.approx(500.0)
        assert fields['PingNumber'] == 99999
        assert fields['IntensityFlag'] == 1
        assert fields['PingLatency_s'] == pytest.approx(0.0123)
        assert fields['DataLatency_s'] == pytest.approx(0.0456)
        assert fields['SampleRateFlag'] == 1
        assert fields['OptionFlags'] == 3
        assert fields['NumPingsAvg'] == 5
        assert fields['CenterPingTimeOffset_s'] == pytest.approx(0.001)
        assert fields['UserDefinedByte'] == 7
        assert fields['Altitude_m'] == pytest.approx(12.34)
        assert fields['ExternalSensorFlags'] == 9
        assert fields['PulseLength_s'] == pytest.approx(0.0001)
        assert fields['ForeAftBeamwidth_deg'] == pytest.approx(2.5)
        assert fields['AthwartshipsBeamwidth_deg'] == pytest.approx(1.5)
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeR2SonicSpecific:
    """
    Whitebox test for the ping-level _decode_r2sonic_specific() (ids
    151/152/153) -- distinct from the unrelated, smaller
    _decode_r2sonic_imagery_specific() decoder used inside the intensity
    series subrecord (id 21).
    """

    def test_decode(self):
        payload = b"2024".ljust(12, b'\x00') + b"100017".ljust(12, b'\x00') \
            + struct.pack(
                '>10I2iI6IiI2H6i6i2Ii',
                1700000000, 250000000, 42,
                100000, 150000, 400000000, 21000, 500000, 1000000, 2000000,
                -500000, 750000,
                7,
                80000, 500000, 200000, 2500, 30000, 1200,
                -250000,
                3,
                0, 256,
                1000000, -2000000, 3000000, 0, 0, 0,
                500000, 0, 0, 0, 0, -1000000,
                100000, 2000000,
                -300000)
        payload += b"\x00" * 32

        fields, tables, consumed = _decode_r2sonic_specific(payload, 0)

        assert fields['ModelNumber'] == "2024"
        assert fields['SerialNumber'] == "100017"
        assert fields['PingNumber'] == 42
        assert fields['PingPeriod_s'] == pytest.approx(0.1)
        assert fields['SoundSpeed_mps'] == pytest.approx(1500.0)
        assert fields['Frequency_Hz'] == pytest.approx(400000.0)
        assert fields['TxPower_dB'] == pytest.approx(210.0)
        assert fields['TxPulseWidth_s'] == pytest.approx(0.05)
        assert fields['TxBeamwidthVert_deg'] == pytest.approx(1.0)
        assert fields['TxBeamwidthHoriz_deg'] == pytest.approx(2.0)
        assert fields['TxSteeringVert_deg'] == pytest.approx(-0.5)
        assert fields['TxSteeringHoriz_deg'] == pytest.approx(0.75)
        assert fields['TxMiscInfo'] == 7
        assert fields['RxBandwidth_Hz'] == pytest.approx(8.0)
        assert fields['RxSampleRate_Hz'] == pytest.approx(500.0)
        assert fields['RxRange_m'] == pytest.approx(2.0)
        assert fields['RxGain_dB'] == pytest.approx(25.0)
        assert fields['RxSpreading'] == pytest.approx(30.0)
        assert fields['RxAbsorption_dBkm'] == pytest.approx(1.2)
        assert fields['RxMountTilt_deg'] == pytest.approx(-0.25)
        assert fields['RxMiscInfo'] == 3
        assert fields['Reserved'] == 0
        assert fields['NumBeams'] == 256
        assert fields['A0MoreInfo'] == pytest.approx([1.0, -2.0, 3.0, 0.0, 0.0, 0.0])
        assert fields['A2MoreInfo'] == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0, -1.0])
        assert fields['G0DepthGateMin_s'] == pytest.approx(0.1)
        assert fields['G0DepthGateMax_s'] == pytest.approx(2.0)
        assert fields['G0DepthGateSlope_deg'] == pytest.approx(-0.3)
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeReson7125Specific:
    def test_decode(self):
        payload = struct.pack(
            '>HI', 1, 7125) + b"\x00" * 16 + struct.pack(
            '>IIIH', 100, 200, 42, 0) + struct.pack(
            '>IIII', 400000000, 347830000, 80000000, 300) + struct.pack(
            '>III', 0, 1, 5000) + struct.pack(
            '>I', 0) + struct.pack(
            '>III', 40000000, 25000000, 10000000) + struct.pack(
            '>Ii', 18000, -300) + struct.pack(
            '>II', 12345, 1) + struct.pack(
            '>ii', -1500, 2500) + struct.pack(
            '>HH', 100, 100) + struct.pack(
            '>I', 0) + struct.pack(
            '>II', 0, 0) + struct.pack(
            '>II', 1, 2) + struct.pack(
            '>II', 1, 0) + struct.pack(
            '>I', 5) + struct.pack(
            '>HHHHH', 100, 0, 1500, 0, 1500) + struct.pack(
            '>I', 40000) + struct.pack(
            '>H', 15000) + struct.pack(
            '>I', 20000) + struct.pack(
            '>B', 1) + b"\x00" * 15 + struct.pack('>BB', 0, 1) + b"\x00" * 8

        fields, tables, consumed = _decode_reson7125_specific(payload, 0)

        assert fields['ProtocolVersion'] == 1
        assert fields['DeviceID'] == 7125
        assert fields['MajorSerialNumber'] == 100
        assert fields['MinorSerialNumber'] == 200
        assert fields['PingNumber'] == 42
        assert fields['MultiPingSeq'] == 0
        assert fields['Frequency_Hz'] == pytest.approx(400000.0)
        assert fields['SampleRate_Hz'] == pytest.approx(34783.0)
        assert fields['ReceiverBandwidth_Hz'] == pytest.approx(8000.0)
        assert fields['TxPulseWidth_s'] == pytest.approx(0.00003)
        assert fields['TxPulseTypeID'] == 0
        assert fields['TxPulseEnvelopeID'] == 1
        assert fields['TxPulseEnvelopeParam'] == pytest.approx(50.0)
        assert fields['TxPulseReserved'] == 0
        assert fields['MaxPingRate_pps'] == pytest.approx(40.0)
        assert fields['PingPeriod_s'] == pytest.approx(25.0)
        assert fields['Range_m'] == pytest.approx(100000.0)
        assert fields['Power_dB'] == pytest.approx(180.0)
        assert fields['Gain_dB'] == pytest.approx(-3.0)
        assert fields['ControlFlags'] == 12345
        assert fields['ProjectorID'] == 1
        assert fields['ProjectorSteerAnglVert_deg'] == pytest.approx(-1.5)
        assert fields['ProjectorSteerAnglHoriz_deg'] == pytest.approx(2.5)
        assert fields['ProjectorBeamWidthVert_deg'] == pytest.approx(1.0)
        assert fields['ProjectorBeamWidthHoriz_deg'] == pytest.approx(1.0)
        assert fields['ProjectorBeamFocalPt_m'] == pytest.approx(0.0)
        assert fields['ProjectorBeamWeightingWindowType'] == 0
        assert fields['ProjectorBeamWeightingWindowParam'] == 0
        assert fields['TransmitFlags'] == 1
        assert fields['HydrophoneID'] == 2
        assert fields['ReceivingBeamWeightingWindowType'] == 1
        assert fields['ReceivingBeamWeightingWindowParam'] == 0
        assert fields['ReceiveFlags'] == 5
        assert fields['ReceiveBeamWidth_deg'] == pytest.approx(1.0)
        assert fields['RangeFiltMin_m'] == pytest.approx(0.0)
        assert fields['RangeFiltMax_m'] == pytest.approx(150.0)
        assert fields['DepthFiltMin_m'] == pytest.approx(0.0)
        assert fields['DepthFiltMax_m'] == pytest.approx(150.0)
        assert fields['Absorption_dBkm'] == pytest.approx(40.0)
        assert fields['SoundVelocity_mps'] == pytest.approx(1500.0)
        assert fields['Spreading_dB'] == pytest.approx(20.0)
        assert fields['RawDataFrom7027'] == 1
        assert fields['SvSource'] == 0
        assert fields['LayerCompFlag'] == 1
        assert tables == {}
        assert consumed == len(payload) == 186


class TestDecodeEmRunTime:
    """
    Whitebox test for _decode_em_run_time() -- the shared t_gsfEMRunTime
    inline-block decoder reused by EM4_SPECIFIC (_decode_em4_specific())
    and, later, the EM3 "_RAW"-variant subrecords.
    """

    def test_decode(self):
        payload = struct.pack(
            '>HIIHHBBBBBBHHHHHbBBBBBHBBBBHhB',
            710, 1700000000, 250000000, 42, 123,
            1, 2, 3, 4, 5, 6,
            10, 500, 1234, 150, 15,
            -3,
            25, 50, 20, 30, 1,
            500,
            1, 65, 1, 65,
            500,
            -50, 2) + b"\x00" * 16

        fields, consumed = _decode_em_run_time(payload, 0)

        assert fields['ModelNumber'] == 710
        assert fields['PingCounter'] == 42
        assert fields['SerialNumber'] == 123
        assert fields['OperatorStationStatus'] == 1
        assert fields['ProcessingUnitStatus'] == 2
        assert fields['BspStatus'] == 3
        assert fields['HeadTransceiverStatus'] == 4
        assert fields['Mode'] == 5
        assert fields['FilterID'] == 6
        assert fields['MinDepth_m'] == pytest.approx(10.0)
        assert fields['MaxDepth_m'] == pytest.approx(500.0)
        assert fields['Absorption_dBkm'] == pytest.approx(12.34)
        assert fields['TxPulseLength_us'] == pytest.approx(150.0)
        assert fields['TxBeamWidth_deg'] == pytest.approx(1.5)
        assert fields['TxPowerReMax_dB'] == pytest.approx(-3.0)
        assert fields['RxBeamWidth_deg'] == pytest.approx(2.5)
        assert fields['RxBandwidth_Hz'] == pytest.approx(2500.0)
        assert fields['RxFixedGain_dB'] == pytest.approx(20.0)
        assert fields['TvgCrossOverAngle_deg'] == pytest.approx(30.0)
        assert fields['SsvSource'] == 1
        assert fields['MaxPortSwathWidth_m'] == 500
        assert fields['BeamSpacing'] == 1
        assert fields['MaxPortCoverage_deg'] == 65
        assert fields['Stabilization'] == 1
        assert fields['MaxStbdCoverage_deg'] == 65
        assert fields['MaxStbdSwathWidth_m'] == 500
        assert fields['TxAlongTilt_deg'] == pytest.approx(-0.5)
        assert fields['FilterID2'] == 2
        assert consumed == len(payload) == 63


class TestDecodeEmPuStatus:
    """ Whitebox test for _decode_em_pu_status() -- shared t_gsfEMPUStatus block. """

    def test_decode(self):
        payload = struct.pack('>BHbbh', 45, 63, -10, 12, -150) + b"\x00" * 16

        fields, consumed = _decode_em_pu_status(payload, 0)

        assert fields['PuCpuLoad_pct'] == pytest.approx(45.0)
        assert fields['SensorStatus'] == 63
        assert fields['AchievedPortCoverage_deg'] == -10
        assert fields['AchievedStbdCoverage_deg'] == 12
        assert fields['YawStabilization_deg'] == pytest.approx(-1.5)
        assert consumed == len(payload) == 23


class TestDecodeEm4Specific:
    """
    Whitebox test for _decode_em4_specific() (ids 133/134/135/149/157).
    Builds the run_time/pu_status inline sub-blocks via the already-tested
    _encode_em_run_time()/_encode_em_pu_status() (composing the payload
    from already-verified units, the way a real GSF encoder would), and
    hand-packs the rest via struct.pack for a true whitebox check of the
    top-level fields and the transmit-sector array.
    """

    def test_decode_with_two_sectors(self):
        fixed = struct.pack('>HHHHiHIIIi',
                             710, 1, 100, 15000, 100000,
                             256, 191, 125000000, 7, 0)
        fixed += b"\x00" * 16  # spare_1
        fixed += struct.pack('>H', 2)  # transmit_sectors

        sector0 = struct.pack('>hHIIIHBBI', 150, 1000, 5000, 0, 71000000, 4000, 1, 0, 14000000)
        sector0 += b"\x00" * 16
        sector1 = struct.pack('>hHIIIHBBI', -150, 0, 3000, 1000, 72000000, 4100, 0, 1, 15000000)
        sector1 += b"\x00" * 16

        run_time_fields = {
            'ModelNumber': 710, 'PingCounter': 42, 'SerialNumber': 123,
            'MinDepth_m': 10.0, 'MaxDepth_m': 500.0,
        }
        pu_status_fields = {'SensorStatus': 63, 'YawStabilization_deg': -1.5}

        payload = fixed + sector0 + sector1 + b"\x00" * 16 \
            + _encode_em_run_time(run_time_fields) + _encode_em_pu_status(pu_status_fields)

        fields, tables, consumed = _decode_em4_specific(payload, 0)

        assert fields['ModelNumber'] == 710
        assert fields['PingCounter'] == 1
        assert fields['SerialNumber'] == 100
        assert fields['SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert fields['TransducerDepth_m'] == pytest.approx(5.0)
        assert fields['ValidDetections'] == 256
        assert fields['SamplingFrequency_Hz'] == pytest.approx(191.0 + 125000000 / 4.0e9)
        assert fields['DopplerCorrScale'] == 7
        assert fields['VehicleDepth_m'] == pytest.approx(0.0)
        assert fields['RunTime.ModelNumber'] == 710
        assert fields['RunTime.MinDepth_m'] == pytest.approx(10.0)
        assert fields['PuStatus.SensorStatus'] == 63
        assert fields['PuStatus.YawStabilization_deg'] == pytest.approx(-1.5)

        assert len(tables['TxSectors']) == 2
        assert tables['TxSectors'].iloc[0]['TiltAngle_deg'] == pytest.approx(1.5)
        assert tables['TxSectors'].iloc[0]['SectorNumber'] == 0
        assert tables['TxSectors'].iloc[0]['CenterFrequency_Hz'] == pytest.approx(71000.0)
        assert tables['TxSectors'].iloc[1]['TiltAngle_deg'] == pytest.approx(-1.5)
        assert tables['TxSectors'].iloc[1]['SectorNumber'] == 1

        assert consumed == len(payload)

    def test_decode_zero_sectors(self):
        fixed = struct.pack('>HHHHiHIIIi', 122, 1, 100, 15000, 0, 400, 40, 0, 0, 0)
        fixed += b"\x00" * 16 + struct.pack('>H', 0) + b"\x00" * 16
        payload = fixed + _encode_em_run_time({}) + _encode_em_pu_status({})

        fields, tables, consumed = _decode_em4_specific(payload, 0)

        assert fields['ModelNumber'] == 122
        assert len(tables['TxSectors']) == 0
        assert consumed == len(payload)


class TestDecodeEm3RunTime:
    """
    Whitebox tests for _decode_em3_run_time() -- the OLDER gsfEM3RunTime
    inline sub-block used only by plain EM3_SPECIFIC (ids 118-132/139),
    distinct from _decode_em_run_time() (t_gsfEMRunTime, used by
    EM4_SPECIFIC/EM3Raw instead).
    """

    @staticmethod
    def _payload(port_swath=100, stbd_swath=0, port_coverage=60, stbd_coverage=0):
        return struct.pack('>H', 3000) + struct.pack('>2I', 1700000000, 250000000) \
            + struct.pack('>HH', 5, 100) + struct.pack('>I', 0) \
            + struct.pack('>BB', 1, 2) \
            + struct.pack('>HH', 5, 500) + struct.pack('>H', 50) + struct.pack('>H', 150) \
            + struct.pack('>H', 15) + struct.pack('>B', 0) + struct.pack('>B', 20) \
            + struct.pack('>B', 3) + struct.pack('>B', 10) + struct.pack('>B', 30) \
            + struct.pack('>B', 0) \
            + struct.pack('>H', port_swath) + struct.pack('>B', 1) \
            + struct.pack('>B', port_coverage) + struct.pack('>B', 0) \
            + struct.pack('>B', stbd_coverage) + struct.pack('>H', stbd_swath) \
            + struct.pack('>B', 2) + b"\x00" * 4

    def test_decode_fixed_fields(self):
        payload = self._payload()
        fields, consumed = _decode_em3_run_time(payload, 0)

        assert fields['ModelNumber'] == 3000
        assert fields['PingNumber'] == 5
        assert fields['SerialNumber'] == 100
        assert fields['SystemStatus'] == 0
        assert fields['Mode'] == 1
        assert fields['FilterID'] == 2
        assert fields['MinDepth_m'] == pytest.approx(5.0)
        assert fields['MaxDepth_m'] == pytest.approx(500.0)
        assert fields['Absorption_dBkm'] == pytest.approx(0.5)
        assert fields['PulseLength_us'] == pytest.approx(150.0)
        assert fields['TransmitBeamWidth_deg'] == pytest.approx(1.5)
        assert fields['ReceiveBeamWidth_deg'] == pytest.approx(2.0)
        assert fields['ReceiveBandwidth_Hz'] == pytest.approx(3 * 50)
        assert fields['ReceiveGain_dB'] == 10
        assert fields['CrossOverAngle_deg'] == 30
        assert fields['SsvSource'] == 0
        assert fields['BeamSpacing'] == 1
        assert fields['Stabilization'] == 0
        assert fields['HiloFreqAbsorpRatio'] == 2
        assert consumed == len(payload) == 49

    def test_swath_width_fallback_halves_when_stbd_zero(self):
        fields, _consumed = _decode_em3_run_time(self._payload(port_swath=100, stbd_swath=0), 0)
        assert fields['SwathWidth_m'] == 100
        assert fields['PortSwathWidth_m'] == 50
        assert fields['StbdSwathWidth_m'] == 50

    def test_swath_width_sums_when_stbd_nonzero(self):
        fields, _consumed = _decode_em3_run_time(self._payload(port_swath=100, stbd_swath=80), 0)
        assert fields['SwathWidth_m'] == 180
        assert fields['PortSwathWidth_m'] == 100
        assert fields['StbdSwathWidth_m'] == 80

    def test_coverage_sector_fallback_halves_when_stbd_zero(self):
        fields, _consumed = _decode_em3_run_time(self._payload(port_coverage=60, stbd_coverage=0), 0)
        assert fields['CoverageSector_deg'] == 60
        assert fields['PortCoverageSector_deg'] == 30
        assert fields['StbdCoverageSector_deg'] == 30

    def test_coverage_sector_sums_when_stbd_nonzero(self):
        fields, _consumed = _decode_em3_run_time(self._payload(port_coverage=60, stbd_coverage=20), 0)
        assert fields['CoverageSector_deg'] == 80
        assert fields['PortCoverageSector_deg'] == 60
        assert fields['StbdCoverageSector_deg'] == 20


class TestDecodeEm3Specific:
    """
    Whitebox tests for _decode_em3_specific() (ids 118-132/139) -- the
    variable-length, run_time_id-bitmask-gated family. See
    TestDecodeEm3RunTime for the inline sub-block's own field coverage;
    these tests focus on the top-level fields and the bitmask-driven
    presence/absence/nesting of run-time blocks.
    """

    _FIXED = struct.pack('>HHHHHHH', 3000, 5, 100, 15000, 600, 200, 15000) \
        + struct.pack('>h', 50) + struct.pack('>b', -1)

    def test_neither_bit_set_no_run_time_blocks(self):
        payload = self._FIXED + struct.pack('>I', 0)
        fields, tables, consumed = _decode_em3_specific(payload, 0)

        assert fields['ModelNumber'] == 3000
        assert fields['PingNumber'] == 5
        assert fields['SerialNumber'] == 100
        assert fields['SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert fields['TransducerDepth_m'] == pytest.approx(6.0)
        assert fields['ValidBeams'] == 200
        assert fields['SampleRate_Hz'] == 15000
        assert fields['DepthDifference_m'] == pytest.approx(0.5)
        assert fields['OffsetMultiplier'] == -1
        assert len(tables['RunTime']) == 0
        assert consumed == len(payload)

    def test_only_bit0_set_one_head0_row(self):
        run_time_bytes = TestDecodeEm3RunTime._payload()
        payload = self._FIXED + struct.pack('>I', 0x1) + run_time_bytes
        _fields, tables, consumed = _decode_em3_specific(payload, 0)

        assert len(tables['RunTime']) == 1
        assert tables['RunTime'].iloc[0]['Head'] == 0
        assert tables['RunTime'].iloc[0]['ModelNumber'] == 3000
        assert consumed == len(payload)

    def test_bit1_alone_without_bit0_yields_no_run_time_blocks(self):
        # Mirrors DecodeEM3Specific()'s nesting: bit 1 is only inspected
        # inside the "if bit 0" branch, so bit 1 alone is a no-op.
        payload = self._FIXED + struct.pack('>I', 0x2)
        _fields, tables, consumed = _decode_em3_specific(payload, 0)

        assert len(tables['RunTime']) == 0
        assert consumed == len(payload)

    def test_both_bits_set_two_rows(self):
        head0_bytes = TestDecodeEm3RunTime._payload()
        head1_bytes = TestDecodeEm3RunTime._payload(port_swath=50, stbd_swath=50)
        payload = self._FIXED + struct.pack('>I', 0x3) + head0_bytes + head1_bytes
        _fields, tables, consumed = _decode_em3_specific(payload, 0)

        assert len(tables['RunTime']) == 2
        assert tables['RunTime'].iloc[0]['Head'] == 0
        assert tables['RunTime'].iloc[1]['Head'] == 1
        assert tables['RunTime'].iloc[1]['PortSwathWidth_m'] == 50
        assert consumed == len(payload)


class TestDecodeEm3RawSpecific:
    """
    Whitebox test for _decode_em3raw_specific() (ids 140-148). Like
    TestDecodeEm4Specific, builds the run_time/pu_status inline sub-blocks
    via the already-tested _encode_em_run_time()/_encode_em_pu_status(),
    and hand-packs the rest via struct.pack for a true whitebox check of
    the top-level fields (which differ from EM4_SPECIFIC's: no
    doppler_corr_scale, depth_difference after vehicle_depth, signed byte
    offset_multiplier) and the transmit-sector array (8 fields, no
    mean_absorption -- one fewer than EM4's sector struct).
    """

    def test_decode_with_two_sectors(self):
        fixed = struct.pack('>HHHHiHIIih',
                             300, 1, 100, 15000, 650000,
                             256, 191, 125000000, -1500, 50)
        fixed += struct.pack('>b', -1)  # offset_multiplier
        fixed += b"\x00" * 16  # spare_1
        fixed += struct.pack('>H', 2)  # transmit_sectors

        sector0 = struct.pack('>hHIII', 150, 1000, 5000, 0, 71000000) \
            + struct.pack('>BB', 1, 0) + struct.pack('>I', 14000000) + b"\x00" * 16
        sector1 = struct.pack('>hHIII', -150, 0, 3000, 1000, 72000000) \
            + struct.pack('>BB', 0, 1) + struct.pack('>I', 15000000) + b"\x00" * 16

        run_time_fields = {
            'ModelNumber': 300, 'PingCounter': 42, 'SerialNumber': 123,
            'MinDepth_m': 10.0, 'MaxDepth_m': 500.0,
        }
        pu_status_fields = {'SensorStatus': 63, 'YawStabilization_deg': -1.5}

        payload = fixed + sector0 + sector1 + b"\x00" * 16 \
            + _encode_em_run_time(run_time_fields) + _encode_em_pu_status(pu_status_fields)

        fields, tables, consumed = _decode_em3raw_specific(payload, 0)

        assert fields['ModelNumber'] == 300
        assert fields['PingCounter'] == 1
        assert fields['SerialNumber'] == 100
        assert fields['SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert fields['TransducerDepth_m'] == pytest.approx(650000 / 20000.0)
        assert fields['ValidDetections'] == 256
        assert fields['SamplingFrequency_Hz'] == pytest.approx(191.0 + 125000000 / 4.0e9)
        assert fields['VehicleDepth_m'] == pytest.approx(-1.5)
        assert fields['DepthDifference_m'] == pytest.approx(0.5)
        assert fields['OffsetMultiplier'] == -1
        assert fields['RunTime.ModelNumber'] == 300
        assert fields['RunTime.MinDepth_m'] == pytest.approx(10.0)
        assert fields['PuStatus.SensorStatus'] == 63
        assert fields['PuStatus.YawStabilization_deg'] == pytest.approx(-1.5)

        assert len(tables['TxSectors']) == 2
        assert tables['TxSectors'].iloc[0]['TiltAngle_deg'] == pytest.approx(1.5)
        assert tables['TxSectors'].iloc[0]['SectorNumber'] == 0
        assert tables['TxSectors'].iloc[0]['CenterFrequency_Hz'] == pytest.approx(71000.0)
        assert 'MeanAbsorption_dBkm' not in tables['TxSectors'].columns
        assert tables['TxSectors'].iloc[1]['TiltAngle_deg'] == pytest.approx(-1.5)
        assert tables['TxSectors'].iloc[1]['SectorNumber'] == 1

        assert consumed == len(payload)

    def test_decode_zero_sectors(self):
        fixed = struct.pack('>HHHHiHIIih', 120, 1, 100, 15000, 0, 400, 40, 0, 0, 0)
        fixed += struct.pack('>b', 0)
        fixed += b"\x00" * 16 + struct.pack('>H', 0) + b"\x00" * 16
        payload = fixed + _encode_em_run_time({}) + _encode_em_pu_status({})

        fields, tables, consumed = _decode_em3raw_specific(payload, 0)

        assert fields['ModelNumber'] == 120
        assert len(tables['TxSectors']) == 0
        assert consumed == len(payload)


class TestDecodeEchotracSpecific:
    """
    Whitebox test for _decode_echotrac_specific() -- the reference example
    for the single-beam sensor-specific subrecord registry
    (_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS), shared by ids 201 (Echotrac)
    and 202 (Bathy2000).
    """

    def test_decode(self):
        payload = struct.pack('>hBB', -5, 1, 2)
        fields, tables, consumed = _decode_echotrac_specific(payload, 0)

        assert fields == {'NavigationError': -5, 'MppSource': 1, 'TideSource': 2}
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeMgd77Specific:
    def test_decode(self):
        payload = struct.pack('>HHHHHI', 5, 1, 2, 3, 4, 123456)
        fields, tables, consumed = _decode_mgd77_specific(payload, 0)

        assert fields == {
            'TimeZoneCorr': 5, 'PositionTypeCode': 1, 'CorrectionCode': 2,
            'BathyTypeCode': 3, 'QualityCode': 4, 'TravelTime_sec': pytest.approx(12.3456),
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeBdbSpecific:
    def test_decode(self):
        payload = struct.pack('>i', 12345) + b'1UYSDW'
        fields, tables, consumed = _decode_bdb_specific(payload, 0)

        assert fields == {
            'DocNo': 12345, 'Eval': '1', 'Classification': 'U', 'TrackAdjFlag': 'Y',
            'SourceFlag': 'S', 'PtOrTrackLn': 'D', 'DatumFlag': 'W',
        }
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeNoshdbSpecific:
    def test_decode(self):
        payload = struct.pack('>HH', 7, 9)
        fields, tables, consumed = _decode_noshdb_specific(payload, 0)

        assert fields == {'TypeCode': 7, 'CartoCode': 9}
        assert tables == {}
        assert consumed == len(payload)


class TestDecodeResonTSeriesSpecific:
    def test_decode(self):
        payload = b""
        payload += struct.pack('>H', 1)                # ProtocolVersion
        payload += struct.pack('>I', 7125)              # DeviceID
        payload += struct.pack('>I', 2)                 # NumberDevices
        payload += struct.pack('>H', 1)                 # SystemEnumerator
        payload += b"\x00" * 10                          # reserved_1
        payload += struct.pack('>I', 100)               # MajorSerialNumber
        payload += struct.pack('>I', 200)               # MinorSerialNumber
        payload += struct.pack('>I', 12345)             # PingNumber
        payload += struct.pack('>H', 0)                 # MultiPingSeq
        payload += struct.pack('>I', 400000000)         # Frequency_Hz raw
        payload += struct.pack('>I', 347830000)         # SampleRate_Hz raw
        payload += struct.pack('>I', 8000000)           # ReceiverBandwidth_Hz raw
        payload += struct.pack('>I', 30000)             # TxPulseWidth_s raw
        payload += struct.pack('>I', 0)                 # TxPulseTypeID
        payload += struct.pack('>I', 1)                 # TxPulseEnvelopeID
        payload += struct.pack('>I', 5000000)           # TxPulseEnvelopeParam raw
        payload += struct.pack('>H', 1)                 # TxPulseMode
        payload += struct.pack('>H', 7)                 # TxPulseReserved
        payload += struct.pack('>I', 40000000)          # MaxPingRate_pps raw
        payload += struct.pack('>I', 25000000)          # PingPeriod_s raw
        payload += struct.pack('>I', 10000000)          # Range_m raw
        payload += struct.pack('>I', 18000)             # Power_dB raw
        payload += struct.pack('>i', -300)              # Gain_dB raw
        payload += struct.pack('>I', 3)                 # ControlFlags
        payload += struct.pack('>I', 1)                 # ProjectorID
        payload += struct.pack('>i', -1500)             # ProjectorSteerAnglVert_deg raw
        payload += struct.pack('>i', 2500)              # ProjectorSteerAnglHoriz_deg raw
        payload += struct.pack('>H', 100)               # ProjectorBeamWidthVert_deg raw
        payload += struct.pack('>H', 100)               # ProjectorBeamWidthHoriz_deg raw
        payload += struct.pack('>I', 0)                 # ProjectorBeamFocalPt_m raw
        payload += struct.pack('>I', 0)                 # ProjectorBeamWeightingWindowType
        payload += struct.pack('>I', 4)                 # ProjectorBeamWeightingWindowParam
        payload += struct.pack('>I', 5)                 # TransmitFlags
        payload += struct.pack('>I', 9)                 # HydrophoneID
        payload += struct.pack('>I', 1)                 # ReceivingBeamWeightingWindowType
        payload += struct.pack('>I', 7)                 # ReceivingBeamWeightingWindowParam
        payload += struct.pack('>I', 0x1234)            # ReceiveFlags
        payload += struct.pack('>H', 150)               # ReceiveBeamWidth_deg raw
        payload += struct.pack('>i', -105)              # RangeFiltMin_m raw
        payload += struct.pack('>i', 2005)              # RangeFiltMax_m raw
        payload += struct.pack('>i', -55)               # DepthFiltMin_m raw
        payload += struct.pack('>i', 3005)              # DepthFiltMax_m raw
        payload += struct.pack('>I', 15000)             # Absorption_dBkm raw
        payload += struct.pack('>H', 15005)             # SoundVelocity_mps low-precision raw
        payload += struct.pack('>B', 1)                 # SvSource
        payload += struct.pack('>I', 30500)             # Spreading_dB raw
        payload += struct.pack('>H', 2)                 # BeamSpacingMode
        payload += struct.pack('>H', 1)                 # SonarSourceMode
        payload += struct.pack('>B', 1)                 # CoverageMode
        payload += struct.pack('>I', 12050)             # CoverageAngle_deg raw
        payload += struct.pack('>i', -350)              # HorizontalReceiverSteeringAngle_deg raw
        payload += b"\x00" * 3                           # reserved_2
        payload += struct.pack('>I', 2)                 # UncertaintyType
        payload += struct.pack('>i', -5000)             # TransmitterSteeringAngle_rad raw
        payload += struct.pack('>i', 2000)              # AppliedRoll_rad raw
        payload += struct.pack('>H', 3)                 # DetectionAlgorithm
        payload += struct.pack('>I', 0xFF00)            # DetectionFlags
        payload += b"T50-S SN 12345".ljust(60, b"\x00")  # DeviceDescription
        payload += struct.pack('>I', 1500123456)        # SoundVelocity_mps high-precision raw
        payload += b"\x00" * 60                          # reserved_7027
        payload += struct.pack('>B', 1)                 # MatchFilterControl
        payload += struct.pack('>I', 38000000)          # MatchFilterStartFreq_Hz raw
        payload += struct.pack('>I', 42000000)          # MatchFilterEndFreq_Hz raw
        payload += struct.pack('>B', 2)                 # MatchFilterWindowType
        payload += struct.pack('>H', 5000)              # MatchFilterShadingValue raw
        payload += struct.pack('>I', 9000000)           # MatchFilterEffectivePulseWidth_s raw
        payload += b"\x00" * 52                          # reserved_7002
        payload += b"\x00" * 32                          # reserved_3
        payload += b"\x00" * 288                         # reserved_4

        fields, tables, consumed = _decode_reson_tseries_specific(payload, 0)

        assert fields['ProtocolVersion'] == 1
        assert fields['DeviceID'] == 7125
        assert fields['NumberDevices'] == 2
        assert fields['SystemEnumerator'] == 1
        assert fields['MajorSerialNumber'] == 100
        assert fields['MinorSerialNumber'] == 200
        assert fields['PingNumber'] == 12345
        assert fields['MultiPingSeq'] == 0
        assert fields['Frequency_Hz'] == pytest.approx(400000.0)
        assert fields['SampleRate_Hz'] == pytest.approx(34783.0)
        assert fields['ReceiverBandwidth_Hz'] == pytest.approx(800.0)
        assert fields['TxPulseWidth_s'] == pytest.approx(0.003)
        assert fields['TxPulseTypeID'] == 0
        assert fields['TxPulseEnvelopeID'] == 1
        assert fields['TxPulseEnvelopeParam'] == pytest.approx(50000.0)
        assert fields['TxPulseMode'] == 1
        assert fields['TxPulseReserved'] == 7
        assert fields['MaxPingRate_pps'] == pytest.approx(40.0)
        assert fields['PingPeriod_s'] == pytest.approx(25.0)
        assert fields['Range_m'] == pytest.approx(100000.0)
        assert fields['Power_dB'] == pytest.approx(180.0)
        assert fields['Gain_dB'] == pytest.approx(-3.0)
        assert fields['ControlFlags'] == 3
        assert fields['ProjectorID'] == 1
        assert fields['ProjectorSteerAnglVert_deg'] == pytest.approx(-1.5)
        assert fields['ProjectorSteerAnglHoriz_deg'] == pytest.approx(2.5)
        assert fields['ProjectorBeamWidthVert_deg'] == pytest.approx(1.0)
        assert fields['ProjectorBeamWidthHoriz_deg'] == pytest.approx(1.0)
        assert fields['ProjectorBeamFocalPt_m'] == pytest.approx(0.0)
        assert fields['ProjectorBeamWeightingWindowType'] == 0
        assert fields['ProjectorBeamWeightingWindowParam'] == 4
        assert fields['TransmitFlags'] == 5
        assert fields['HydrophoneID'] == 9
        assert fields['ReceivingBeamWeightingWindowType'] == 1
        assert fields['ReceivingBeamWeightingWindowParam'] == 7
        assert fields['ReceiveFlags'] == 0x1234
        assert fields['ReceiveBeamWidth_deg'] == pytest.approx(1.5)
        assert fields['RangeFiltMin_m'] == pytest.approx(-10.5)
        assert fields['RangeFiltMax_m'] == pytest.approx(200.5)
        assert fields['DepthFiltMin_m'] == pytest.approx(-5.5)
        assert fields['DepthFiltMax_m'] == pytest.approx(300.5)
        assert fields['Absorption_dBkm'] == pytest.approx(15.0)
        assert fields['SoundVelocity_mps'] == pytest.approx(1500.123456)
        assert fields['SvSource'] == 1
        assert fields['Spreading_dB'] == pytest.approx(30.5)
        assert fields['BeamSpacingMode'] == 2
        assert fields['SonarSourceMode'] == 1
        assert fields['CoverageMode'] == 1
        assert fields['CoverageAngle_deg'] == pytest.approx(120.5)
        assert fields['HorizontalReceiverSteeringAngle_deg'] == pytest.approx(-3.5)
        assert fields['UncertaintyType'] == 2
        assert fields['TransmitterSteeringAngle_rad'] == pytest.approx(-0.05)
        assert fields['AppliedRoll_rad'] == pytest.approx(0.02)
        assert fields['DetectionAlgorithm'] == 3
        assert fields['DetectionFlags'] == 0xFF00
        assert fields['DeviceDescription'] == "T50-S SN 12345"
        assert fields['MatchFilterControl'] == 1
        assert fields['MatchFilterStartFreq_Hz'] == pytest.approx(380000.0)
        assert fields['MatchFilterEndFreq_Hz'] == pytest.approx(420000.0)
        assert fields['MatchFilterWindowType'] == 2
        assert fields['MatchFilterShadingValue'] == pytest.approx(0.5)
        assert fields['MatchFilterEffectivePulseWidth_s'] == pytest.approx(9.0e-5)
        assert tables == {}
        assert consumed == len(payload) == 715

    def test_high_precision_sound_velocity_zero_keeps_low_precision(self):
        # When the 4-byte high-precision sound-velocity override is 0,
        # gsf_dec.c keeps the low-precision (*10) value instead.
        payload = b""
        payload += struct.pack('>H', 1) + struct.pack('>I', 7125) + struct.pack('>I', 1) + struct.pack('>H', 0)
        payload += b"\x00" * 10
        payload += struct.pack('>III', 0, 0, 0) + struct.pack('>H', 0)
        payload += struct.pack('>IIII', 0, 0, 0, 0)
        payload += struct.pack('>III', 0, 0, 0) + struct.pack('>HH', 0, 0)
        payload += struct.pack('>III', 0, 0, 0) + struct.pack('>Ii', 0, 0)
        payload += struct.pack('>II', 0, 0) + struct.pack('>ii', 0, 0) + struct.pack('>HH', 0, 0)
        payload += struct.pack('>I', 0) + struct.pack('>II', 0, 0) + struct.pack('>II', 0, 0)
        payload += struct.pack('>II', 0, 0) + struct.pack('>I', 0) + struct.pack('>H', 0)
        payload += struct.pack('>iiii', 0, 0, 0, 0)
        payload += struct.pack('>I', 0)                  # Absorption_dBkm raw
        payload += struct.pack('>H', 15005)              # SoundVelocity_mps low-precision raw = 1500.5
        payload += struct.pack('>B', 0)                  # SvSource
        payload += struct.pack('>I', 0)                  # Spreading_dB raw
        payload += struct.pack('>HH', 0, 0) + struct.pack('>B', 0)
        payload += struct.pack('>I', 0) + struct.pack('>i', 0)
        payload += b"\x00" * 3
        payload += struct.pack('>I', 0) + struct.pack('>i', 0) + struct.pack('>i', 0)
        payload += struct.pack('>H', 0) + struct.pack('>I', 0)
        payload += b"\x00" * 60                           # DeviceDescription
        payload += struct.pack('>I', 0)                  # SoundVelocity_mps high-precision raw = 0
        payload += b"\x00" * 60
        payload += struct.pack('>B', 0) + struct.pack('>II', 0, 0)
        payload += struct.pack('>B', 0) + struct.pack('>H', 0) + struct.pack('>I', 0)
        payload += b"\x00" * 52 + b"\x00" * 32 + b"\x00" * 288

        fields, _tables, consumed = _decode_reson_tseries_specific(payload, 0)

        assert fields['SoundVelocity_mps'] == pytest.approx(1500.5)
        assert consumed == 715


class TestDecodeQualityFlagsArraySynthetic:
    """
    Whitebox tests for _decode_quality_flags_array() -- the 2-bit packed
    per-beam quality flag decoder (gsf_dec.c's DecodeQualityFlagsArray()).
    """

    def test_four_beams_one_byte(self):
        # beam0=3 (11), beam1=1 (01), beam2=2 (10), beam3=0 (00):
        # byte = 11_01_10_00 = 0xD8.
        payload = bytes([0xD8])
        values = _decode_quality_flags_array(payload, 0, num_beams=4, subrecord_size=1)
        assert list(values) == [3, 1, 2, 0]

    def test_not_evenly_divisible_by_four(self):
        # 5 beams -> 2 bytes; byte0 = 11_10_01_00 (beams 0-3), byte1's top
        # 2 bits (11) are beam 4, the rest are padding zeros on encode.
        payload = bytes([0b11100100, 0b11000000])
        values = _decode_quality_flags_array(payload, 0, num_beams=5, subrecord_size=2)
        assert list(values) == [3, 2, 1, 0, 3]

    def test_truncated_subrecord_pads_remaining_beams_with_zero(self):
        # subrecord_size covers only 4 of 6 beams (gsf_dec.c: "not all the
        # beams were encoded, only read the encoded beams").
        payload = bytes([0xD8])
        values = _decode_quality_flags_array(payload, 0, num_beams=6, subrecord_size=1)
        assert list(values) == [3, 1, 2, 0, 0, 0]

    def test_offset_into_payload(self):
        payload = b"\x00\x00" + bytes([0xD8])
        values = _decode_quality_flags_array(payload, 2, num_beams=4, subrecord_size=1)
        assert list(values) == [3, 1, 2, 0]


class TestDecodeBRBIntensitySynthetic:
    """
    Whitebox tests for _decode_brb_intensity() -- the per-beam backscatter
    time series decoder. The fixed header and per-beam sample loop are
    sensor-agnostic; the sensor-specific imagery preamble between them
    (size varies by sensor_id, gsf_dec.c's DecodeBRBIntensity() switch) is
    exercised per family below, plus the "no preamble at all" default case
    that covers every other sensor.
    """

    #: gsf_dec.c's DecodeBRBIntensity() fixed header: bits_per_sample(1) +
    #: applied_corrections(4) + spare(16) = 21 bytes.
    _HEADER_SIZE = 21

    @staticmethod
    def _header(bits_per_sample, applied_corrections=0):
        return struct.pack('>B', bits_per_sample) + struct.pack('>I', applied_corrections) + b"\x00" * 16

    @classmethod
    def _preamble(cls, bits_per_sample, sensor_imagery=b"", applied_corrections=0):
        return cls._header(bits_per_sample, applied_corrections) + sensor_imagery

    @staticmethod
    def _beam(sample_count, detect_sample, start_range_samples, samples, fmt):
        header = struct.pack('>3H', sample_count, detect_sample, start_range_samples) + b"\x00" * 6
        return header + b"".join(struct.pack(fmt, v) for v in samples)

    def test_unlisted_sensor_id_decodes_with_no_preamble(self):
        # gsf_dec.c's switch default: sensor_size = 0 -- covers any sensor_id
        # gsflib doesn't special-case (e.g. SeaBat, SeaBeam, EM12/100/950/
        # 1000/121, GeoSwath, DeltaT), and any id this decoder doesn't
        # recognize at all.
        payload = self._preamble(8) + self._beam(1, 0, 0, [42], '>B')
        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=999)
        assert header == {'BitsPerSample': 8, 'AppliedCorrections': 0}
        assert beam_rows[0]['Samples'] == [42]
        assert consumed == len(payload)

    def test_zero_beams_returns_none(self):
        payload = self._preamble(8)
        assert _decode_brb_intensity(payload, 0, num_beams=0, sensor_id=999) is None

    def test_8_bit_samples_decoded_per_beam(self):
        payload = self._preamble(8) \
            + self._beam(3, 1, 100, [10, 20, 30], '>B') \
            + self._beam(2, 0, 50, [200, 201], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=2, sensor_id=999)

        assert header['BitsPerSample'] == 8
        assert len(beam_rows) == 2
        assert beam_rows[0] == {
            'SampleCount': 3, 'DetectSample': 1, 'StartRangeSamples': 100, 'Samples': [10, 20, 30]}
        assert beam_rows[1] == {
            'SampleCount': 2, 'DetectSample': 0, 'StartRangeSamples': 50, 'Samples': [200, 201]}
        assert consumed == len(payload)

    def test_16_bit_samples_decoded_per_beam(self):
        payload = self._preamble(16) + self._beam(2, 5, 10, [1000, 65000], '>H')

        _header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=999)

        assert beam_rows[0]['Samples'] == [1000, 65000]
        assert consumed == len(payload)

    def test_12_bit_packed_samples_decoded(self):
        # Two 12-bit samples pack into 3 bytes: sample1 = (b0<<4)|(b1>>4),
        # sample2 = ((b1&0x0F)<<8)|b2. Choose sample1=0xABC, sample2=0x123:
        # b0 = 0xAB, b1 = (0xC<<4)|(0x1) = 0xC1, b2 = 0x23.
        packed = bytes([0xAB, 0xC1, 0x23])
        payload = self._preamble(12) \
            + struct.pack('>3H', 2, 0, 0) + b"\x00" * 6 + packed

        _header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=999)

        assert beam_rows[0]['Samples'] == [0xABC, 0x123]
        assert consumed == len(payload)

    def test_12_bit_odd_sample_count_drops_trailing_half_sample(self):
        # sample_count=1 with 12-bit packing: only the first sample of the
        # pair is emitted (mirrors gsf_dec.c's "if (j+1 < sample_count)" guard).
        packed = bytes([0xAB, 0xC0, 0x00])  # sample1 = 0xABC; sample2 unused
        payload = self._preamble(12) \
            + struct.pack('>3H', 1, 0, 0) + b"\x00" * 6 + packed

        _header, beam_rows, _consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=999)

        assert beam_rows[0]['Samples'] == [0xABC]

    def test_kmall_preamble_skipped_as_pure_spare(self):
        payload = self._preamble(8, sensor_imagery=b"\x00" * 64) + self._beam(1, 0, 0, [7], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=156)

        assert header == {'BitsPerSample': 8, 'AppliedCorrections': 0}
        assert beam_rows[0]['Samples'] == [7]
        assert consumed == len(payload)

    @pytest.mark.parametrize("sensor_id", sorted(_SUBRECORD_EM3_IMAGERY_IDS))
    def test_em3_imagery_preamble_decoded(self, sensor_id):
        imagery = struct.pack('>HHHBBHhh', 100, 5, 200, 12, 34, 5000, -10, 2) + b"\x00" * 4
        payload = self._preamble(8, sensor_imagery=imagery) + self._beam(1, 0, 0, [9], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=sensor_id)

        assert header['RangeNorm_samples'] == 100
        assert header['StartTvgRamp_samples'] == 5
        assert header['StopTvgRamp_samples'] == 200
        assert header['BSNormal_dB'] == 12
        assert header['BSOblique_dB'] == 34
        assert header['MeanAbsorption_dBkm'] == pytest.approx(50.0)
        assert header['Offset'] == -10
        assert header['Scale'] == 2
        assert beam_rows[0]['Samples'] == [9]
        assert consumed == len(payload)

    @pytest.mark.parametrize("sensor_id", sorted(_SUBRECORD_EM4_IMAGERY_IDS))
    def test_em4_imagery_preamble_decoded(self, sensor_id):
        imagery = struct.pack('>IIHHHHHhhHHhh',
                               191, 500000000, 3000, 150, 100, 5, 200, -50, 120, 7, 250, -10, 10) \
            + b"\x00" * 20
        payload = self._preamble(8, sensor_imagery=imagery) + self._beam(1, 0, 0, [9], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=sensor_id)

        assert header['SamplingFrequency_Hz'] == pytest.approx(191.125)
        assert header['MeanAbsorption_dBkm'] == pytest.approx(30.0)
        assert header['TxPulseLength_us'] == 150
        assert header['RangeNorm_samples'] == 100
        assert header['StartTvgRamp_samples'] == 5
        assert header['StopTvgRamp_samples'] == 200
        assert header['BSNormal_dB'] == pytest.approx(-5.0)
        assert header['BSOblique_dB'] == pytest.approx(12.0)
        assert header['TxBeamWidth_deg'] == pytest.approx(0.7)
        assert header['TvgCrossOver_deg'] == pytest.approx(25.0)
        assert header['Offset'] == -10
        assert header['Scale'] == 10
        assert beam_rows[0]['Samples'] == [9]
        assert consumed == len(payload)

    @pytest.mark.parametrize("sensor_id", sorted(_SUBRECORD_RESON_SIZE_SPARE_IMAGERY_IDS))
    def test_reson_size_spare_imagery_preamble_decoded(self, sensor_id):
        imagery = struct.pack('>H', 42) + b"\x00" * 64
        payload = self._preamble(8, sensor_imagery=imagery) + self._beam(1, 0, 0, [9], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=sensor_id)

        assert header['Size'] == 42
        assert beam_rows[0]['Samples'] == [9]
        assert consumed == len(payload)

    @pytest.mark.parametrize("sensor_id", sorted(_SUBRECORD_RESON_8100_IMAGERY_IDS))
    def test_reson_8100_imagery_preamble_skipped_as_pure_spare(self, sensor_id):
        payload = self._preamble(8, sensor_imagery=b"\x00" * 8) + self._beam(1, 0, 0, [9], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=sensor_id)

        assert header == {'BitsPerSample': 8, 'AppliedCorrections': 0}
        assert beam_rows[0]['Samples'] == [9]
        assert consumed == len(payload)

    def test_klein5410bss_imagery_preamble_decoded(self):
        imagery = struct.pack('>7H', 1, 2, 10, 11, 12, 13, 14) + b"\x00" * 4
        payload = self._preamble(8, sensor_imagery=imagery) + self._beam(1, 0, 0, [9], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(
            payload, 0, num_beams=1, sensor_id=_SUBRECORD_KLEIN_5410_BSS_SPECIFIC)

        assert header['ResMode'] == 1
        assert header['TvgPage'] == 2
        assert header['BeamID'] == [10, 11, 12, 13, 14]
        assert beam_rows[0]['Samples'] == [9]
        assert consumed == len(payload)

    @pytest.mark.parametrize("sensor_id", sorted(_SUBRECORD_R2SONIC_IMAGERY_IDS))
    def test_r2sonic_imagery_preamble_decoded(self, sensor_id):
        imagery = struct.pack('>12s12siIIIIIIIIIiiIIIIIIIiIHH',
                               b"2024", b"SN123",
                               1700000000, 500000000, 7, 100000, 150000, 300000000,
                               2000, 500000, 800000, 900000, -100000, -200000, 3,
                               50000, 400000, 120000, 1000, 2000, 1500, -300000, 4, 0, 5) \
            + struct.pack('>6i', 1000000, 2000000, -3000000, 0, 0, 0) + b"\x00" * 32
        payload = self._preamble(8, sensor_imagery=imagery) + self._beam(1, 0, 0, [9], '>B')

        header, beam_rows, consumed = _decode_brb_intensity(payload, 0, num_beams=1, sensor_id=sensor_id)

        assert header['ModelNumber'] == "2024"
        assert header['SerialNumber'] == "SN123"
        assert header['PingNumber'] == 7
        assert header['SoundSpeed_mps'] == pytest.approx(1500.0)
        assert header['Frequency_Hz'] == pytest.approx(300000.0)
        assert header['NumBeams'] == 5
        assert header['MoreInfo'] == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(-3.0), 0.0, 0.0, 0.0]
        assert beam_rows[0]['Samples'] == [9]
        assert consumed == len(payload)


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
