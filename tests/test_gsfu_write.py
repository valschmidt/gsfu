#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for GSFU.gsfu's write/encode side.

Every encoder here is the ported inverse of a decoder already covered by
test_gsfu.py, so the primary correctness check throughout is a round trip:
encode a payload, then decode it back with the existing (already-verified-
against-real-data) decoder, and confirm the values match. A few tests also
check the encoders' own error handling (out-of-range values, missing
required fields, mismatched array lengths) and ported rounding/framing
quirks (the +/-0.501 rounding convention, gsfKMALLVersion always 0, etc.)
directly, independent of decode.
"""
import datetime
import struct

import pytest

from GSFU.gsfu import (
    DEFAULT_PING_SCALE_FACTORS,
    GSF_NULL_COURSE,
    GSF_NULL_DEPTH_CORRECTOR,
    GSF_NULL_HEADING,
    GSF_NULL_HEAVE,
    GSF_NULL_HEIGHT,
    GSF_NULL_PITCH,
    GSF_NULL_ROLL,
    GSF_NULL_SEP,
    GSF_NULL_SPEED,
    GSF_NULL_TIDE_CORRECTOR,
    GSF_VERSION,
    RecordType,
    _decode_attitude,
    _decode_header,
    _decode_kmall_specific,
    _decode_name_value_parameters,
    _decode_ping_array,
    _decode_scale_factors,
    _decode_sound_velocity_profile,
    _decode_swath_bathymetry_ping,
    _encode_attitude,
    _encode_header,
    _encode_kmall_specific,
    _encode_name_value_parameters,
    _encode_ping_array,
    _encode_scale_factors,
    _encode_sound_velocity_profile,
    _encode_swath_bathymetry_ping,
    _gsf_epoch,
    _gsf_round,
    gsf,
    gsf_checksum,
)


# ---------------------------------------------------------------------------
# _gsf_round / _gsf_epoch
# ---------------------------------------------------------------------------

class TestGsfRound:
    def test_positive_rounds_up_at_half(self):
        assert _gsf_round(2.5) == 3

    def test_negative_rounds_away_from_zero_at_half(self):
        assert _gsf_round(-2.5) == -3

    def test_plain_truncation_cases(self):
        assert _gsf_round(2.4) == 2
        assert _gsf_round(2.6) == 3
        assert _gsf_round(-2.4) == -2
        assert _gsf_round(-2.6) == -3

    def test_zero(self):
        assert _gsf_round(0.0) == 0


class TestGsfEpoch:
    def test_float_epoch(self):
        assert _gsf_epoch(1700000000.0) == (1700000000, 0)

    def test_float_epoch_with_fraction(self):
        sec, nsec = _gsf_epoch(1700000000.5)
        assert sec == 1700000000
        assert nsec == pytest.approx(500000000, abs=1)

    def test_naive_datetime_assumed_utc(self):
        dt = datetime.datetime(2023, 11, 14, 22, 13, 20)
        sec, nsec = _gsf_epoch(dt)
        expected = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        assert sec == int(expected)

    def test_aware_datetime(self):
        dt = datetime.datetime(2023, 11, 14, 22, 13, 20, tzinfo=datetime.timezone.utc)
        sec, nsec = _gsf_epoch(dt)
        assert sec == int(dt.timestamp())
        assert nsec == 0

    def test_isoformat_string_round_trips(self):
        dt = datetime.datetime(2023, 11, 14, 22, 13, 20, 500000, tzinfo=datetime.timezone.utc)
        sec, nsec = _gsf_epoch(dt.isoformat())
        assert sec == int(dt.timestamp())
        assert nsec == pytest.approx(500000000, abs=1000)


# ---------------------------------------------------------------------------
# _encode_header
# ---------------------------------------------------------------------------

class TestEncodeHeader:
    def test_default_version_decodes_back(self):
        scalars, _tables, _notes = _decode_header(_encode_header())
        assert scalars['Version'] == GSF_VERSION

    def test_explicit_version_decodes_back(self):
        scalars, _tables, _notes = _decode_header(_encode_header("GSF-v03.09"))
        assert scalars['Version'] == "GSF-v03.09"

    def test_payload_is_exactly_version_size(self):
        from GSFU.gsfu import GSF_VERSION_SIZE
        assert len(_encode_header()) == GSF_VERSION_SIZE


# ---------------------------------------------------------------------------
# _encode_name_value_parameters
# ---------------------------------------------------------------------------

class TestEncodeNameValueParameters:
    def test_round_trip(self):
        params = {"PLATFORM_TYPE": "SURFACE_SHIP", "FULL_RAW_DATA": "TRUE"}
        payload = _encode_name_value_parameters(1700000000.0, params)
        scalars, _tables, _notes = _decode_name_value_parameters(payload)

        assert scalars['PLATFORM_TYPE'] == "SURFACE_SHIP"
        assert scalars['FULL_RAW_DATA'] == "TRUE"

    def test_no_embedded_nul_in_decoded_values(self):
        payload = _encode_name_value_parameters(1700000000.0, {"A": "B"})
        scalars, _tables, _notes = _decode_name_value_parameters(payload)
        assert '\x00' not in scalars['A']

    def test_empty_params(self):
        payload = _encode_name_value_parameters(1700000000.0, {})
        scalars, _tables, _notes = _decode_name_value_parameters(payload)
        assert 'ParamTime' in scalars
        assert len(scalars) == 1


# ---------------------------------------------------------------------------
# _encode_sound_velocity_profile
# ---------------------------------------------------------------------------

class TestEncodeSoundVelocityProfile:
    def test_round_trip(self):
        payload = _encode_sound_velocity_profile(
            observation_time=1700000000.0, application_time=1700000100.0,
            latitude_deg=43.1, longitude_deg=-70.5,
            depth_m=[0.0, 10.0, 10000.0], sound_speed_mPerSec=[1500.0, 1500.5, 1490.25])

        scalars, tables, _notes = _decode_sound_velocity_profile(payload)

        assert scalars['Latitude_deg'] == pytest.approx(43.1)
        assert scalars['Longitude_deg'] == pytest.approx(-70.5)
        assert scalars['NumberPoints'] == 3
        assert list(tables['Profile']['Depth_m']) == pytest.approx([0.0, 10.0, 10000.0])
        assert list(tables['Profile']['SoundSpeed_mPerSec']) == pytest.approx([1500.0, 1500.5, 1490.25])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            _encode_sound_velocity_profile(0.0, 0.0, 0.0, 0.0, [1.0, 2.0], [1500.0])


# ---------------------------------------------------------------------------
# _encode_attitude
# ---------------------------------------------------------------------------

class TestEncodeAttitude:
    def test_round_trip(self):
        payload = _encode_attitude(
            attitude_time=[1700000000.0, 1700000000.1, 1700000000.2],
            pitch_deg=[-1.1, -1.0, -0.9], roll_deg=[0.4, 0.5, 0.3],
            heave_m=[0.2, 0.1, 0.15], heading_deg=[12.3, 12.4, 359.99])

        scalars, tables, _notes = _decode_attitude(payload)

        assert scalars['NumMeasurements'] == 3
        m = tables['Measurements']
        assert list(m['Pitch_deg']) == pytest.approx([-1.1, -1.0, -0.9])
        assert list(m['Roll_deg']) == pytest.approx([0.4, 0.5, 0.3])
        assert list(m['Heave_m']) == pytest.approx([0.2, 0.1, 0.15])
        assert list(m['Heading_deg']) == pytest.approx([12.3, 12.4, 359.99], abs=0.01)

    def test_base_time_is_first_sample(self):
        payload = _encode_attitude(
            attitude_time=[1700000005.0, 1700000005.5],
            pitch_deg=[0.0, 0.0], roll_deg=[0.0, 0.0],
            heave_m=[0.0, 0.0], heading_deg=[0.0, 0.0])
        (base_sec, base_nsec) = struct.unpack_from('>2I', payload, 0)
        assert base_sec == 1700000005

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            _encode_attitude([1.0, 2.0], [0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# _encode_scale_factors
# ---------------------------------------------------------------------------

class TestEncodeScaleFactors:
    def test_round_trip(self):
        scale_factors = {2: (100.0, 5.0, 0x20), 1: (1000.0, 0.0, 0x40)}
        payload = _encode_scale_factors(scale_factors)

        # Skip the leading 4-byte subrecord id+size word (that's the outer
        # framing _decode_scale_factors doesn't itself consume).
        table, consumed = _decode_scale_factors(payload, 4)

        assert table[1] == (1000.0, 0.0, 0x40)
        assert table[2] == (100.0, 5.0, 0x20)
        assert consumed == len(payload) - 4

    def test_entries_written_in_ascending_id_order(self):
        payload = _encode_scale_factors({5: (1.0, 0.0, 0), 1: (1.0, 0.0, 0), 3: (1.0, 0.0, 0)})
        # word0 = subrecord header, then each entry's first word has the id
        # in its top byte.
        ids_in_order = []
        pos = 8  # skip header word + numArraySubrecords
        for _ in range(3):
            word, = struct.unpack_from('>I', payload, pos)
            ids_in_order.append((word >> 24) & 0xFF)
            pos += 12
        assert ids_in_order == [1, 3, 5]


# ---------------------------------------------------------------------------
# _encode_ping_array
# ---------------------------------------------------------------------------

class TestEncodePingArray:
    def test_round_trip_unsigned_two_byte(self):
        payload = _encode_ping_array(1, [10.0, 10.5, 9.95], multiplier=100.0, offset=0.0,
                                      signed=False, width=2)
        values = _decode_ping_array(payload, 4, len(payload) - 4, 3, 100.0, 0.0, False, False)
        assert list(values) == pytest.approx([10.0, 10.5, 9.95])

    def test_round_trip_signed_with_offset(self):
        payload = _encode_ping_array(2, [-8.0, 5.0], multiplier=10.0, offset=5.0,
                                      signed=True, width=2)
        values = _decode_ping_array(payload, 4, len(payload) - 4, 2, 10.0, 5.0, True, False)
        assert list(values) == pytest.approx([-8.0, 5.0])

    def test_out_of_range_raises_value_error(self):
        # multiplier=1, offset=0, 1-byte unsigned: max representable is 255.
        with pytest.raises(ValueError):
            _encode_ping_array(9, [1000.0], multiplier=1.0, offset=0.0, signed=False, width=1)

    def test_subrecord_header_word_correct(self):
        payload = _encode_ping_array(7, [1.0, 2.0, 3.0], multiplier=1.0, offset=0.0,
                                      signed=False, width=1)
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 7
        assert word & 0xFFFFFF == 3  # 3 beams * 1 byte


# ---------------------------------------------------------------------------
# _encode_kmall_specific
# ---------------------------------------------------------------------------

class TestEncodeKmallSpecific:
    def test_round_trip_scalars(self):
        s = {'EchoSounderID': 712, 'DgmType': 1, 'DgmVersion': 3, 'SystemID': 71,
             'PingRate_Hz': 40.0, 'FreqRangeLowLim_Hz': 49000.0, 'Latitude_deg': 43.1}
        payload = _encode_kmall_specific(s)

        decoded, sector_rows, class_rows, consumed = _decode_kmall_specific(payload, 4)

        assert decoded['EchoSounderID'] == 712
        assert decoded['DgmType'] == 1
        assert decoded['PingRate_Hz'] == pytest.approx(40.0)
        assert decoded['FreqRangeLowLim_Hz'] == pytest.approx(49000.0)
        assert decoded['Latitude_deg'] == pytest.approx(43.1)
        assert sector_rows == []
        assert class_rows == []
        assert consumed == len(payload) - 4

    def test_gsf_kmall_version_always_zero(self):
        # Ported quirk: gsf_enc.c's EncodeKMALLSpecific() always writes 0
        # for gsfKMALLVersion, ignoring any caller value.
        payload = _encode_kmall_specific({'GSFKMALLVersion': 99})
        decoded, _sectors, _classes, _consumed = _decode_kmall_specific(payload, 4)
        assert decoded['GSFKMALLVersion'] == 0

    def test_missing_fields_default_to_zero(self):
        payload = _encode_kmall_specific({})
        decoded, _sectors, _classes, _consumed = _decode_kmall_specific(payload, 4)
        assert decoded['EchoSounderID'] == 0
        assert decoded['NumTxSectors'] == 0

    def test_tx_sectors_round_trip_and_count_derived_from_list(self):
        sectors = [
            {'TxSectorNumb': 0, 'CentreFreq_Hz': 49000.0, 'TiltAngleReTx_deg': 4.5},
            {'TxSectorNumb': 1, 'CentreFreq_Hz': 52000.0, 'TiltAngleReTx_deg': -3.2},
        ]
        # Deliberately wrong NumTxSectors in `s` -- must be ignored in favor
        # of len(sectors).
        payload = _encode_kmall_specific({'NumTxSectors': 99}, sector_rows=sectors)

        decoded, decoded_sectors, _classes, _consumed = _decode_kmall_specific(payload, 4)

        assert decoded['NumTxSectors'] == 2
        assert len(decoded_sectors) == 2
        assert decoded_sectors[0]['CentreFreq_Hz'] == pytest.approx(49000.0)
        assert decoded_sectors[1]['TiltAngleReTx_deg'] == pytest.approx(-3.2)

    def test_extra_detection_classes_round_trip(self):
        classes = [{'NumExtraDetInClass': 5, 'AlarmFlag': 1}]
        payload = _encode_kmall_specific({}, class_rows=classes)

        decoded, _sectors, decoded_classes, _consumed = _decode_kmall_specific(payload, 4)

        assert decoded['NumExtraDetectionClasses'] == 1
        assert decoded_classes == [{'NumExtraDetInClass': 5, 'AlarmFlag': 1}]

    def test_byte_length_matches_header_word(self):
        payload = _encode_kmall_specific({}, sector_rows=[{}], class_rows=[{}])
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 156
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_swath_bathymetry_ping
# ---------------------------------------------------------------------------

class TestEncodeSwathBathymetryPing:
    _SCALARS = {
        'PingTime': 1700000000.5, 'Longitude_deg': -70.5, 'Latitude_deg': 43.1,
        'NumberBeams': 3, 'CenterBeam': 1, 'PingFlags': 0,
        'TideCorrector_m': 0.1, 'DepthCorrector_m': 2.0,
        'Heading_deg': 12.3, 'Pitch_deg': -1.1, 'Roll_deg': 0.4, 'Heave_m': 0.2,
        'Course_deg': 10.0, 'Speed_kn': 5.0,
    }
    _BEAMS = {'Depth_m': [10.0, 10.5, 9.95], 'AcrossTrack_m': [-8.0, 0.0, 5.0]}

    def test_round_trip_scalars_and_beams(self):
        payload = _encode_swath_bathymetry_ping(self._SCALARS, self._BEAMS, major_version=3)

        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['NumberBeams'] == 3
        assert scalars['Longitude_deg'] == pytest.approx(-70.5)
        assert scalars['Latitude_deg'] == pytest.approx(43.1)
        assert scalars['Pitch_deg'] == pytest.approx(-1.1)
        assert list(tables['Beams']['Depth_m']) == pytest.approx([10.0, 10.5, 9.95])
        assert list(tables['Beams']['AcrossTrack_m']) == pytest.approx([-8.0, 0.0, 5.0])
        assert notes == []

    def test_major_version_2_omits_height_sep_gps_fields(self):
        payload_v2 = _encode_swath_bathymetry_ping(self._SCALARS, {}, major_version=2)
        payload_v3 = _encode_swath_bathymetry_ping(self._SCALARS, {}, major_version=3)
        # Height_m/SEP_m/GPSTideCorrector_m (4 bytes each) plus 2 spare bytes.
        assert len(payload_v3) - len(payload_v2) == 14

        scalars_v2, _t, _n = _decode_swath_bathymetry_ping(payload_v2, major_version=2, scale_factors={})
        scalars_v3, _t, _n = _decode_swath_bathymetry_ping(payload_v3, major_version=3, scale_factors={})
        assert 'Height_m' not in scalars_v2
        assert 'Height_m' in scalars_v3

    def test_missing_required_scalar_raises_keyerror(self):
        incomplete = dict(self._SCALARS)
        del incomplete['NumberBeams']
        with pytest.raises(KeyError):
            _encode_swath_bathymetry_ping(incomplete, self._BEAMS)

    def test_unknown_beams_column_raises_keyerror(self):
        with pytest.raises(KeyError):
            _encode_swath_bathymetry_ping(self._SCALARS, {'NotARealColumn': [1, 2, 3]})

    def test_omitted_optional_scalars_default_to_null_sentinels_not_zero(self):
        # Per gsf.h's "Define null values to be used for missing data": a
        # field left out of `scalars` must not silently become 0/0.0 --
        # several of these fields are equally valid at exactly zero.
        required_only = {
            'PingTime': self._SCALARS['PingTime'],
            'Longitude_deg': self._SCALARS['Longitude_deg'],
            'Latitude_deg': self._SCALARS['Latitude_deg'],
            'NumberBeams': self._SCALARS['NumberBeams'],
        }
        payload = _encode_swath_bathymetry_ping(required_only, {}, major_version=3)
        scalars, _tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['TideCorrector_m'] == pytest.approx(GSF_NULL_TIDE_CORRECTOR)
        assert scalars['DepthCorrector_m'] == pytest.approx(GSF_NULL_DEPTH_CORRECTOR)
        assert scalars['Heading_deg'] == pytest.approx(GSF_NULL_HEADING)
        assert scalars['Pitch_deg'] == pytest.approx(GSF_NULL_PITCH)
        assert scalars['Roll_deg'] == pytest.approx(GSF_NULL_ROLL)
        assert scalars['Heave_m'] == pytest.approx(GSF_NULL_HEAVE)
        assert scalars['Course_deg'] == pytest.approx(GSF_NULL_COURSE)
        assert scalars['Speed_kn'] == pytest.approx(GSF_NULL_SPEED)
        assert scalars['Height_m'] == pytest.approx(GSF_NULL_HEIGHT)
        assert scalars['SEP_m'] == pytest.approx(GSF_NULL_SEP)
        # No GSF_NULL_* sentinel is defined for these -- 0 is the only
        # available default.
        assert scalars['CenterBeam'] == 0
        assert scalars['GPSTideCorrector_m'] == pytest.approx(0.0)

    def test_explicit_zero_survives_round_trip_distinct_from_null_default(self):
        # A caller-supplied 0.0 (a real, known-zero measurement) must not
        # collapse into the same on-disk value as the null default above.
        scalars = dict(self._SCALARS, Course_deg=0.0, Speed_kn=0.0)
        payload = _encode_swath_bathymetry_ping(scalars, self._BEAMS, major_version=3)
        decoded, _tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert decoded['Course_deg'] == pytest.approx(0.0)
        assert decoded['Speed_kn'] == pytest.approx(0.0)

    def test_beam_flags_round_trip(self):
        beams = dict(self._BEAMS)
        beams['BeamFlags'] = [0, 0, 1]
        payload = _encode_swath_bathymetry_ping(self._SCALARS, beams)
        _scalars, tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})
        assert list(tables['Beams']['BeamFlags']) == [0, 0, 1]

    def test_kmall_specific_and_tx_sectors_round_trip(self):
        kmall_specific = {'EchoSounderID': 712, 'DgmType': 1}
        tx_sectors = [{'TxSectorNumb': 0, 'CentreFreq_Hz': 49000.0}]
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, kmall_specific=kmall_specific, tx_sectors=tx_sectors)

        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['KMALL.EchoSounderID'] == 712
        assert 'TxSectors' in tables
        assert len(tables['TxSectors']) == 1
        assert notes == []

    def test_scale_factors_override_changes_precision(self):
        override = dict(DEFAULT_PING_SCALE_FACTORS)
        override[1] = (10.0, 0.0, 4, False)  # coarser depth precision (0.1m)
        scalars = dict(self._SCALARS, NumberBeams=1)
        payload = _encode_swath_bathymetry_ping(
            scalars, {'Depth_m': [10.03]}, scale_factors=override)

        _scalars, tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        # 10.03 rounds to the nearest 0.1m under the coarser override.
        assert list(tables['Beams']['Depth_m']) == pytest.approx([10.0])


# ---------------------------------------------------------------------------
# gsf.write_record() / write_header() / etc. -- integration, real file I/O
# ---------------------------------------------------------------------------

class TestGsfWriteRecord:
    def test_write_record_round_trips_via_read_record_header(self, tmp_path):
        path = tmp_path / "out.gsf"
        G = gsf(str(path))
        G.write_record(RecordType.GSF_RECORD_COMMENT, b"0123456789")
        G.closeFile()

        G2 = gsf(str(path))
        G2.OpenFiletoRead()
        dataSize, readSize, data_id = G2.read_record_header()
        payload = G2.FID.read(dataSize)

        assert data_id.recordID == RecordType.GSF_RECORD_COMMENT
        assert data_id.checksumFlag is False
        assert payload == b"0123456789"
        assert dataSize == readSize == 10

    def test_write_record_with_checksum(self, tmp_path):
        path = tmp_path / "out.gsf"
        G = gsf(str(path))
        G.write_record(RecordType.GSF_RECORD_COMMENT, b"0123456789", checksum=True)
        G.closeFile()

        G2 = gsf(str(path))
        G2.OpenFiletoRead()
        dataSize, readSize, data_id = G2.read_record_header()
        checksum_bytes = G2.FID.read(4)
        payload = G2.FID.read(dataSize)

        assert data_id.checksumFlag is True
        assert readSize == dataSize + 4
        (stored_checksum,) = struct.unpack('>I', checksum_bytes)
        assert stored_checksum == gsf_checksum(payload)

    def test_write_record_auto_opens_file(self, tmp_path):
        path = tmp_path / "out.gsf"
        G = gsf(str(path))
        assert G.FID is None
        G.write_record(RecordType.GSF_RECORD_COMMENT, b"0123456789")
        assert G.FID is not None


class TestGsfWriteMethods:
    def test_write_header_sets_gsf_version(self, tmp_path):
        path = tmp_path / "out.gsf"
        G = gsf(str(path))
        assert G.gsfVersion is None
        G.write_header()
        assert G.gsfVersion == GSF_VERSION

    def test_full_file_round_trip(self, tmp_path, capsys):
        path = tmp_path / "out.gsf"
        G = gsf(str(path))
        G.write_header()
        G.write_processing_parameters({"PLATFORM_TYPE": "SURFACE_SHIP"}, param_time=1700000000.0)
        G.write_sound_velocity_profile(
            observation_time=1700000000.0, application_time=1700000100.0,
            latitude_deg=43.1, longitude_deg=-70.5,
            depth_m=[0.0, 10.0], sound_speed_mPerSec=[1500.0, 1500.5])
        G.write_attitude(
            attitude_time=[1700000000.0, 1700000000.1],
            pitch_deg=[-1.1, -1.0], roll_deg=[0.4, 0.5],
            heave_m=[0.2, 0.1], heading_deg=[12.3, 12.4])
        G.write_swath_bathymetry_ping(
            {'PingTime': 1700000000.5, 'Longitude_deg': -70.5, 'Latitude_deg': 43.1, 'NumberBeams': 2},
            {'Depth_m': [10.0, 10.5]})
        G.closeFile()

        G2 = gsf(str(path))
        G2.index_file()

        assert list(G2.Index['RecordType']) == [
            'GSF_RECORD_HEADER', 'GSF_RECORD_PROCESSING_PARAMETERS',
            'GSF_RECORD_SOUND_VELOCITY_PROFILE', 'GSF_RECORD_ATTITUDE',
            'GSF_RECORD_SWATH_BATHYMETRY_PING']
        assert int(G2.Index['TotalBytes'].sum()) == path.stat().st_size

        G3 = gsf(str(path))
        G3.print_records()
        captured = capsys.readouterr()
        assert "decode failed" not in captured.out
        assert "PLATFORM_TYPE" in captured.out
        assert "Depth_m" in captured.out

    def test_ping_write_uses_gsfversion_for_major_version(self, tmp_path):
        # write_header() with an explicit v2-style version should make the
        # subsequent ping omit the major_version > 2 fields.
        path = tmp_path / "out.gsf"
        G = gsf(str(path))
        G.write_header(version="GSF-v02.05")
        G.write_swath_bathymetry_ping(
            {'PingTime': 1700000000.0, 'Longitude_deg': 0.0, 'Latitude_deg': 0.0, 'NumberBeams': 1},
            {'Depth_m': [10.0]})
        G.closeFile()

        G2 = gsf(str(path))
        G2.index_file()
        ping_offset = int(
            G2.Index.loc[G2.Index['RecordType'] == 'GSF_RECORD_SWATH_BATHYMETRY_PING', 'ByteOffset'].iloc[0])
        G2.FID.seek(ping_offset)
        dataSize, _readSize, _data_id = G2.read_record_header()
        payload = G2.FID.read(dataSize)
        scalars, _tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=2, scale_factors={})
        assert 'Height_m' not in scalars
