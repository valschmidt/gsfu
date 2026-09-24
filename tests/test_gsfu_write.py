#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for GSFU.gsfu's write/encode side.

Every encoder here is the ported inverse of a decoder already covered by
test_gsfu.py, so the primary correctness check throughout is a round trip:
encode a payload, then decode it back with the existing decoder, and
confirm the values match. Most of those decoders are further verified
against real sample data in test_gsfu.py; for the handful that aren't
(no sample .gsf file carries that record type -- see each encoder's own
"Untested against a verified GSF file" docstring note), this round trip is
only a self-consistency check against this library's own decoder, not
independent confirmation against real-world bytes. A few tests also check
the encoders' own error handling (out-of-range values, missing required
fields, mismatched array lengths) and ported rounding/framing quirks (the
+/-0.501 rounding convention, gsfKMALLVersion always 0, etc.) directly,
independent of decode.
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
    _PING_SENSOR_SPECIFIC_CODECS,
    _decode_attitude,
    _decode_bdb_specific,
    _decode_cmp_sass_specific,
    _decode_comment,
    _decode_delta_t_specific,
    _decode_echotrac_specific,
    _decode_em3_run_time,
    _decode_em3_specific,
    _decode_em3raw_specific,
    _decode_em4_specific,
    _decode_em_pu_status,
    _decode_em_run_time,
    _decode_elac_mkii_specific,
    _decode_em12_specific,
    _decode_em100_specific,
    _decode_em121a_specific,
    _decode_em950_specific,
    _decode_geoswath_plus_specific,
    _decode_header,
    _decode_history,
    _decode_hv_navigation_error,
    _decode_klein5410bss_specific,
    _decode_kmall_specific,
    _decode_mgd77_specific,
    _decode_name_value_parameters,
    _decode_navigation_error,
    _decode_noshdb_specific,
    _decode_ping_array,
    _decode_quality_flags_array,
    _decode_r2sonic_specific,
    _decode_reson7125_specific,
    _decode_reson_tseries_specific,
    _decode_reson8100_specific,
    _decode_sass_specific,
    _decode_scale_factors,
    _decode_sb_amp_specific,
    _decode_seabat8101_specific,
    _decode_seabat_ii_specific,
    _decode_seabat_specific,
    _decode_seabeam_2112_specific,
    _decode_seabeam_specific,
    _decode_seamap_specific,
    _decode_single_beam_ping,
    _decode_sound_velocity_profile,
    _decode_swath_bathy_summary,
    _decode_swath_bathymetry_ping,
    _encode_attitude,
    _encode_bdb_specific,
    _encode_cmp_sass_specific,
    _encode_comment,
    _encode_delta_t_specific,
    _encode_echotrac_specific,
    _encode_em3_run_time,
    _encode_em3_specific,
    _encode_em3raw_specific,
    _encode_em4_specific,
    _encode_em_pu_status,
    _encode_em_run_time,
    _encode_elac_mkii_specific,
    _encode_em12_specific,
    _encode_em100_specific,
    _encode_em121a_specific,
    _encode_em950_specific,
    _encode_geoswath_plus_specific,
    _encode_header,
    _encode_history,
    _encode_hv_navigation_error,
    _encode_klein5410bss_specific,
    _encode_kmall_specific,
    _encode_mgd77_specific,
    _encode_name_value_parameters,
    _encode_navigation_error,
    _encode_noshdb_specific,
    _encode_ping_array,
    _encode_quality_flags_array,
    _encode_r2sonic_specific,
    _encode_reson7125_specific,
    _encode_reson_tseries_specific,
    _encode_reson8100_specific,
    _encode_sass_specific,
    _encode_scale_factors,
    _encode_sb_amp_specific,
    _encode_seabat8101_specific,
    _encode_seabat_ii_specific,
    _encode_seabat_specific,
    _encode_seabeam_2112_specific,
    _encode_seabeam_specific,
    _encode_seamap_specific,
    _encode_single_beam_ping,
    _encode_sound_velocity_profile,
    _encode_swath_bathy_summary,
    _encode_swath_bathymetry_ping,
    _gsf_epoch,
    _gsf_round,
    gsf,
    gsf_checksum,
    new_kmall_specific,
    new_kmall_tx_sector,
    new_swath_bathymetry_ping_scalars,
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
# _encode_swath_bathy_summary
# ---------------------------------------------------------------------------

class TestEncodeSwathBathySummary:
    def test_round_trip(self):
        payload = _encode_swath_bathy_summary(
            start_time=1700000000.0, end_time=1700003600.0,
            min_latitude_deg=42.9, min_longitude_deg=-70.6,
            max_latitude_deg=43.3, max_longitude_deg=-70.4,
            min_depth_m=5.0, max_depth_m=125.5)

        scalars, _tables, _notes = _decode_swath_bathy_summary(payload)

        assert scalars['MinLatitude_deg'] == pytest.approx(42.9)
        assert scalars['MaxLatitude_deg'] == pytest.approx(43.3)
        assert scalars['MinLongitude_deg'] == pytest.approx(-70.6)
        assert scalars['MaxLongitude_deg'] == pytest.approx(-70.4)
        assert scalars['MinDepth_m'] == pytest.approx(5.0)
        assert scalars['MaxDepth_m'] == pytest.approx(125.5)


# ---------------------------------------------------------------------------
# _encode_comment
# ---------------------------------------------------------------------------

class TestEncodeComment:
    def test_round_trip(self):
        payload = _encode_comment(1700000000.0, "this is a test comment")
        scalars, _tables, _notes = _decode_comment(payload)
        assert scalars['Comment'] == "this is a test comment"

    def test_empty_comment(self):
        payload = _encode_comment(1700000000.0, "")
        scalars, _tables, _notes = _decode_comment(payload)
        assert scalars['Comment'] == ""


# ---------------------------------------------------------------------------
# _encode_history
# ---------------------------------------------------------------------------

class TestEncodeHistory:
    def test_round_trip(self):
        payload = _encode_history(
            1700000000.0, host_name="host1", operator_name="vschmidt",
            command_line="kmall2gsf.py -f x.kmall -o x.gsf", comment="converted")

        scalars, _tables, _notes = _decode_history(payload)

        # _decode_history() doesn't strip the embedded NUL gsf_enc.c writes
        # into host_name/operator_name/command_line's counted size -- see
        # _encode_history()'s docstring.
        assert scalars['HostName'] == "host1\x00"
        assert scalars['OperatorName'] == "vschmidt\x00"
        assert scalars['CommandLine'] == "kmall2gsf.py -f x.kmall -o x.gsf\x00"
        assert scalars['Comment'] == "converted"


# ---------------------------------------------------------------------------
# _encode_navigation_error
# ---------------------------------------------------------------------------

class TestEncodeNavigationError:
    def test_round_trip(self):
        payload = _encode_navigation_error(
            1700000000.0, record_id=12345, longitude_error_m=1.3, latitude_error_m=-0.8)

        scalars, _tables, _notes = _decode_navigation_error(payload)

        assert scalars['RecordID'] == 12345
        assert scalars['LongitudeError_m'] == pytest.approx(1.3)
        assert scalars['LatitudeError_m'] == pytest.approx(-0.8)


# ---------------------------------------------------------------------------
# _encode_hv_navigation_error
# ---------------------------------------------------------------------------

class TestEncodeHvNavigationError:
    def test_round_trip(self):
        payload = _encode_hv_navigation_error(
            1700000000.0, record_id=54321, horizontal_error_m=0.35,
            vertical_error_m=0.12, sep_uncertainty_m=0.5, position_type="GPS")

        scalars, _tables, _notes = _decode_hv_navigation_error(payload)

        assert scalars['RecordID'] == 54321
        assert scalars['HorizontalError_m'] == pytest.approx(0.35)
        assert scalars['VerticalError_m'] == pytest.approx(0.12)
        assert scalars['SEPUncertainty_m'] == pytest.approx(0.5)
        assert scalars['PositionType'] == "GPS"

    def test_empty_position_type(self):
        payload = _encode_hv_navigation_error(
            1700000000.0, record_id=1, horizontal_error_m=0.0,
            vertical_error_m=0.0, sep_uncertainty_m=0.0)
        scalars, _tables, _notes = _decode_hv_navigation_error(payload)
        assert scalars['PositionType'] == ""


# ---------------------------------------------------------------------------
# _encode_single_beam_ping
# ---------------------------------------------------------------------------

class TestEncodeSingleBeamPing:
    def test_round_trip(self):
        payload = _encode_single_beam_ping(
            ping_time=1700000000.5, longitude_deg=-70.5, latitude_deg=43.1,
            tide_corrector_m=0.1, depth_corrector_m=-1.2, heading_deg=123.45,
            pitch_deg=-1.1, roll_deg=0.4, heave_m=0.2, depth_m=25.75,
            sound_speed_correction_m=0.05, positioning_system_type=3)

        scalars, _tables, notes = _decode_single_beam_ping(payload)

        assert scalars['Longitude_deg'] == pytest.approx(-70.5)
        assert scalars['Latitude_deg'] == pytest.approx(43.1)
        assert scalars['TideCorrector_m'] == pytest.approx(0.1)
        assert scalars['DepthCorrector_m'] == pytest.approx(-1.2)
        assert scalars['Heading_deg'] == pytest.approx(123.45)
        assert scalars['Pitch_deg'] == pytest.approx(-1.1)
        assert scalars['Roll_deg'] == pytest.approx(0.4)
        assert scalars['Heave_m'] == pytest.approx(0.2)
        assert scalars['Depth_m'] == pytest.approx(25.75)
        assert scalars['SoundSpeedCorrection_m'] == pytest.approx(0.05)
        assert scalars['PositioningSystemType'] == 3
        assert notes == []

    def test_sensor_specific_round_trip(self):
        fields = {'NavigationError': -5, 'MppSource': 1, 'TideSource': 2}
        payload = _encode_single_beam_ping(
            ping_time=1700000000.5, longitude_deg=-70.5, latitude_deg=43.1,
            tide_corrector_m=0.1, depth_corrector_m=-1.2, heading_deg=123.45,
            pitch_deg=-1.1, roll_deg=0.4, heave_m=0.2, depth_m=25.75,
            sound_speed_correction_m=0.05, sensor_specific=(201, fields))

        scalars, _tables, notes = _decode_single_beam_ping(payload)

        assert scalars['Echotrac.NavigationError'] == -5
        assert scalars['Echotrac.MppSource'] == 1
        assert scalars['Echotrac.TideSource'] == 2
        assert notes == []

    def test_noshdb_sensor_specific_round_trip(self):
        fields = {'TypeCode': 7, 'CartoCode': 9}
        payload = _encode_single_beam_ping(
            ping_time=1700000000.5, longitude_deg=-70.5, latitude_deg=43.1,
            tide_corrector_m=0.1, depth_corrector_m=-1.2, heading_deg=123.45,
            pitch_deg=-1.1, roll_deg=0.4, heave_m=0.2, depth_m=25.75,
            sound_speed_correction_m=0.05, sensor_specific=(205, fields))

        scalars, _tables, notes = _decode_single_beam_ping(payload)

        assert scalars['NOSHDB.TypeCode'] == 7
        assert scalars['NOSHDB.CartoCode'] == 9
        assert notes == []

    def test_sensor_specific_unregistered_id_raises_keyerror(self):
        with pytest.raises(KeyError):
            _encode_single_beam_ping(
                ping_time=1700000000.5, longitude_deg=-70.5, latitude_deg=43.1,
                tide_corrector_m=0.1, depth_corrector_m=-1.2, heading_deg=123.45,
                pitch_deg=-1.1, roll_deg=0.4, heave_m=0.2, depth_m=25.75,
                sound_speed_correction_m=0.05, sensor_specific=(999, {}))


# ---------------------------------------------------------------------------
# _encode_echotrac_specific
# ---------------------------------------------------------------------------

class TestEncodeEchotracSpecific:
    def test_round_trip(self):
        fields = {'NavigationError': -5, 'MppSource': 1, 'TideSource': 2}
        payload = _encode_echotrac_specific(201, fields)

        decoded, tables, consumed = _decode_echotrac_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_uses_given_id(self):
        # Same codec is registered for both 201 (Echotrac) and 202 (Bathy2000).
        payload = _encode_echotrac_specific(202, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 202
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_mgd77_specific
# ---------------------------------------------------------------------------

class TestEncodeMgd77Specific:
    def test_round_trip(self):
        fields = {
            'TimeZoneCorr': 5, 'PositionTypeCode': 1, 'CorrectionCode': 2,
            'BathyTypeCode': 3, 'QualityCode': 4, 'TravelTime_sec': 12.3456,
        }
        payload = _encode_mgd77_specific(203, fields)

        decoded, tables, consumed = _decode_mgd77_specific(payload, 4)

        assert decoded['TimeZoneCorr'] == 5
        assert decoded['QualityCode'] == 4
        assert decoded['TravelTime_sec'] == pytest.approx(12.3456)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_mgd77_specific(203, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 203
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_bdb_specific
# ---------------------------------------------------------------------------

class TestEncodeBdbSpecific:
    def test_round_trip(self):
        fields = {
            'DocNo': 12345, 'Eval': '1', 'Classification': 'U', 'TrackAdjFlag': 'Y',
            'SourceFlag': 'S', 'PtOrTrackLn': 'D', 'DatumFlag': 'W',
        }
        payload = _encode_bdb_specific(204, fields)

        decoded, tables, consumed = _decode_bdb_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_missing_flags_default_to_nul(self):
        payload = _encode_bdb_specific(204, {'DocNo': 1})
        decoded, _tables, _consumed = _decode_bdb_specific(payload, 4)
        assert decoded['Eval'] == '\x00'

    def test_subrecord_header_word_correct(self):
        payload = _encode_bdb_specific(204, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 204
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_noshdb_specific
# ---------------------------------------------------------------------------

class TestEncodeNoshdbSpecific:
    def test_round_trip(self):
        fields = {'TypeCode': 7, 'CartoCode': 9}
        payload = _encode_noshdb_specific(205, fields)

        decoded, tables, consumed = _decode_noshdb_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_noshdb_specific(205, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 205
        assert word & 0xFFFFFF == len(payload) - 4


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
# _encode_quality_flags_array
# ---------------------------------------------------------------------------

class TestEncodeQualityFlagsArray:
    def test_round_trip_multiple_of_four(self):
        payload = _encode_quality_flags_array([3, 1, 2, 0, 0, 2, 1, 3])
        values = _decode_quality_flags_array(payload, 4, num_beams=8, subrecord_size=len(payload) - 4)
        assert list(values) == [3, 1, 2, 0, 0, 2, 1, 3]

    def test_round_trip_not_multiple_of_four(self):
        payload = _encode_quality_flags_array([3, 2, 1, 0, 3])
        values = _decode_quality_flags_array(payload, 4, num_beams=5, subrecord_size=len(payload) - 4)
        assert list(values) == [3, 2, 1, 0, 3]

    def test_values_masked_to_two_bits(self):
        # gsf_enc.c's EncodeQualityFlagsArray() does `array[i] << shift`
        # with no masking of its own -- a caller-supplied value outside
        # 0-3 would corrupt adjacent beams' bits in the real C encoder.
        # This encoder masks to 2 bits first instead, so out-of-range
        # input degrades gracefully (mod 4) rather than corrupting beams.
        payload = _encode_quality_flags_array([7])
        values = _decode_quality_flags_array(payload, 4, num_beams=1, subrecord_size=len(payload) - 4)
        assert list(values) == [3]

    def test_subrecord_header_word_correct(self):
        payload = _encode_quality_flags_array([1, 2, 3, 0, 1])
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 15
        assert word & 0xFFFFFF == 2  # 5 beams -> ceil(5/4) = 2 bytes


# ---------------------------------------------------------------------------
# _encode_elac_mkii_specific
# ---------------------------------------------------------------------------

class TestEncodeElacMkIISpecific:
    def test_round_trip(self):
        fields = {'Mode': 5, 'PingNumber': 42, 'SoundVelocity_mps': 1500,
                  'PulseLength_hundredth_ms': 200, 'ReceiverGainStbd_dB': 10,
                  'ReceiverGainPort_dB': 12, 'Reserved': 0}
        payload = _encode_elac_mkii_specific(117, fields)

        decoded, tables, consumed = _decode_elac_mkii_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_elac_mkii_specific(117, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 117
        assert word & 0xFFFFFF == len(payload) - 4

    def test_missing_fields_default_to_zero(self):
        payload = _encode_elac_mkii_specific(117, {})
        decoded, _tables, _consumed = _decode_elac_mkii_specific(payload, 4)
        assert decoded['Mode'] == 0
        assert decoded['PingNumber'] == 0


# ---------------------------------------------------------------------------
# _encode_seabeam_specific
# ---------------------------------------------------------------------------

class TestEncodeSeabeamSpecific:
    def test_round_trip(self):
        fields = {'EclipseTime_tenths_s': 1234}
        payload = _encode_seabeam_specific(102, fields)

        decoded, tables, consumed = _decode_seabeam_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_seabeam_specific(102, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 102
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_em12_specific
# ---------------------------------------------------------------------------

class TestEncodeEM12Specific:
    def test_round_trip(self):
        fields = {'PingNumber': 100, 'Resolution': 1, 'PingQuality': 50,
                  'SoundVelocity_mps': 1500.3, 'Mode': 2}
        payload = _encode_em12_specific(103, fields)

        decoded, tables, consumed = _decode_em12_specific(payload, 4)

        assert decoded['PingNumber'] == 100
        assert decoded['Resolution'] == 1
        assert decoded['PingQuality'] == 50
        assert decoded['SoundVelocity_mps'] == pytest.approx(1500.3)
        assert decoded['Mode'] == 2
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_em12_specific(103, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 103
        assert word & 0xFFFFFF == len(payload) - 4

    def test_missing_fields_default_to_zero(self):
        payload = _encode_em12_specific(103, {})
        decoded, _tables, _consumed = _decode_em12_specific(payload, 4)
        assert decoded['PingNumber'] == 0
        assert decoded['SoundVelocity_mps'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _encode_em100_specific
# ---------------------------------------------------------------------------

class TestEncodeEM100Specific:
    def test_round_trip(self):
        fields = {'ShipPitch_deg': -1.23, 'TransducerPitch_deg': 0.45, 'Mode': 1,
                  'Power': 2, 'Attenuation': 3, 'TVG': 4, 'PulseLength': 5, 'Counter': 999}
        payload = _encode_em100_specific(104, fields)

        decoded, tables, consumed = _decode_em100_specific(payload, 4)

        assert decoded['ShipPitch_deg'] == pytest.approx(-1.23)
        assert decoded['TransducerPitch_deg'] == pytest.approx(0.45)
        assert decoded['Mode'] == 1
        assert decoded['Power'] == 2
        assert decoded['Attenuation'] == 3
        assert decoded['TVG'] == 4
        assert decoded['PulseLength'] == 5
        assert decoded['Counter'] == 999
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_em100_specific(104, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 104
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_cmp_sass_specific
# ---------------------------------------------------------------------------

class TestEncodeCmpSassSpecific:
    def test_round_trip(self):
        fields = {'SurfaceSoundVelocity_ftps': 4900.5, 'Heave_ftps': 2.5}
        payload = _encode_cmp_sass_specific(121, fields)

        decoded, tables, consumed = _decode_cmp_sass_specific(payload, 4)

        assert decoded['SurfaceSoundVelocity_ftps'] == pytest.approx(4900.5)
        assert decoded['Heave_ftps'] == pytest.approx(2.5)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_cmp_sass_specific(121, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 121
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_em950_specific
# ---------------------------------------------------------------------------

class TestEncodeEm950Specific:
    def test_round_trip(self):
        fields = {'PingNumber': 42, 'Mode': 3, 'PingQuality': -5,
                  'ShipPitch_deg': -1.1, 'TransducerPitch_deg': 2.2, 'SurfaceVelocity_mps': 1500.5}
        payload = _encode_em950_specific(105, fields)

        decoded, tables, consumed = _decode_em950_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_same_codec_registered_for_em950_and_em1000(self):
        assert _PING_SENSOR_SPECIFIC_CODECS[105][1] is _PING_SENSOR_SPECIFIC_CODECS[111][1]
        assert _PING_SENSOR_SPECIFIC_CODECS[105][2] is _PING_SENSOR_SPECIFIC_CODECS[111][2]

    def test_subrecord_header_word_uses_given_id(self):
        payload = _encode_em950_specific(111, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 111
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_em121a_specific
# ---------------------------------------------------------------------------

class TestEncodeEm121aSpecific:
    def test_round_trip(self):
        fields = {'PingNumber': 7, 'Mode': 1, 'ValidBeams': 32, 'PulseLength': 3,
                  'BeamWidth': 2, 'TxPower': 8, 'TxStatus': 0, 'RxStatus': 0,
                  'SurfaceVelocity_mps': 1500.5}
        payload = _encode_em121a_specific(106, fields)

        decoded, tables, consumed = _decode_em121a_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_same_codec_registered_for_em121a_and_em121(self):
        assert _PING_SENSOR_SPECIFIC_CODECS[106][1] is _PING_SENSOR_SPECIFIC_CODECS[107][1]
        assert _PING_SENSOR_SPECIFIC_CODECS[106][2] is _PING_SENSOR_SPECIFIC_CODECS[107][2]

    def test_subrecord_header_word_uses_given_id(self):
        payload = _encode_em121a_specific(107, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 107
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_seamap_specific
# ---------------------------------------------------------------------------

class TestEncodeSeamapSpecific:
    def test_round_trip(self):
        fields = {
            'PortTransmitter0': 1.0, 'PortTransmitter1': 1.1, 'StbdTransmitter0': 1.2,
            'StbdTransmitter1': 1.3, 'PortGain': 1.4, 'StbdGain': 1.5,
            'PortPulseLength': 1.6, 'StbdPulseLength': 1.7, 'PressureDepth': 1.8,
            'Altitude': 1.9, 'Temperature': 2.0,
        }
        payload = _encode_seamap_specific(109, fields)

        decoded, tables, consumed = _decode_seamap_specific(payload, 4)

        for key, value in fields.items():
            assert decoded[key] == pytest.approx(value)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_seamap_specific(109, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 109
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_seabat_specific
# ---------------------------------------------------------------------------

class TestEncodeSeabatSpecific:
    def test_round_trip(self):
        fields = {'PingNumber': 42, 'SurfaceVelocity_mps': 1500.5, 'Mode': 3,
                  'SonarRange_m': 100, 'TransmitPower': 5, 'ReceiveGain': 6}
        payload = _encode_seabat_specific(110, fields)

        decoded, tables, consumed = _decode_seabat_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_seabat_specific(110, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 110
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_sb_amp_specific
# ---------------------------------------------------------------------------

class TestEncodeSBAmpSpecific:
    def test_round_trip(self):
        fields = {'Hour': 12, 'Minute': 30, 'Second': 15, 'Hundredths': 50,
                  'BlockNumber': 123456, 'AvgGateDepth': -100}
        payload = _encode_sb_amp_specific(113, fields)

        decoded, tables, consumed = _decode_sb_amp_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_sb_amp_specific(113, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 113
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_seabat_ii_specific
# ---------------------------------------------------------------------------

class TestEncodeSeabatIISpecific:
    def test_round_trip(self):
        fields = {
            'PingNumber': 100, 'SurfaceVelocity_mps': 1490.5, 'Mode': 7,
            'SonarRange_m': 200, 'TransmitPower': 10, 'ReceiveGain': 20,
            'ForeAftBW_deg': 1.5, 'AthwartBW_deg': 2.5,
        }
        payload = _encode_seabat_ii_specific(114, fields)

        decoded, tables, consumed = _decode_seabat_ii_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_seabat_ii_specific(114, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 114
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_seabeam_2112_specific
# ---------------------------------------------------------------------------

class TestEncodeSeabeam2112Specific:
    def test_round_trip(self):
        fields = {
            'Mode': 5, 'SurfaceVelocity_mps': 1490.0, 'SsvSource': ord('V'),
            'PingGain_dB': 10, 'PulseWidth_ms': 3, 'TransmitterAttenuation_dB': 2,
            'NumberAlgorithms': 2, 'AlgorithmOrder': "WMTB",
        }
        payload = _encode_seabeam_2112_specific(116, fields)

        decoded, tables, consumed = _decode_seabeam_2112_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_algorithm_order_shorter_string_round_trips(self):
        fields = {'AlgorithmOrder': "B"}
        payload = _encode_seabeam_2112_specific(116, fields)
        decoded, _tables, _consumed = _decode_seabeam_2112_specific(payload, 4)
        assert decoded['AlgorithmOrder'] == "B"

    def test_subrecord_header_word_correct(self):
        payload = _encode_seabeam_2112_specific(116, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 116
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_seabat8101_specific
# ---------------------------------------------------------------------------

class TestEncodeSeabat8101Specific:
    def test_round_trip(self):
        fields = {
            'PingNumber': 5, 'SurfaceVelocity_mps': 1500.1, 'Mode': 1, 'Range_m': 100,
            'Power': 3, 'Gain': 20, 'PulseWidth_us': 200, 'TvgSpreading': 4,
            'TvgAbsorption': 6, 'ForeAftBW_deg': 1.5, 'AthwartBW_deg': 2.5,
            'RangeFiltMin': 0, 'RangeFiltMax': 0, 'DepthFiltMin': 0, 'DepthFiltMax': 0,
            'Projector': 7,
        }
        payload = _encode_seabat8101_specific(115, fields)

        decoded, tables, consumed = _decode_seabat8101_specific(payload, 4)

        assert decoded == pytest.approx(fields)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_seabat8101_specific(115, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 115
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_reson8100_specific
# ---------------------------------------------------------------------------

class TestEncodeReson8100Specific:
    def test_round_trip(self):
        fields = {
            'Latency_ms': 10, 'PingNumber': 5, 'SonarID': 12345, 'SonarModel': 8111,
            'Frequency_kHz': 200, 'SurfaceVelocity_mps': 1500.1, 'SampleRate_Hz': 40000,
            'PingRate_mHz': 1000, 'Mode': 1, 'Range_m': 100, 'Power': 3, 'Gain': 20,
            'PulseWidth_us': 200, 'TvgSpreading': 4, 'TvgAbsorption': 6,
            'ForeAftBW_deg': 1.5, 'AthwartBW_deg': 2.5, 'ProjectorType': 9,
            'ProjectorAngle': -100, 'RangeFiltMin': 0, 'RangeFiltMax': 0,
            'DepthFiltMin': 0, 'DepthFiltMax': 0, 'FiltersActive': 1,
            'Temperature_tenth_degC': 250, 'BeamSpacing_deg': 0.25,
        }
        payload = _encode_reson8100_specific(122, fields)

        decoded, tables, consumed = _decode_reson8100_specific(payload, 4)

        assert decoded == pytest.approx(fields)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_registered_under_every_reson8100_id(self):
        for subrecord_id in (122, 123, 124, 125, 126, 127):
            family_label, decode_fn, encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
            assert family_label == "Reson8100"
            assert decode_fn is _decode_reson8100_specific
            assert encode_fn is _encode_reson8100_specific

    def test_subrecord_header_word_uses_given_id(self):
        payload = _encode_reson8100_specific(127, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 127
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_geoswath_plus_specific
# ---------------------------------------------------------------------------

class TestEncodeGeoswathPlusSpecific:
    def test_round_trip(self):
        fields = {
            'DataSource': 0, 'Side': 1, 'ModelNumber': 5410, 'Frequency_Hz': 123450.0,
            'EchosounderType': 2, 'PingNumber': 7, 'NumNavSamples': 3,
            'NumAttitudeSamples': 4, 'NumHeadingSamples': 5, 'NumMiniSVSSamples': 6,
            'NumEchosounderSamples': 7, 'NumRaaSamples': 8, 'MeanSV_mps': 1500.0,
            'SurfaceVelocity_mps': 1500.1, 'ValidBeams': 100, 'SampleRate_Hz': 150000.0,
            'PulseLength_us': 150.0, 'PingLength_m': 250, 'TransmitPower': 10,
            'SidescanGainChannel': 2, 'Stabilization': 1, 'GpsQuality': 3,
            'RangeUncertainty_m': 0.123, 'AngleUncertainty_deg': 1.25,
        }
        payload = _encode_geoswath_plus_specific(136, fields)

        decoded, tables, consumed = _decode_geoswath_plus_specific(payload, 4)

        assert decoded == pytest.approx(fields)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_subrecord_header_word_correct(self):
        payload = _encode_geoswath_plus_specific(136, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 136
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_klein5410bss_specific
# ---------------------------------------------------------------------------

class TestEncodeKlein5410bssSpecific:
    def test_round_trip(self):
        fields = {
            'DataSource': 0, 'Side': 1, 'ModelNumber': 5410,
            'AcousticFrequency_Hz': 455000.0, 'SamplingFrequency_Hz': 100000.0,
            'PingNumber': 9, 'NumSamples': 1000, 'NumRaaSamples': 900,
            'ErrorFlags': 0, 'Range': 200, 'FishDepth_V': 3.3,
            'FishAltitude_m': 25.0, 'SoundSpeed_mps': 1500.123,
            'TxWaveform': 1, 'Altimeter': 0, 'RawDataConfig': 7,
        }
        payload = _encode_klein5410bss_specific(137, fields)

        decoded, tables, consumed = _decode_klein5410bss_specific(payload, 4)

        assert decoded == pytest.approx(fields)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_negative_fish_depth_raises(self):
        # Documented asymmetry: gsf_dec.c decodes FishDepth_V/FishAltitude_m/
        # SoundSpeed_mps as unsigned, so a negative value can't be
        # represented -- see _decode_klein5410bss_specific()'s docstring.
        with pytest.raises(struct.error):
            _encode_klein5410bss_specific(137, {'FishDepth_V': -1.0})

    def test_subrecord_header_word_correct(self):
        payload = _encode_klein5410bss_specific(137, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 137
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_sass_specific
# ---------------------------------------------------------------------------

class TestEncodeSassSpecific:
    def test_round_trip(self):
        fields = {
            'LeftmostBeam': 1, 'RightmostBeam': 60, 'TotalBeams': 61,
            'NavMode': 2, 'PingNumber': 100, 'MissionNumber': 5,
        }
        payload = _encode_sass_specific(108, fields)

        decoded, tables, consumed = _decode_sass_specific(payload, 4)

        assert decoded == fields
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_shared_by_both_ids(self):
        # Same codec registered for both 108 (SASS) and 112 (TypeIII SeaBeam).
        assert _PING_SENSOR_SPECIFIC_CODECS[108][1] is _PING_SENSOR_SPECIFIC_CODECS[112][1]
        assert _PING_SENSOR_SPECIFIC_CODECS[108][2] is _PING_SENSOR_SPECIFIC_CODECS[112][2]

        payload = _encode_sass_specific(112, {'PingNumber': 42})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 112
        decoded, _tables, _consumed = _decode_sass_specific(payload, 4)
        assert decoded['PingNumber'] == 42

    def test_subrecord_header_word_correct(self):
        payload = _encode_sass_specific(108, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 108
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_delta_t_specific
# ---------------------------------------------------------------------------

class TestEncodeDeltaTSpecific:
    def test_round_trip(self):
        fields = {
            'DecodeFileType': 'DT4', 'Version': 3, 'PingByteSize': 1234,
            'InterrogationTime': 1700000000.25, 'SamplesPerBeam': 500,
            'SectorSize_deg': 120.0, 'StartAngle_deg': -60.0, 'AngleIncrement_deg': 0.25,
            'AcousticRange_m': 100, 'AcousticFrequency_kHz': 260, 'SoundVelocity_mps': 1500.0,
            'RangeResolution_cm': 1.0, 'ProfileTiltAngle_deg': -10.0, 'RepetitionRate_ms': 500.0,
            'PingNumber': 99999, 'IntensityFlag': 1, 'PingLatency_s': 0.0123,
            'DataLatency_s': 0.0456, 'SampleRateFlag': 1, 'OptionFlags': 3,
            'NumPingsAvg': 5, 'CenterPingTimeOffset_s': 0.001, 'UserDefinedByte': 7,
            'Altitude_m': 12.34, 'ExternalSensorFlags': 9, 'PulseLength_s': 0.0001,
            'ForeAftBeamwidth_deg': 2.5, 'AthwartshipsBeamwidth_deg': 1.5,
        }
        payload = _encode_delta_t_specific(150, fields)

        decoded, tables, consumed = _decode_delta_t_specific(payload, 4)

        for key, value in fields.items():
            if key == 'InterrogationTime':
                continue
            assert decoded[key] == pytest.approx(value), key
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_interrogation_time_round_trips(self):
        payload = _encode_delta_t_specific(150, {'InterrogationTime': 1700000000.5})
        decoded, _tables, _consumed = _decode_delta_t_specific(payload, 4)
        assert decoded['InterrogationTime'].timestamp() == pytest.approx(1700000000.5, abs=1e-3)

    def test_missing_fields_default_to_zero(self):
        payload = _encode_delta_t_specific(150, {})
        decoded, _tables, _consumed = _decode_delta_t_specific(payload, 4)
        assert decoded['PingNumber'] == 0
        assert decoded['SoundVelocity_mps'] == pytest.approx(0.0)

    def test_subrecord_header_word_correct(self):
        payload = _encode_delta_t_specific(150, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 150
        assert word & 0xFFFFFF == len(payload) - 4


# ---------------------------------------------------------------------------
# _encode_kmall_specific
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _encode_r2sonic_specific
# ---------------------------------------------------------------------------

class TestEncodeR2SonicSpecific:
    def test_round_trip(self):
        fields = {
            'ModelNumber': "2024", 'SerialNumber': "100017",
            'PingTime': 1700000000.25, 'PingNumber': 42,
            'PingPeriod_s': 0.1, 'SoundSpeed_mps': 1500.0, 'Frequency_Hz': 400000.0,
            'TxPower_dB': 210.0, 'TxPulseWidth_s': 0.05,
            'TxBeamwidthVert_deg': 1.0, 'TxBeamwidthHoriz_deg': 2.0,
            'TxSteeringVert_deg': -0.5, 'TxSteeringHoriz_deg': 0.75, 'TxMiscInfo': 7,
            'RxBandwidth_Hz': 8.0, 'RxSampleRate_Hz': 500.0, 'RxRange_m': 2.0,
            'RxGain_dB': 25.0, 'RxSpreading': 30.0, 'RxAbsorption_dBkm': 1.2,
            'RxMountTilt_deg': -0.25, 'RxMiscInfo': 3, 'Reserved': 0, 'NumBeams': 256,
            'A0MoreInfo': [1.0, -2.0, 3.0, 0.0, 0.0, 0.0],
            'A2MoreInfo': [0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
            'G0DepthGateMin_s': 0.1, 'G0DepthGateMax_s': 2.0, 'G0DepthGateSlope_deg': -0.3,
        }
        payload = _encode_r2sonic_specific(153, fields)

        decoded, tables, consumed = _decode_r2sonic_specific(payload, 4)

        assert decoded['ModelNumber'] == "2024"
        assert decoded['SerialNumber'] == "100017"
        assert decoded['PingNumber'] == 42
        assert decoded['PingPeriod_s'] == pytest.approx(0.1)
        assert decoded['SoundSpeed_mps'] == pytest.approx(1500.0)
        assert decoded['Frequency_Hz'] == pytest.approx(400000.0)
        assert decoded['TxPower_dB'] == pytest.approx(210.0)
        assert decoded['TxPulseWidth_s'] == pytest.approx(0.05)
        assert decoded['TxBeamwidthVert_deg'] == pytest.approx(1.0)
        assert decoded['TxBeamwidthHoriz_deg'] == pytest.approx(2.0)
        assert decoded['TxSteeringVert_deg'] == pytest.approx(-0.5)
        assert decoded['TxSteeringHoriz_deg'] == pytest.approx(0.75)
        assert decoded['TxMiscInfo'] == 7
        assert decoded['RxBandwidth_Hz'] == pytest.approx(8.0)
        assert decoded['RxSampleRate_Hz'] == pytest.approx(500.0)
        assert decoded['RxRange_m'] == pytest.approx(2.0)
        assert decoded['RxGain_dB'] == pytest.approx(25.0)
        assert decoded['RxSpreading'] == pytest.approx(30.0)
        assert decoded['RxAbsorption_dBkm'] == pytest.approx(1.2)
        assert decoded['RxMountTilt_deg'] == pytest.approx(-0.25)
        assert decoded['RxMiscInfo'] == 3
        assert decoded['NumBeams'] == 256
        assert decoded['A0MoreInfo'] == pytest.approx([1.0, -2.0, 3.0, 0.0, 0.0, 0.0])
        assert decoded['A2MoreInfo'] == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0, -1.0])
        assert decoded['G0DepthGateMin_s'] == pytest.approx(0.1)
        assert decoded['G0DepthGateMax_s'] == pytest.approx(2.0)
        assert decoded['G0DepthGateSlope_deg'] == pytest.approx(-0.3)
        assert tables == {}
        assert consumed == len(payload) - 4

    def test_registered_under_every_r2sonic_id(self):
        for subrecord_id in (151, 152, 153):
            family_label, decode_fn, encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
            assert family_label == "R2Sonic"
            assert decode_fn is _decode_r2sonic_specific
            assert encode_fn is _encode_r2sonic_specific

    def test_subrecord_header_word_uses_given_id(self):
        payload = _encode_r2sonic_specific(151, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 151
        assert word & 0xFFFFFF == len(payload) - 4

    def test_missing_fields_default_to_zero(self):
        payload = _encode_r2sonic_specific(153, {})
        decoded, _tables, _consumed = _decode_r2sonic_specific(payload, 4)
        assert decoded['ModelNumber'] == ""
        assert decoded['PingNumber'] == 0
        assert decoded['A0MoreInfo'] == pytest.approx([0.0] * 6)


# ---------------------------------------------------------------------------
# _encode_reson7125_specific
# ---------------------------------------------------------------------------

class TestEncodeReson7125Specific:
    _FIELDS = {
        'ProtocolVersion': 1, 'DeviceID': 7125, 'MajorSerialNumber': 100,
        'MinorSerialNumber': 200, 'PingNumber': 42, 'MultiPingSeq': 0,
        'Frequency_Hz': 400000.0, 'SampleRate_Hz': 34783.0, 'ReceiverBandwidth_Hz': 8000.0,
        'TxPulseWidth_s': 0.00003, 'TxPulseTypeID': 0, 'TxPulseEnvelopeID': 1,
        'TxPulseEnvelopeParam': 50.0, 'TxPulseReserved': 0, 'MaxPingRate_pps': 40.0,
        'PingPeriod_s': 0.025, 'Range_m': 100.0, 'Power_dB': 180.0, 'Gain_dB': -3.0,
        'ControlFlags': 12345, 'ProjectorID': 1, 'ProjectorSteerAnglVert_deg': -1.5,
        'ProjectorSteerAnglHoriz_deg': 2.5, 'ProjectorBeamWidthVert_deg': 1.0,
        'ProjectorBeamWidthHoriz_deg': 1.0, 'ProjectorBeamFocalPt_m': 0.0,
        'ProjectorBeamWeightingWindowType': 0, 'ProjectorBeamWeightingWindowParam': 0,
        'TransmitFlags': 1, 'HydrophoneID': 2, 'ReceivingBeamWeightingWindowType': 1,
        'ReceivingBeamWeightingWindowParam': 0, 'ReceiveFlags': 5, 'ReceiveBeamWidth_deg': 1.0,
        'RangeFiltMin_m': 0.0, 'RangeFiltMax_m': 150.0, 'DepthFiltMin_m': 0.0,
        'DepthFiltMax_m': 150.0, 'Absorption_dBkm': 40.0, 'SoundVelocity_mps': 1500.0,
        'Spreading_dB': 20.0, 'RawDataFrom7027': 1, 'SvSource': 0, 'LayerCompFlag': 1,
    }

    def test_round_trip(self):
        payload = _encode_reson7125_specific(138, self._FIELDS)

        decoded, tables, consumed = _decode_reson7125_specific(payload, 4)

        assert decoded == pytest.approx(self._FIELDS)
        assert tables == {}
        assert consumed == len(payload) - 4 == 186

    def test_subrecord_header_word_correct(self):
        payload = _encode_reson7125_specific(138, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 138
        assert word & 0xFFFFFF == len(payload) - 4

    def test_missing_fields_default_to_zero(self):
        payload = _encode_reson7125_specific(138, {})
        decoded, _tables, _consumed = _decode_reson7125_specific(payload, 4)
        assert decoded['DeviceID'] == 0
        assert decoded['Gain_dB'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _encode_reson_tseries_specific
# ---------------------------------------------------------------------------

class TestEncodeResonTSeriesSpecific:
    _FIELDS = {
        'ProtocolVersion': 1, 'DeviceID': 7125, 'NumberDevices': 2, 'SystemEnumerator': 1,
        'MajorSerialNumber': 100, 'MinorSerialNumber': 200, 'PingNumber': 12345, 'MultiPingSeq': 0,
        'Frequency_Hz': 400000.0, 'SampleRate_Hz': 34783.0, 'ReceiverBandwidth_Hz': 800.0,
        'TxPulseWidth_s': 0.003, 'TxPulseTypeID': 0, 'TxPulseEnvelopeID': 1,
        'TxPulseEnvelopeParam': 50000.0, 'TxPulseMode': 1, 'TxPulseReserved': 7,
        'MaxPingRate_pps': 40.0, 'PingPeriod_s': 25.0, 'Range_m': 100000.0, 'Power_dB': 180.0,
        'Gain_dB': -3.0, 'ControlFlags': 3, 'ProjectorID': 1,
        'ProjectorSteerAnglVert_deg': -1.5, 'ProjectorSteerAnglHoriz_deg': 2.5,
        'ProjectorBeamWidthVert_deg': 1.0, 'ProjectorBeamWidthHoriz_deg': 1.0,
        'ProjectorBeamFocalPt_m': 0.0, 'ProjectorBeamWeightingWindowType': 0,
        'ProjectorBeamWeightingWindowParam': 4, 'TransmitFlags': 5, 'HydrophoneID': 9,
        'ReceivingBeamWeightingWindowType': 1, 'ReceivingBeamWeightingWindowParam': 7,
        'ReceiveFlags': 0x1234, 'ReceiveBeamWidth_deg': 1.5, 'RangeFiltMin_m': -10.5,
        'RangeFiltMax_m': 200.5, 'DepthFiltMin_m': -5.5, 'DepthFiltMax_m': 300.5,
        'Absorption_dBkm': 15.0, 'SoundVelocity_mps': 1500.123456, 'SvSource': 1,
        'Spreading_dB': 30.5, 'BeamSpacingMode': 2, 'SonarSourceMode': 1, 'CoverageMode': 1,
        'CoverageAngle_deg': 120.5, 'HorizontalReceiverSteeringAngle_deg': -3.5,
        'UncertaintyType': 2, 'TransmitterSteeringAngle_rad': -0.05, 'AppliedRoll_rad': 0.02,
        'DetectionAlgorithm': 3, 'DetectionFlags': 0xFF00, 'DeviceDescription': 'T50-S SN 12345',
        'MatchFilterControl': 1, 'MatchFilterStartFreq_Hz': 380000.0,
        'MatchFilterEndFreq_Hz': 420000.0, 'MatchFilterWindowType': 2,
        'MatchFilterShadingValue': 0.5, 'MatchFilterEffectivePulseWidth_s': 9.0e-5,
    }

    def test_round_trip(self):
        payload = _encode_reson_tseries_specific(155, self._FIELDS)

        decoded, tables, consumed = _decode_reson_tseries_specific(payload, 4)

        for key, value in self._FIELDS.items():
            if isinstance(value, float):
                assert decoded[key] == pytest.approx(value), key
            else:
                assert decoded[key] == value, key
        assert tables == {}
        assert consumed == len(payload) - 4 == 715

    def test_subrecord_header_word_correct(self):
        payload = _encode_reson_tseries_specific(155, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 155
        assert word & 0xFFFFFF == len(payload) - 4

    def test_missing_fields_default_to_zero(self):
        payload = _encode_reson_tseries_specific(155, {})
        decoded, _tables, _consumed = _decode_reson_tseries_specific(payload, 4)
        assert decoded['DeviceID'] == 0
        assert decoded['Gain_dB'] == pytest.approx(0.0)
        assert decoded['DeviceDescription'] == ""
        assert decoded['SoundVelocity_mps'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _encode_em_run_time / _encode_em_pu_status / _encode_em4_specific
# ---------------------------------------------------------------------------

class TestEncodeEmRunTime:
    def test_round_trip(self):
        fields = {
            'ModelNumber': 710, 'PingTime': 1700000000.25, 'PingCounter': 42, 'SerialNumber': 123,
            'OperatorStationStatus': 1, 'ProcessingUnitStatus': 2, 'BspStatus': 3,
            'HeadTransceiverStatus': 4, 'Mode': 5, 'FilterID': 6,
            'MinDepth_m': 10.0, 'MaxDepth_m': 500.0, 'Absorption_dBkm': 12.34,
            'TxPulseLength_us': 150.0, 'TxBeamWidth_deg': 1.5, 'TxPowerReMax_dB': -3.0,
            'RxBeamWidth_deg': 2.5, 'RxBandwidth_Hz': 2500.0, 'RxFixedGain_dB': 20.0,
            'TvgCrossOverAngle_deg': 30.0, 'SsvSource': 1, 'MaxPortSwathWidth_m': 500,
            'BeamSpacing': 1, 'MaxPortCoverage_deg': 65, 'Stabilization': 1,
            'MaxStbdCoverage_deg': 65, 'MaxStbdSwathWidth_m': 500,
            'TxAlongTilt_deg': -0.5, 'FilterID2': 2,
        }
        payload = _encode_em_run_time(fields)

        decoded, consumed = _decode_em_run_time(payload, 0)

        for key, value in fields.items():
            if key == 'PingTime':
                continue
            if isinstance(value, float):
                assert decoded[key] == pytest.approx(value)
            else:
                assert decoded[key] == value
        assert consumed == len(payload) == 63

    def test_missing_fields_default_to_zero(self):
        payload = _encode_em_run_time({})
        decoded, _consumed = _decode_em_run_time(payload, 0)
        assert decoded['ModelNumber'] == 0
        assert decoded['MinDepth_m'] == pytest.approx(0.0)


class TestEncodeEmPuStatus:
    def test_round_trip(self):
        fields = {'PuCpuLoad_pct': 45.0, 'SensorStatus': 63, 'AchievedPortCoverage_deg': -10,
                  'AchievedStbdCoverage_deg': 12, 'YawStabilization_deg': -1.5}
        payload = _encode_em_pu_status(fields)

        decoded, consumed = _decode_em_pu_status(payload, 0)

        assert decoded['PuCpuLoad_pct'] == pytest.approx(45.0)
        assert decoded['SensorStatus'] == 63
        assert decoded['AchievedPortCoverage_deg'] == -10
        assert decoded['AchievedStbdCoverage_deg'] == 12
        assert decoded['YawStabilization_deg'] == pytest.approx(-1.5)
        assert consumed == len(payload) == 23


class TestEncodeEm4Specific:
    def test_round_trip_with_sectors(self):
        fields = {
            'ModelNumber': 710, 'PingCounter': 1, 'SerialNumber': 100,
            'SurfaceVelocity_mps': 1500.0, 'TransducerDepth_m': 5.0, 'ValidDetections': 256,
            'SamplingFrequency_Hz': 191.03125, 'DopplerCorrScale': 7, 'VehicleDepth_m': 0.0,
            'RunTime.ModelNumber': 710, 'RunTime.MinDepth_m': 10.0,
            'PuStatus.SensorStatus': 63, 'PuStatus.YawStabilization_deg': -1.5,
        }
        sectors = [
            {'TiltAngle_deg': 1.5, 'FocusRange_m': 100.0, 'SignalLength_sec': 0.005,
             'TransmitDelay_sec': 0.0, 'CenterFrequency_Hz': 71000.0, 'MeanAbsorption_dBkm': 40.0,
             'WaveformID': 1, 'SectorNumber': 0, 'SignalBandwidth_Hz': 14000.0},
            {'TiltAngle_deg': -1.5, 'FocusRange_m': 0.0, 'SignalLength_sec': 0.003,
             'TransmitDelay_sec': 0.001, 'CenterFrequency_Hz': 72000.0, 'MeanAbsorption_dBkm': 41.0,
             'WaveformID': 0, 'SectorNumber': 1, 'SignalBandwidth_Hz': 15000.0},
        ]
        payload = _encode_em4_specific(133, fields, {'TxSectors': sectors})

        decoded, tables, consumed = _decode_em4_specific(payload, 4)

        assert decoded['ModelNumber'] == 710
        assert decoded['SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert decoded['TransducerDepth_m'] == pytest.approx(5.0)
        assert decoded['SamplingFrequency_Hz'] == pytest.approx(191.03125)
        assert decoded['DopplerCorrScale'] == 7
        assert decoded['RunTime.ModelNumber'] == 710
        assert decoded['RunTime.MinDepth_m'] == pytest.approx(10.0)
        assert decoded['PuStatus.SensorStatus'] == 63
        assert decoded['PuStatus.YawStabilization_deg'] == pytest.approx(-1.5)
        assert len(tables['TxSectors']) == 2
        assert tables['TxSectors'][0]['CenterFrequency_Hz'] == pytest.approx(71000.0)
        assert tables['TxSectors'][1]['TiltAngle_deg'] == pytest.approx(-1.5)
        assert consumed == len(payload) - 4

    def test_registered_under_every_em4_id(self):
        for subrecord_id in (133, 134, 135, 149, 157):
            family_label, decode_fn, encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
            assert family_label == "EM4"
            assert decode_fn is _decode_em4_specific
            assert encode_fn is _encode_em4_specific

    def test_subrecord_header_word_and_zero_sectors(self):
        payload = _encode_em4_specific(157, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 157
        assert word & 0xFFFFFF == len(payload) - 4

        _decoded, tables, consumed = _decode_em4_specific(payload, 4)
        assert tables == {'TxSectors': []}
        assert consumed == len(payload) - 4


class TestEncodeEm3RunTime:
    def test_round_trip(self):
        fields = {
            'ModelNumber': 3000, 'PingTime': 1700000000.25, 'PingNumber': 5, 'SerialNumber': 100,
            'SystemStatus': 7, 'Mode': 1, 'FilterID': 2, 'MinDepth_m': 5.0, 'MaxDepth_m': 500.0,
            'Absorption_dBkm': 0.5, 'PulseLength_us': 150.0, 'TransmitBeamWidth_deg': 1.5,
            'PowerReduction_dB': 0, 'ReceiveBeamWidth_deg': 2.0, 'ReceiveBandwidth_Hz': 150.0,
            'ReceiveGain_dB': 10, 'CrossOverAngle_deg': 30, 'SsvSource': 0,
            'PortSwathWidth_m': 100, 'BeamSpacing': 1, 'PortCoverageSector_deg': 60,
            'Stabilization': 0, 'StbdCoverageSector_deg': 20, 'StbdSwathWidth_m': 80,
            'HiloFreqAbsorpRatio': 2,
        }
        payload = _encode_em3_run_time(fields)

        decoded, consumed = _decode_em3_run_time(payload, 0)

        assert decoded['ModelNumber'] == 3000
        assert decoded['PingNumber'] == 5
        assert decoded['SerialNumber'] == 100
        assert decoded['SystemStatus'] == 7
        assert decoded['Mode'] == 1
        assert decoded['FilterID'] == 2
        assert decoded['MinDepth_m'] == pytest.approx(5.0)
        assert decoded['MaxDepth_m'] == pytest.approx(500.0)
        assert decoded['Absorption_dBkm'] == pytest.approx(0.5)
        assert decoded['PulseLength_us'] == pytest.approx(150.0)
        assert decoded['TransmitBeamWidth_deg'] == pytest.approx(1.5)
        assert decoded['ReceiveBeamWidth_deg'] == pytest.approx(2.0)
        assert decoded['ReceiveBandwidth_Hz'] == pytest.approx(150.0)
        assert decoded['ReceiveGain_dB'] == 10
        assert decoded['CrossOverAngle_deg'] == 30
        # stbd values are nonzero, so port/stbd are used as given (no
        # halving fallback) and swath_width/coverage_sector are their sum.
        assert decoded['PortSwathWidth_m'] == 100
        assert decoded['StbdSwathWidth_m'] == 80
        assert decoded['SwathWidth_m'] == 180
        assert decoded['PortCoverageSector_deg'] == 60
        assert decoded['StbdCoverageSector_deg'] == 20
        assert decoded['CoverageSector_deg'] == 80
        assert decoded['HiloFreqAbsorpRatio'] == 2
        assert consumed == len(payload) == 49

    def test_swath_width_and_coverage_sector_ignored_on_encode(self):
        # SwathWidth_m/CoverageSector_deg are derived, read-only fields
        # with no wire storage -- encode must not consume them.
        fields = {'PortSwathWidth_m': 100, 'StbdSwathWidth_m': 0,
                  'SwathWidth_m': 999999, 'CoverageSector_deg': 999999}
        payload = _encode_em3_run_time(fields)
        decoded, _consumed = _decode_em3_run_time(payload, 0)
        assert decoded['SwathWidth_m'] == 100  # from PortSwathWidth_m fallback, not 999999


class TestEncodeEm3Specific:
    _FIELDS = {
        'ModelNumber': 3000, 'PingNumber': 5, 'SerialNumber': 100,
        'SurfaceVelocity_mps': 1500.0, 'TransducerDepth_m': 6.0, 'ValidBeams': 200,
        'SampleRate_Hz': 15000, 'DepthDifference_m': 0.5, 'OffsetMultiplier': -1,
    }
    _HEAD0 = {
        'Head': 0, 'ModelNumber': 3000, 'PingTime': 1700000000.0, 'PingNumber': 5,
        'SerialNumber': 100, 'SystemStatus': 0, 'Mode': 1, 'FilterID': 2,
        'MinDepth_m': 5.0, 'MaxDepth_m': 500.0, 'Absorption_dBkm': 0.5,
        'PulseLength_us': 150.0, 'TransmitBeamWidth_deg': 1.5, 'PowerReduction_dB': 0,
        'ReceiveBeamWidth_deg': 2.0, 'ReceiveBandwidth_Hz': 150.0, 'ReceiveGain_dB': 10,
        'CrossOverAngle_deg': 30, 'SsvSource': 0, 'PortSwathWidth_m': 100, 'BeamSpacing': 1,
        'PortCoverageSector_deg': 60, 'Stabilization': 0, 'StbdCoverageSector_deg': 0,
        'StbdSwathWidth_m': 0, 'HiloFreqAbsorpRatio': 2,
    }

    def test_round_trip_no_run_time_blocks(self):
        payload = _encode_em3_specific(118, self._FIELDS)
        decoded, tables, consumed = _decode_em3_specific(payload, 4)

        assert decoded == self._FIELDS
        assert tables == {'RunTime': []}
        assert consumed == len(payload) - 4

    def test_round_trip_head0_only(self):
        payload = _encode_em3_specific(118, self._FIELDS, {'RunTime': [self._HEAD0]})
        _decoded, tables, consumed = _decode_em3_specific(payload, 4)

        assert len(tables['RunTime']) == 1
        assert tables['RunTime'][0]['Head'] == 0
        assert tables['RunTime'][0]['ModelNumber'] == 3000
        assert consumed == len(payload) - 4

    def test_round_trip_both_heads(self):
        # Deliberate improvement over the reference gsf_enc.c, which as
        # shipped can only ever write head 0 -- confirms this encoder can
        # write both heads of an EM3000D dual-head system.
        head1 = dict(self._HEAD0, Head=1, SerialNumber=101)
        payload = _encode_em3_specific(130, self._FIELDS, {'RunTime': [self._HEAD0, head1]})
        _decoded, tables, consumed = _decode_em3_specific(payload, 4)

        assert len(tables['RunTime']) == 2
        assert tables['RunTime'][0]['Head'] == 0
        assert tables['RunTime'][1]['Head'] == 1
        assert tables['RunTime'][1]['SerialNumber'] == 101
        assert consumed == len(payload) - 4

    def test_head1_without_head0_raises(self):
        with pytest.raises(ValueError):
            _encode_em3_specific(118, self._FIELDS, {'RunTime': [dict(self._HEAD0, Head=1)]})

    def test_duplicate_head_raises(self):
        with pytest.raises(ValueError):
            _encode_em3_specific(118, self._FIELDS, {'RunTime': [self._HEAD0, dict(self._HEAD0)]})

    def test_invalid_head_value_raises(self):
        with pytest.raises(ValueError):
            _encode_em3_specific(118, self._FIELDS, {'RunTime': [dict(self._HEAD0, Head=2)]})

    def test_registered_under_every_em3_id(self):
        for subrecord_id in (118, 119, 120, 128, 129, 130, 131, 132, 139):
            family_label, decode_fn, encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
            assert family_label == "EM3"
            assert decode_fn is _decode_em3_specific
            assert encode_fn is _encode_em3_specific

    def test_subrecord_header_word(self):
        payload = _encode_em3_specific(139, self._FIELDS)
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 139
        assert word & 0xFFFFFF == len(payload) - 4


class TestEncodeEm3RawSpecific:
    def test_round_trip_with_sectors(self):
        fields = {
            'ModelNumber': 300, 'PingCounter': 1, 'SerialNumber': 100,
            'SurfaceVelocity_mps': 1500.0, 'TransducerDepth_m': 32.5, 'ValidDetections': 256,
            'SamplingFrequency_Hz': 191.03125, 'VehicleDepth_m': -1.5, 'DepthDifference_m': 0.5,
            'OffsetMultiplier': -1,
            'RunTime.ModelNumber': 300, 'RunTime.MinDepth_m': 10.0,
            'PuStatus.SensorStatus': 63, 'PuStatus.YawStabilization_deg': -1.5,
        }
        sectors = [
            {'TiltAngle_deg': 1.5, 'FocusRange_m': 100.0, 'SignalLength_sec': 0.005,
             'TransmitDelay_sec': 0.0, 'CenterFrequency_Hz': 71000.0,
             'WaveformID': 1, 'SectorNumber': 0, 'SignalBandwidth_Hz': 14000.0},
            {'TiltAngle_deg': -1.5, 'FocusRange_m': 0.0, 'SignalLength_sec': 0.003,
             'TransmitDelay_sec': 0.001, 'CenterFrequency_Hz': 72000.0,
             'WaveformID': 0, 'SectorNumber': 1, 'SignalBandwidth_Hz': 15000.0},
        ]
        payload = _encode_em3raw_specific(140, fields, {'TxSectors': sectors})

        decoded, tables, consumed = _decode_em3raw_specific(payload, 4)

        assert decoded['ModelNumber'] == 300
        assert decoded['SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert decoded['TransducerDepth_m'] == pytest.approx(32.5)
        assert decoded['SamplingFrequency_Hz'] == pytest.approx(191.03125)
        assert decoded['VehicleDepth_m'] == pytest.approx(-1.5)
        assert decoded['DepthDifference_m'] == pytest.approx(0.5)
        assert decoded['OffsetMultiplier'] == -1
        assert decoded['RunTime.ModelNumber'] == 300
        assert decoded['RunTime.MinDepth_m'] == pytest.approx(10.0)
        assert decoded['PuStatus.SensorStatus'] == 63
        assert decoded['PuStatus.YawStabilization_deg'] == pytest.approx(-1.5)
        assert len(tables['TxSectors']) == 2
        assert tables['TxSectors'][0]['CenterFrequency_Hz'] == pytest.approx(71000.0)
        assert 'MeanAbsorption_dBkm' not in tables['TxSectors'][0]
        assert tables['TxSectors'][1]['TiltAngle_deg'] == pytest.approx(-1.5)
        assert consumed == len(payload) - 4

    def test_registered_under_every_em3raw_id(self):
        for subrecord_id in (140, 141, 142, 143, 144, 145, 146, 147, 148):
            family_label, decode_fn, encode_fn = _PING_SENSOR_SPECIFIC_CODECS[subrecord_id]
            assert family_label == "EM3Raw"
            assert decode_fn is _decode_em3raw_specific
            assert encode_fn is _encode_em3raw_specific

    def test_subrecord_header_word_and_zero_sectors(self):
        payload = _encode_em3raw_specific(148, {})
        word, = struct.unpack_from('>I', payload, 0)
        assert (word >> 24) & 0xFF == 148
        assert word & 0xFFFFFF == len(payload) - 4

        _decoded, tables, consumed = _decode_em3raw_specific(payload, 4)
        assert tables == {'TxSectors': []}
        assert consumed == len(payload) - 4


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

    def test_missing_required_scalar_raises_valueerror(self):
        incomplete = dict(self._SCALARS)
        del incomplete['NumberBeams']
        with pytest.raises(ValueError, match='NumberBeams'):
            _encode_swath_bathymetry_ping(incomplete, self._BEAMS)

    def test_required_scalar_left_as_none_raises_valueerror(self):
        # new_swath_bathymetry_ping_scalars() sets required fields to None
        # as a placeholder -- forgetting to overwrite one must not
        # silently encode None (a struct.pack TypeError deep in the
        # encoder) or a nonsense value.
        scalars = new_swath_bathymetry_ping_scalars()
        scalars.update(Longitude_deg=-70.5, Latitude_deg=43.1, NumberBeams=1)
        # PingTime deliberately left None.
        with pytest.raises(ValueError, match='PingTime'):
            _encode_swath_bathymetry_ping(scalars, {'Depth_m': [10.0]})

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

    def test_quality_flags_round_trip(self):
        beams = dict(self._BEAMS)
        beams['QualityFlags'] = [3, 0, 2]
        payload = _encode_swath_bathymetry_ping(self._SCALARS, beams)
        _scalars, tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})
        assert list(tables['Beams']['QualityFlags']) == [3, 0, 2]

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

    def test_sensor_specific_round_trip(self):
        # Generic _PING_SENSOR_SPECIFIC_CODECS dispatch (elac_mkii_specific
        # id 117 as the reference example -- any other registered id
        # exercises the same code path in _encode_/_decode_swath_bathymetry_ping()).
        fields = {'Mode': 5, 'PingNumber': 42}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(117, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['ElacMkII.Mode'] == 5
        assert scalars['ElacMkII.PingNumber'] == 42
        assert notes == []

    def test_em12_sensor_specific_round_trip(self):
        fields = {'PingNumber': 100, 'Resolution': 1, 'PingQuality': 50,
                  'SoundVelocity_mps': 1500.3, 'Mode': 2}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(103, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['EM12.PingNumber'] == 100
        assert scalars['EM12.SoundVelocity_mps'] == pytest.approx(1500.3)
        assert notes == []

    def test_em1000_sensor_specific_round_trip(self):
        # EM1000 (id 111) shares EM950's codec -- confirm the shared-codec
        # dispatch works through the full ping encoder/decoder, not just
        # in isolation (see TestEncodeEm950Specific.
        # test_same_codec_registered_for_em950_and_em1000).
        fields = {'PingNumber': 42, 'Mode': 3}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(111, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['EM1000.PingNumber'] == 42
        assert scalars['EM1000.Mode'] == 3
        assert notes == []

    def test_seabat_sensor_specific_round_trip(self):
        fields = {'PingNumber': 42, 'SurfaceVelocity_mps': 1500.5, 'Mode': 3,
                  'SonarRange_m': 100, 'TransmitPower': 5, 'ReceiveGain': 6}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(110, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['SeaBat.PingNumber'] == 42
        assert scalars['SeaBat.SurfaceVelocity_mps'] == pytest.approx(1500.5)
        assert notes == []

    def test_reson8100_sensor_specific_round_trip(self):
        # id 125 (RESON_8125) exercises the shared Reson8100 codec under a
        # non-first id in its group, confirming the header byte and
        # dispatch both follow subrecord_id correctly.
        fields = {'PingNumber': 3, 'SonarID': 99, 'SurfaceVelocity_mps': 1500.2,
                  'BeamSpacing_deg': 0.5, 'ProjectorAngle': -50}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(125, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['Reson8100.PingNumber'] == 3
        assert scalars['Reson8100.SonarID'] == 99
        assert scalars['Reson8100.SurfaceVelocity_mps'] == pytest.approx(1500.2)
        assert scalars['Reson8100.BeamSpacing_deg'] == pytest.approx(0.5)
        assert scalars['Reson8100.ProjectorAngle'] == -50
        assert notes == []

    def test_delta_t_sensor_specific_round_trip(self):
        fields = {'DecodeFileType': 'DT4', 'PingNumber': 77, 'SoundVelocity_mps': 1500.0,
                  'Altitude_m': 12.34}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(150, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['DeltaT.DecodeFileType'] == 'DT4'
        assert scalars['DeltaT.PingNumber'] == 77
        assert scalars['DeltaT.SoundVelocity_mps'] == pytest.approx(1500.0)
        assert scalars['DeltaT.Altitude_m'] == pytest.approx(12.34)
        assert notes == []

    def test_sass_sensor_specific_round_trip(self):
        # id 112 (TypeIII SeaBeam) exercises the shared SASS codec under
        # its non-primary id.
        fields = {'LeftmostBeam': 1, 'RightmostBeam': 60, 'PingNumber': 5}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(112, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['TypeIIISeaBeam.LeftmostBeam'] == 1
        assert scalars['TypeIIISeaBeam.RightmostBeam'] == 60
        assert scalars['TypeIIISeaBeam.PingNumber'] == 5
        assert notes == []

    def test_r2sonic_sensor_specific_round_trip(self):
        fields = {'ModelNumber': '2024', 'PingNumber': 42, 'SoundSpeed_mps': 1500.0,
                  'A0MoreInfo': [1.0, -2.0, 3.0, 0.0, 0.0, 0.0]}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(151, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['R2Sonic.ModelNumber'] == '2024'
        assert scalars['R2Sonic.PingNumber'] == 42
        assert scalars['R2Sonic.SoundSpeed_mps'] == pytest.approx(1500.0)
        assert scalars['R2Sonic.A0MoreInfo'] == pytest.approx([1.0, -2.0, 3.0, 0.0, 0.0, 0.0])
        assert notes == []

    def test_reson7125_sensor_specific_round_trip(self):
        fields = {'ProtocolVersion': 1, 'DeviceID': 7125, 'PingNumber': 42,
                  'Frequency_Hz': 400000.0, 'Gain_dB': -3.0}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(138, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['Reson7125.DeviceID'] == 7125
        assert scalars['Reson7125.PingNumber'] == 42
        assert scalars['Reson7125.Frequency_Hz'] == pytest.approx(400000.0)
        assert scalars['Reson7125.Gain_dB'] == pytest.approx(-3.0)
        assert notes == []

    def test_reson_tseries_sensor_specific_round_trip(self):
        fields = {'ProtocolVersion': 1, 'DeviceID': 7125, 'PingNumber': 12345,
                  'Frequency_Hz': 400000.0, 'Gain_dB': -3.0,
                  'DeviceDescription': 'T50-S SN 12345', 'SoundVelocity_mps': 1500.123456}
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(155, fields))

        scalars, _tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['ResonTSeries.DeviceID'] == 7125
        assert scalars['ResonTSeries.PingNumber'] == 12345
        assert scalars['ResonTSeries.Frequency_Hz'] == pytest.approx(400000.0)
        assert scalars['ResonTSeries.Gain_dB'] == pytest.approx(-3.0)
        assert scalars['ResonTSeries.DeviceDescription'] == 'T50-S SN 12345'
        assert scalars['ResonTSeries.SoundVelocity_mps'] == pytest.approx(1500.123456)
        assert notes == []

    def test_em4_sensor_specific_round_trip(self):
        fields = {
            'ModelNumber': 710, 'SurfaceVelocity_mps': 1500.0,
            'RunTime.MinDepth_m': 10.0, 'PuStatus.SensorStatus': 63,
        }
        sectors = [{'TiltAngle_deg': 1.5, 'CenterFrequency_Hz': 71000.0, 'SectorNumber': 0}]
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(133, fields, {'TxSectors': sectors}))

        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['EM4.ModelNumber'] == 710
        assert scalars['EM4.SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert scalars['EM4.RunTime.MinDepth_m'] == pytest.approx(10.0)
        assert scalars['EM4.PuStatus.SensorStatus'] == 63
        assert len(tables['EM4.TxSectors']) == 1
        assert tables['EM4.TxSectors'].iloc[0]['CenterFrequency_Hz'] == pytest.approx(71000.0)
        assert notes == []

    def test_em3raw_sensor_specific_round_trip(self):
        fields = {
            'ModelNumber': 300, 'SurfaceVelocity_mps': 1500.0, 'DepthDifference_m': 0.5,
            'RunTime.MinDepth_m': 10.0, 'PuStatus.SensorStatus': 63,
        }
        sectors = [{'TiltAngle_deg': 1.5, 'CenterFrequency_Hz': 71000.0, 'SectorNumber': 0}]
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(140, fields, {'TxSectors': sectors}))

        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['EM3Raw.ModelNumber'] == 300
        assert scalars['EM3Raw.SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert scalars['EM3Raw.DepthDifference_m'] == pytest.approx(0.5)
        assert scalars['EM3Raw.RunTime.MinDepth_m'] == pytest.approx(10.0)
        assert scalars['EM3Raw.PuStatus.SensorStatus'] == 63
        assert len(tables['EM3Raw.TxSectors']) == 1
        assert tables['EM3Raw.TxSectors'].iloc[0]['CenterFrequency_Hz'] == pytest.approx(71000.0)
        assert notes == []

    def test_em3_sensor_specific_round_trip(self):
        fields = {
            'ModelNumber': 3000, 'PingNumber': 5, 'SerialNumber': 100,
            'SurfaceVelocity_mps': 1500.0, 'TransducerDepth_m': 6.0, 'ValidBeams': 200,
            'SampleRate_Hz': 15000, 'DepthDifference_m': 0.5, 'OffsetMultiplier': -1,
        }
        head0 = {
            'Head': 0, 'ModelNumber': 3000, 'PingTime': 1700000000.0, 'PingNumber': 5,
            'SerialNumber': 100, 'SystemStatus': 0, 'Mode': 1, 'FilterID': 2,
            'MinDepth_m': 5.0, 'MaxDepth_m': 500.0, 'Absorption_dBkm': 0.5,
            'PulseLength_us': 150.0, 'TransmitBeamWidth_deg': 1.5, 'PowerReduction_dB': 0,
            'ReceiveBeamWidth_deg': 2.0, 'ReceiveBandwidth_Hz': 150.0, 'ReceiveGain_dB': 10,
            'CrossOverAngle_deg': 30, 'SsvSource': 0, 'PortSwathWidth_m': 100, 'BeamSpacing': 1,
            'PortCoverageSector_deg': 60, 'Stabilization': 0, 'StbdCoverageSector_deg': 0,
            'StbdSwathWidth_m': 0, 'HiloFreqAbsorpRatio': 2,
        }
        payload = _encode_swath_bathymetry_ping(
            self._SCALARS, self._BEAMS, sensor_specific=(118, fields, {'RunTime': [head0]}))

        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert scalars['EM3.ModelNumber'] == 3000
        assert scalars['EM3.SurfaceVelocity_mps'] == pytest.approx(1500.0)
        assert len(tables['EM3.RunTime']) == 1
        assert tables['EM3.RunTime'].iloc[0]['Head'] == 0
        assert tables['EM3.RunTime'].iloc[0]['MinDepth_m'] == pytest.approx(5.0)
        assert notes == []

    def test_sensor_specific_unregistered_id_raises_keyerror(self):
        with pytest.raises(KeyError):
            _encode_swath_bathymetry_ping(
                self._SCALARS, self._BEAMS, sensor_specific=(999, {}))

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
# new_swath_bathymetry_ping_scalars() / new_kmall_specific() /
# new_kmall_tx_sector() -- template-dict helpers
# ---------------------------------------------------------------------------

class TestTemplateHelpers:
    def test_ping_scalars_template_has_required_fields_as_none(self):
        scalars = new_swath_bathymetry_ping_scalars()
        for key in ('PingTime', 'Longitude_deg', 'Latitude_deg', 'NumberBeams'):
            assert scalars[key] is None

    def test_ping_scalars_template_optional_fields_are_null_sentinels(self):
        scalars = new_swath_bathymetry_ping_scalars()
        assert scalars['Speed_kn'] == GSF_NULL_SPEED
        assert scalars['Course_deg'] == GSF_NULL_COURSE
        assert scalars['TideCorrector_m'] == GSF_NULL_TIDE_CORRECTOR
        assert scalars['DepthCorrector_m'] == GSF_NULL_DEPTH_CORRECTOR
        assert scalars['Heading_deg'] == GSF_NULL_HEADING
        assert scalars['Pitch_deg'] == GSF_NULL_PITCH
        assert scalars['Roll_deg'] == GSF_NULL_ROLL
        assert scalars['Heave_m'] == GSF_NULL_HEAVE
        assert scalars['Height_m'] == GSF_NULL_HEIGHT
        assert scalars['SEP_m'] == GSF_NULL_SEP
        # No GSF_NULL_* sentinel exists for these.
        assert scalars['CenterBeam'] == 0
        assert scalars['GPSTideCorrector_m'] == 0.0

    def test_ping_scalars_template_populated_and_used_directly(self):
        # The whole point: fill in the required fields plus whatever you
        # know, leave the rest, and pass it straight to the encoder.
        scalars = new_swath_bathymetry_ping_scalars()
        scalars.update(
            PingTime=1700000000.0, Longitude_deg=-70.5, Latitude_deg=43.1,
            NumberBeams=1, Speed_kn=5.0)
        payload = _encode_swath_bathymetry_ping(scalars, {'Depth_m': [10.0]})
        decoded, _tables, _notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert decoded['Speed_kn'] == pytest.approx(5.0)
        assert decoded['Course_deg'] == pytest.approx(GSF_NULL_COURSE)  # left untouched

    def test_kmall_specific_template_round_trips_and_covers_every_encoded_key(self):
        s = new_kmall_specific()
        payload = _encode_kmall_specific(s)
        decoded, _sectors, _classes, _consumed = _decode_kmall_specific(payload, 4)

        # Every non-derived, non-forced key in the template must survive
        # a round trip unchanged (within float rounding).
        for key, value in s.items():
            assert decoded[key] == pytest.approx(value)

    def test_kmall_tx_sector_template_round_trips(self):
        row = new_kmall_tx_sector()
        row.update(TxSectorNumb=2, CentreFreq_Hz=71000.0)
        payload = _encode_kmall_specific({}, sector_rows=[row])
        _decoded, sectors, _classes, _consumed = _decode_kmall_specific(payload, 4)

        assert len(sectors) == 1
        assert sectors[0]['TxSectorNumb'] == 2
        assert sectors[0]['CentreFreq_Hz'] == pytest.approx(71000.0)
        assert sectors[0]['TxArrNumber'] == 0


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

    def test_previously_unimplemented_record_types_round_trip(self, tmp_path, capsys):
        # write_swath_bathy_summary/write_comment/write_history/
        # write_navigation_error/write_hv_navigation_error/
        # write_single_beam_ping -- all untested against a verified GSF
        # file (see each method's docstring), so this only checks
        # self-consistency via our own reader, the same way the rest of
        # this file's round-trip tests do.
        path = tmp_path / "out.gsf"
        G = gsf(str(path))
        G.write_header()
        G.write_swath_bathy_summary(
            start_time=1700000000.0, end_time=1700003600.0,
            min_latitude_deg=42.9, min_longitude_deg=-70.6,
            max_latitude_deg=43.3, max_longitude_deg=-70.4,
            min_depth_m=5.0, max_depth_m=125.5)
        G.write_comment(1700000000.0, "test comment")
        G.write_history(
            1700000000.0, host_name="host1", operator_name="vschmidt",
            command_line="gsfu.py -f x.gsf -V", comment="test history")
        G.write_navigation_error(1700000000.0, record_id=1, longitude_error_m=1.3, latitude_error_m=-0.8)
        G.write_hv_navigation_error(
            1700000000.0, record_id=2, horizontal_error_m=0.35,
            vertical_error_m=0.12, sep_uncertainty_m=0.5, position_type="GPS")
        G.write_single_beam_ping(
            ping_time=1700000000.5, longitude_deg=-70.5, latitude_deg=43.1,
            tide_corrector_m=0.1, depth_corrector_m=-1.2, heading_deg=123.45,
            pitch_deg=-1.1, roll_deg=0.4, heave_m=0.2, depth_m=25.75,
            sound_speed_correction_m=0.05, positioning_system_type=3)
        G.closeFile()

        G2 = gsf(str(path))
        G2.index_file()
        assert list(G2.Index['RecordType']) == [
            'GSF_RECORD_HEADER', 'GSF_RECORD_SWATH_BATHY_SUMMARY', 'GSF_RECORD_COMMENT',
            'GSF_RECORD_HISTORY', 'GSF_RECORD_NAVIGATION_ERROR', 'GSF_RECORD_HV_NAVIGATION_ERROR',
            'GSF_RECORD_SINGLE_BEAM_PING']
        assert int(G2.Index['TotalBytes'].sum()) == path.stat().st_size

        G3 = gsf(str(path))
        G3.print_records()
        captured = capsys.readouterr()
        assert "decode failed" not in captured.out
        assert "test comment" in captured.out
        assert "test history" in captured.out

