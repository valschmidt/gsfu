#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test cases for GSFU.kmall2gsf.

Two kinds of coverage:

  * Synthetic/unit tests for the pure-Python helpers (attitude
    interpolation, circular heading lerp, lenient key=value parsing) that
    don't need any sample data or the `kmall` package.

  * Real-file tests that run an end-to-end .kmall -> .gsf conversion
    against real sample files and check the result with gsfu's own
    reader. These are skipped if either the sample .kmall files or the
    `KMALL` package (https://github.com/valschmidt/kmall) are not
    available, since neither is a hard dependency of this repo.
"""
from pathlib import Path

import pytest

from GSFU.kmall2gsf import (
    _circular_lerp,
    _lenient_parse_kv_text,
    interpolate_attitude,
)

KMALL_DATA_DIR = Path("/Users/vschmidt/gitsrc/kmall.new/data")
KMALL_SAMPLE_FILES = sorted(KMALL_DATA_DIR.glob("*.kmall")) if KMALL_DATA_DIR.is_dir() else []

try:
    from KMALL.kmall import kmall as _KmallReader  # noqa: F401
    _HAVE_KMALL_PACKAGE = True
except ImportError:
    _HAVE_KMALL_PACKAGE = False

requires_kmall_samples = pytest.mark.skipif(
    not (KMALL_SAMPLE_FILES and _HAVE_KMALL_PACKAGE),
    reason="no sample .kmall files found, or the KMALL package is not installed")


# ---------------------------------------------------------------------------
# _lenient_parse_kv_text
# ---------------------------------------------------------------------------

class TestLenientParseKvText:
    def test_comma_separated(self):
        assert _lenient_parse_kv_text("A=1,B=2,C=3") == {"A": "1", "B": "2", "C": "3"}

    def test_semicolon_separated(self):
        assert _lenient_parse_kv_text("A=1;B=2;C=3") == {"A": "1", "B": "2", "C": "3"}

    def test_newline_separated(self):
        assert _lenient_parse_kv_text("A=1\nB=2\nC=3") == {"A": "1", "B": "2", "C": "3"}

    def test_skips_malformed_entries(self):
        # A blank/no-'=' entry (like the real malformed data that crashes
        # kmall.py's own translate_installation_parameters_todict()) is
        # silently skipped rather than raising.
        assert _lenient_parse_kv_text("A=1,garbage,B=2,,C=3") == {"A": "1", "B": "2", "C": "3"}

    def test_strips_whitespace(self):
        assert _lenient_parse_kv_text(" A = 1 , B=2") == {"A": "1", "B": "2"}

    def test_value_may_contain_equals(self):
        assert _lenient_parse_kv_text("A=1=2") == {"A": "1=2"}

    def test_empty_text(self):
        assert _lenient_parse_kv_text("") == {}

    def test_all_malformed(self):
        assert _lenient_parse_kv_text("garbage,more garbage") == {}


# ---------------------------------------------------------------------------
# _circular_lerp
# ---------------------------------------------------------------------------

class TestCircularLerp:
    def test_no_wraparound(self):
        assert _circular_lerp(10.0, 20.0, 0.5) == pytest.approx(15.0)

    def test_wraparound_forward(self):
        # 350 -> 10 the short way (through 0/360) is +20 total.
        assert _circular_lerp(350.0, 10.0, 0.5) == pytest.approx(0.0)

    def test_wraparound_backward(self):
        assert _circular_lerp(10.0, 350.0, 0.5) == pytest.approx(0.0)

    def test_frac_zero_returns_start(self):
        assert _circular_lerp(30.0, 60.0, 0.0) == pytest.approx(30.0)

    def test_frac_one_returns_end(self):
        assert _circular_lerp(30.0, 60.0, 1.0) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# interpolate_attitude
# ---------------------------------------------------------------------------

class TestInterpolateAttitude:
    SAMPLES = [
        (100.0, 1.0, -1.0, 0.1, 350.0),
        (200.0, 3.0, 1.0, 0.3, 10.0),
    ]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            interpolate_attitude([], 100.0)

    def test_midpoint(self):
        pitch, roll, heave, heading = interpolate_attitude(self.SAMPLES, 150.0)
        assert pitch == pytest.approx(2.0)
        assert roll == pytest.approx(0.0)
        assert heave == pytest.approx(0.2)
        assert heading == pytest.approx(0.0)  # short way through 0/360

    def test_clamps_before_first_sample(self):
        result = interpolate_attitude(self.SAMPLES, 0.0)
        assert result == (1.0, -1.0, 0.1, 350.0)

    def test_clamps_after_last_sample(self):
        result = interpolate_attitude(self.SAMPLES, 999.0)
        assert result == (3.0, 1.0, 0.3, 10.0)

    def test_exact_sample_time(self):
        pitch, roll, heave, heading = interpolate_attitude(self.SAMPLES, 100.0)
        assert (pitch, roll, heave, heading) == (1.0, -1.0, 0.1, 350.0)


# ---------------------------------------------------------------------------
# End-to-end conversion against real sample data
# ---------------------------------------------------------------------------

@requires_kmall_samples
class TestConvertRealFiles:
    @pytest.mark.parametrize("path", KMALL_SAMPLE_FILES, ids=lambda p: p.name)
    def test_convert_runs_and_produces_valid_gsf(self, path, tmp_path):
        from GSFU.gsfu import gsf
        from GSFU.kmall2gsf import convert

        out_path = tmp_path / (path.stem + ".gsf")
        ping_count = convert(str(path), str(out_path), attitude_source=1)
        assert ping_count > 0
        assert out_path.exists()

        g = gsf(str(out_path))
        g.index_file()

        # First record must be the file header, and every ping's beam
        # array offsets must exactly tile the file with no gaps/overlaps
        # (index_file() itself checks this internally and would raise).
        assert g.Index.iloc[0]["RecordType"] == "GSF_RECORD_HEADER"
        counts = g.Index["RecordType"].value_counts()
        assert counts.get("GSF_RECORD_SWATH_BATHYMETRY_PING", 0) == ping_count

    def test_convert_first_ping_has_sane_kmall_specific(self, tmp_path):
        from GSFU.gsfu import gsf, _decode_swath_bathymetry_ping
        from GSFU.kmall2gsf import convert

        path = KMALL_SAMPLE_FILES[0]
        out_path = tmp_path / (path.stem + ".gsf")
        convert(str(path), str(out_path), attitude_source=1)

        g = gsf(str(out_path))
        g.index_file()
        row = g.Index[g.Index["RecordType"] == "GSF_RECORD_SWATH_BATHYMETRY_PING"].iloc[0]
        g.FID.seek(int(row["ByteOffset"]))
        dataSize, _readSize, data_id = g.read_record_header()
        if data_id.checksumFlag:
            g.FID.seek(4, 1)
        payload = g.FID.read(dataSize)
        scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

        assert notes == []
        assert -90.0 <= scalars["Latitude_deg"] <= 90.0
        assert -180.0 <= scalars["Longitude_deg"] <= 180.0
        assert scalars["NumberBeams"] == len(tables["Beams"])
        assert (tables["Beams"]["Depth_m"] > 0).all()
