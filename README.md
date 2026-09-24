# GSF file reader/writer (`gsfu`)

A Python class and command line utility for indexing, reading, and writing
sonar data files in the **Generic Sensor Format (GSF)**. `gsfu.py` can index
a GSF file and print a summary of the record types it contains (`-V`), and
can decode and print any record's fields for debugging (`-p`, `-I`). The
`gsf` class also has a full set of `write_*` methods for encoding new GSF
files from scratch -- see "Capabilities" below for the exact current state.
The repo also includes [`kmall2gsf.py`](#kmall--gsf-conversion-kmall2gsfpy),
a Kongsberg `.kmall` → `.gsf` converter built on the write API.

    ./GSFU/gsfu.py -h
    usage: gsfu.py [-h] [-f GSF_FILENAME] [-V] [-p [RECORDTYPE]] [-I] [-v]

    A python script (and class) for indexing and reading Generic Sensor Format
    (GSF) data files.

    options:
      -h, --help       show this help message and exit
      -f GSF_FILENAME  The path and filename to parse.
      -V               Index the file and print a summary of its record types
                       (count and bytes consumed by each).
      -p [RECORDTYPE]  Print records to stdout as ASCII text, for debugging. With
                       no value, prints every record; optionally restrict to one
                       record type, e.g. -p COMMENT or -p GSF_RECORD_COMMENT.
      -I               Print each swath bathymetry ping's per-beam backscatter
                       time series (the intensity series subrecord) to stdout as
                       CSV, one row per beam. Decoded for KMALL, EM3-series,
                       EM4-series, Reson 7125/T-series/8100-family, Klein 5410
                       BSS, R2Sonic, and any other sensor with no imagery-specific
                       preamble.
      -v               Increasingly verbose output (e.g. -v -vv), for debugging
                       use -vv

    record types (short or full name accepted for -p, e.g. -p COMMENT):

      HEADER                  GSF header record. Identifies the GSF version used to create the file.
      SWATH_BATHYMETRY_PING   Data structure for a ping from a swath bathymetric system.
      SOUND_VELOCITY_PROFILE  Sound velocity profile record.
      PROCESSING_PARAMETERS   Internal record structure for processing parameters.
      SENSOR_PARAMETERS       Sensor parameters record.
      COMMENT                 Comment record.
      HISTORY                 History record.
      NAVIGATION_ERROR        Navigation error record. (Obsolete; replaced by GSF_RECORD_HV_NAVIGATION_ERROR.)
      SWATH_BATHY_SUMMARY     Swath bathymetry summary record.
      SINGLE_BEAM_PING        Single beam ping record.
      HV_NAVIGATION_ERROR     Horizontal/Vertical navigation error record. Replaces GSF_RECORD_NAVIGATION_ERROR. (The HV stands for Horizontal and Vertical.)
      ATTITUDE                Attitude record: one or more time-tagged pitch/roll/heave/heading measurements.

## Why a pure-Python implementation

The [UK Hydrographic Office's `gsfpy`](https://github.com/UKHO/gsfpy) package
wraps the reference GSF C library (`libgsf`) via `ctypes`. It's a fine choice
when a prebuilt `libgsf` shared library is available for your platform, but
the shared libraries bundled in the `gsfpy` wheel are Linux-only -- they
cannot be loaded on macOS or Windows, and even importing `gsfpy` on those
platforms raises at import time. That made it unworkable as this project's
runtime dependency.

Instead, `gsfu.py` re-implements the GSF file format directly in Python,
using `struct` to unpack the on-disk record framing -- the same approach
[`kmall.py`](https://github.com/valschmidt/kmall) takes for Kongsberg's
`.kmall` format. This keeps the tool dependency-free (just `pandas` and
`numpy`) and portable to any platform Python runs on. Ping beam arrays
(depth, travel time, amplitude, etc., stored as scaled 1/2/4-byte integers
per `gsfScaleFactors`) are decoded with `numpy` (`frombuffer` + a single
vectorized scale/offset op per array), not a per-beam Python loop --
benchmarked at roughly 100x faster than the naive per-beam-`struct.unpack`
approach on real multi-hundred-beam pings.

The record framing, naming, and constants used here are ported from the
reference C implementation of the GSF library ("gsflib"),
Copyright 2019 Leidos, Inc., distributed under the LGPL 2.1:
<https://github.com/Spatialnetics/gsflib> (`source/gsf/gsf.h`,
`source/gsf/gsf.c`). No gsflib source is reused -- `gsfu.py` is an
independent re-implementation, with constant names (`GSF_RECORD_HEADER`,
`RecordType`, `gsfDataID`, etc.) and doc comments chosen to match gsflib so
the code is recognizable to anyone already familiar with the reference
library.

## Installation

    pip install -r requirements.txt
    pip install -e .

## Usage

```python
from GSFU.gsfu import gsf

G = gsf("0264_20240826_022211_EM712.gsf")
G.index_file()             # builds G.Index, a pandas DataFrame, one row per record
G.report_record_types()    # prints (and returns) a per-record-type summary
```

```
$ gsfu.py -f data/GSF/0268_20240826_052757_EM712.gsf -V
File: data/GSF/0268_20240826_052757_EM712.gsf
GSF Version: GSF-v03.09
File size: 714924 bytes
Total records: 10670

                                   Count  Total Bytes  Min Bytes  Max Bytes  % of File
RecordType
GSF_RECORD_SWATH_BATHYMETRY_PING      15       410440      25652      28468      57.41
GSF_RECORD_ATTITUDE                10652       298256         28         28      41.72
GSF_RECORD_SOUND_VELOCITY_PROFILE      1         3972       3972       3972       0.56
GSF_RECORD_PROCESSING_PARAMETERS       1         2236       2236       2236       0.31
GSF_RECORD_HEADER                      1           20         20         20       0.00
```

`-p` decodes and prints records to stdout for debugging: scalar fields as
`key : value` pairs, and any per-beam/per-point/per-measurement data (a
ping's beam arrays, an SVP's depth/sound-speed pairs, an attitude record's
measurements) as a table, one row per beam/point/measurement:

```
$ gsfu.py -f data/GSF/0268_20240826_052757_EM712.gsf -p SWATH_BATHYMETRY_PING
=== GSF_RECORD_SWATH_BATHYMETRY_PING  offset=304484  size=25664 ===
  PingTime                          : 2024-08-26T05:27:53.719285+00:00
  Longitude_deg                     : -169.059375
  Latitude_deg                      : -14.2062916
  NumberBeams                       : 400
  ...
  KMALL.GSFKMALLVersion             : 0
  KMALL.DgmType                     : 1
  KMALL.EchoSounderID               : 712
  ...
  # IntensityTimeSeries (21, 13291 bytes) not decoded here: use gsf.print_intensity_series() / -I
-- TxSectors --
        TxSectorNumb  TxArrNumber  TxSubArray  SectorTransmitDelay_sec  ...
Sector
0                  0            0           0                 0.031502  ...
1                  1            0           0                 0.000000  ...
-- Beams --
       Depth_m  AcrossTrack_m  AlongTrack_m  TravelTime_s  BeamAngle_deg  ...
Beam
0     1014.072       -1150.11         62.35       2.04725         -75.85  ...
1     1013.272       -1142.42         61.84       2.03877         -75.65  ...
```

Every `GSF_RECORD_*` type has a decoder. Within a ping's subrecord stream,
every one of gsf.h's 55 `GSF_SWATH_BATHY_SUBRECORD_*_SPECIFIC` vendor
sensor-specific subrecord ids (`EM710_SPECIFIC`, `RESON_8101_SPECIFIC`,
KMALL, EM3-series, Reson 7100/T-series/8100-family, SeaBat, SeaBeam, Klein,
GeoSwath, DeltaT, R2Sonic, and every other historical format, including the
obsolete SASS/TypeIII-SeaBeam pair) is fully field-decoded, shown as
`<Family>.*` scalars (e.g. `KMALL.*`, `EM4.*`) plus any per-element table a
family produces (e.g. `TxSectors`, `EM4.TxSectors`, `EM3.RunTime`). The
per-beam quality-flags array (2-bit packed) is likewise fully decoded.
gsflib's own optional RLE array compression is the one remaining gap, still
noted rather than decoded. Any record a decoder can't make sense of (corrupt
data, an unexpected size) falls back to the same raw ASCII rendering
(non-printable bytes as `.`) used before decoders existed, with a note
explaining why.

Omit a value to dump every record (`-p`), or pass a short or full record
type name to restrict output to one type (`-p COMMENT` or
`-p GSF_RECORD_COMMENT`) -- this works the same way whether called from the
CLI or via `gsf.print_records(record_type=...)` directly.

### Per-beam backscatter time series (`-I`)

The intensity series subrecord (a ping's raw per-beam backscatter time
series -- potentially tens of thousands of samples per ping) is deliberately
*not* included in `-p`'s per-ping table, since it would dwarf the rest of the
output. `-I` prints it on its own, one CSV row per beam, across every ping
in the file:

```
$ gsfu.py -f data/GSF/0268_20240826_052757_EM712.gsf -I
# ping offset=304484 ping_time=2024-08-26T05:27:53.719285+00:00
# Beam,SampleCount,DetectSample,StartRangeSamples,Sample0,Sample1,...
0,140,8,38926,32666,32639,32615,32596,...
1,17,9,34830,32690,32681,32672,32674,...
```

Each row is `Beam,SampleCount,DetectSample,StartRangeSamples,` followed by
that beam's raw samples -- rows are naturally ragged since sample count
varies per beam. Decoded for every sensor family gsflib defines an
imagery-specific preamble for (KMALL, EM3-series, EM4-series, Reson
7125/T-series/8100-family, Klein 5410 BSS, R2Sonic) as well as every sensor
that has no such preamble at all; only a ping with no intensity series
subrecord, or an unsupported bits-per-sample encoding, is noted and skipped.

## Capabilities

**Indexing** (`-V`, `gsf.index_file()`): every one of the 12 top-level GSF
record types is recognized and indexed -- offset, size, and type only, the
payload is never decoded. This is what makes indexing fast even on
multi-hundred-thousand-record files.

**Reading / field-level decoding** (`-p`, `-I`, `gsf.print_records()`,
`gsf.print_intensity_series()`): all 12 record types have a field-level
decoder, so every one of them can be fully decoded, not just indexed:

| Record type | Description | Decoded by `-p`? |
|---|---|---|
| `GSF_RECORD_HEADER` | GSF header record. Identifies the GSF version used to create the file. | Yes |
| `GSF_RECORD_SWATH_BATHYMETRY_PING` | Data structure for a ping from a swath bathymetric system. | Yes, with caveats* |
| `GSF_RECORD_SOUND_VELOCITY_PROFILE` | Sound velocity profile record. | Yes |
| `GSF_RECORD_PROCESSING_PARAMETERS` | Internal record structure for processing parameters. | Yes |
| `GSF_RECORD_SENSOR_PARAMETERS` | Sensor parameters record. | Yes |
| `GSF_RECORD_COMMENT` | Comment record. | Yes |
| `GSF_RECORD_HISTORY` | History record. | Yes |
| `GSF_RECORD_NAVIGATION_ERROR` | Navigation error record. (Obsolete; replaced by `GSF_RECORD_HV_NAVIGATION_ERROR`.) | Yes |
| `GSF_RECORD_SWATH_BATHY_SUMMARY` | Swath bathymetry summary record. | Yes |
| `GSF_RECORD_SINGLE_BEAM_PING` | Single beam ping record. | Yes, including its sensor-specific tail (Echotrac/Bathy2000, MGD77, BDB, NOSHDB) |
| `GSF_RECORD_HV_NAVIGATION_ERROR` | Horizontal/Vertical navigation error record. | Yes |
| `GSF_RECORD_ATTITUDE` | Attitude record: one or more time-tagged pitch/roll/heave/heading measurements. | Yes |

\* Within a `SWATH_BATHYMETRY_PING`'s subrecord stream: the standard
scale-factor-encoded beam arrays (depth, across/along track, travel time,
beam angle, amplitude, errors, etc.), the beam-flags array, and the 2-bit
packed quality-flags array are decoded and vectorized with numpy; every one
of gsf.h's 55 vendor sensor-specific subrecord ids -- KMALL (its own
bespoke path, shown as `KMALL.*` plus a `TxSectors` table), and every other
id (EM3-series, EM4-series, Reson 7100/T-series/8100-family, SeaBat,
SeaBeam, Klein, GeoSwath, DeltaT, R2Sonic, etc., shown as `<Family>.*`
scalars plus any per-element table, e.g. `EM4.TxSectors`, `EM3.RunTime`) --
is fully field-decoded. gsflib's own optional RLE array compression is the
one remaining gap, noted rather than decoded. The per-beam backscatter time
series (decoded separately via `-I`/`print_intensity_series()`) is decoded
for KMALL, EM3-series, EM4-series, Reson 7125/T-series/8100-family, Klein
5410 BSS, R2Sonic, and every other sensor (its imagery-specific preamble,
if any, is a smaller, distinct block from -- and decoded independently of
-- the ping-level sensor-specific subrecord noted above). Anything a
decoder can't make sense of falls back to raw ASCII rendering with a note
explaining why.

**Writing / encoding** (`gsf.write_*`): the write side is the inverse of the
decoders above, ported from `gsf_enc.c` (not merely assumed to be decode's
mirror -- the scale-factor rounding convention, NUL-terminated parameter
strings, and other encode-only quirks were each checked against the real
source):

| Method | Writes |
|---|---|
| `gsf.write_header(version=...)` | `GSF_RECORD_HEADER` |
| `gsf.write_processing_parameters(params, param_time=...)` | `GSF_RECORD_PROCESSING_PARAMETERS` |
| `gsf.write_sensor_parameters(params, param_time=...)` | `GSF_RECORD_SENSOR_PARAMETERS`* |
| `gsf.write_sound_velocity_profile(...)` | `GSF_RECORD_SOUND_VELOCITY_PROFILE` |
| `gsf.write_attitude(...)` | `GSF_RECORD_ATTITUDE` |
| `gsf.write_swath_bathymetry_ping(scalars, beams, kmall_specific=..., tx_sectors=..., sensor_specific=..., scale_factors=..., auto_scale=...)` | `GSF_RECORD_SWATH_BATHYMETRY_PING`, including scale factors (static by default, or computed automatically per ping with `auto_scale=True` -- see "Scale factors" below), the standard beam arrays, the beam-flags and quality-flags arrays, the `KMALL_SPECIFIC` sensor-specific subrecord + its TX sector array (`kmall_specific`/`tx_sectors`), and every other vendor sensor-specific subrecord (`sensor_specific=(subrecord_id, fields[, tables])`, dispatched through `_PING_SENSOR_SPECIFIC_CODECS` -- the same 55-id coverage as decode) |
| `gsf.write_swath_bathy_summary(...)` | `GSF_RECORD_SWATH_BATHY_SUMMARY`* |
| `gsf.write_comment(comment_time, comment)` | `GSF_RECORD_COMMENT`* |
| `gsf.write_history(history_time, host_name, operator_name, command_line, comment)` | `GSF_RECORD_HISTORY`* |
| `gsf.write_navigation_error(...)` | `GSF_RECORD_NAVIGATION_ERROR`* (obsolete; prefer `write_hv_navigation_error()`) |
| `gsf.write_hv_navigation_error(...)` | `GSF_RECORD_HV_NAVIGATION_ERROR`* |
| `gsf.write_single_beam_ping(..., sensor_specific=...)` | `GSF_RECORD_SINGLE_BEAM_PING`*, including its sensor-specific tail (`sensor_specific=(subrecord_id, fields[, tables])`, dispatched through `_SINGLE_BEAM_SENSOR_SPECIFIC_CODECS`) |

\* Untested against a verified GSF file: none of the sample data checked
into this repo carries this record type (or, for the non-KMALL
`sensor_specific` subrecords above, this particular vendor's sensor-specific
subrecord), so these encoders are only verified by round-tripping through
this library's own decoders -- see each method's docstring.

The per-beam intensity time series (subrecord id 21) has no encoder yet
(only decode). The `scalars`/`beams` dict shapes accepted by `write_*` match
what the corresponding `_decode_*` function returns, so a record decoded
with `-p` can be re-encoded with only the field names already familiar from
that output. Rather than writing `scalars`/`kmall_specific`/a `tx_sectors`
row out by hand, `new_swath_bathymetry_ping_scalars()`,
`new_kmall_specific()`, and `new_kmall_tx_sector()` return a dict with
every valid key already present -- required fields as `None`, optional
fields pre-set to their `GSF_NULL_*` "not available" sentinel -- ready to
populate and pass straight to `write_swath_bathymetry_ping()`. See
[`convert.md`](convert.md) for a worked example.

### Scale factors

Every per-beam array inside a `SWATH_BATHYMETRY_PING` (depth, across/along-track,
travel time, beam angle, amplitudes, and about twenty others) is stored on disk
as a scaled integer rather than a float, to keep files small: a
`GSF_SWATH_BATHY_SUBRECORD_SCALE_FACTORS` subrecord at the start of the ping's
subrecord stream declares, per array, a `multiplier` and an `offset`, and each
value is written as

    raw_int = round((value + offset) * multiplier)

and read back as `value = raw_int / multiplier - offset`. A finer multiplier
means more decimal precision; a nonzero offset lets an otherwise-unsigned field
(depth, for instance, which is unsigned so a given field width covers twice the
positive range) hold values that dip slightly negative, by shifting them into
the field's representable range before scaling.

**Static scale factors (the default).** `write_swath_bathymetry_ping()` picks
each array's multiplier/offset/field-width from `DEFAULT_PING_SCALE_FACTORS`
unless told otherwise -- e.g. depth defaults to a 4-byte unsigned field at
1000.0 (1mm precision), offset 0.0. This is simple and predictable, but a fixed
offset of 0.0 on an unsigned field means any negative value (a beam reading a
few centimeters above the transducer near the surface, for example) raises
`ValueError` rather than being written at all, since there's no headroom to
represent it.

**Automatic scale factors (`auto_scale=True`).** Passing `auto_scale=True` to
`gsf(path, auto_scale=True)` (for every ping written through that instance) or
to an individual `write_swath_bathymetry_ping(..., auto_scale=True)` call makes
each beam array's multiplier and offset get computed from that ping's actual
data instead: `DEFAULT_PING_SCALE_FACTORS`'s multiplier becomes a *ceiling* --
the finest precision to use if the data fits -- rather than the value always
used, and the offset is derived directly from the ping's minimum and maximum
values instead of always being 0.0. This is implemented by
`_pick_ping_scale_factor()` in `gsfu.py`, and works the same way for every
array, not just depth -- so a field like `BeamAngleForward_deg`, which also
happens to be unsigned and already needs a hand-picked nonzero offset in
`DEFAULT_PING_SCALE_FACTORS` (`90.0`, to accommodate angles that swing negative
after roll correction), would have that same offset derived automatically
instead of needing someone to notice and hard-code it.

Because this runs on every ping written, it's built to be fast and to avoid
unnecessary work: it's vectorized (a single numpy min/max reduction per beam
array, no per-beam Python loop), and it keeps a per-subrecord "currently
active" scale factor on the `gsf` instance across calls, only recomputing one
when the newest ping's data no longer fits it -- most pings need no
recomputation at all. When a recompute *is* needed, the newly solved range is
padded outward first (by 25% of the ping's observed value span, or by 20 steps
at the target precision, whichever is larger) so that a slightly different
next ping doesn't immediately force another recompute -- this is the same
kind of hysteresis gsflib's own (depth-only, offset-only) auto-scale function
uses, generalized here to solve for both multiplier and offset together, for
any array, directly from the data rather than from an indirect proxy like a
tide corrector.

**Worked example.** The first ping written for the `Depth_m` array has values
ranging from -0.03m to 45.2m (a few beams read slightly negative near the
surface). With `auto_scale=True`:

1. The observed span is 45.23m. The hysteresis padding is
   `max(25% × 45.23, 20 steps ÷ 1000.0) ≈ 11.31m`, giving a padded target
   range of about -11.34m to 56.51m.
   
2. That padded range comfortably fits a 4-byte unsigned field even at the full
   1mm target precision, so the multiplier stays at 1000.0 -- no precision is
   sacrificed.

3. An offset of 0.0 isn't enough to keep the padded minimum non-negative once
   scaled, so the smallest whole-number offset that works is chosen: `12.0`.

4. The scale factor `(1000.0, 12.0)` is written for this ping, and every
   subsequent ping whose depths keep falling within roughly -12m to
   4,294,955m (the 4-byte unsigned field's full range at this multiplier and
   offset) reuses it unchanged -- no new `SCALE_FACTORS` values, no
   reprocessing -- until a ping's data eventually falls outside that range,
   at which point the whole process repeats.

## `.kmall` → `.gsf` conversion (`kmall2gsf.py`)

`GSFU/kmall2gsf.py` converts a Kongsberg `.kmall` file to `.gsf`, reading the
source file with the sibling [`kmall`](https://github.com/valschmidt/kmall)
package (imported lazily -- not a hard dependency of `gsfu.py` itself) and
writing `PROCESSING_PARAMETERS`, `SOUND_VELOCITY_PROFILE`,
`SWATH_BATHYMETRY_PING` (with the `KMALL_SPECIFIC` sensor-specific subrecord
and its TX sector array), and `ATTITUDE` records via the `write_*` API above.

```
kmall2gsf.py -f 0007_20190513_154724_ASVBEN.kmall -o 0007.gsf
```

KMALL files can carry attitude (`#SKM`) from more than one configured sensor
system (K-Controller's "Attitude 1/2/3..."); `-a`/`--attitude-source` selects
which one (1 by default) supplies the ping's interpolated pitch/roll/heave
and the `ATTITUDE` records written to the file. Ping position and heading
always come directly from the `#MRZ` datagram's own `pingInfo`, matching how
a native GSF writer would behave.

It builds its own lightweight, seek-based index of the `.kmall` file (one
pass reading only each datagram's 8-byte framing header) rather than reading
every datagram's full payload, and does not modify `kmall.py` itself --
including working around a couple of pre-existing bugs in that package
(a file-position reset in `decode_datagram()`, and a parser crash on
malformed installation-parameter text) entirely from within `kmall2gsf.py`.

Known simplifications (see the module docstring for the full list and
reasoning): `TideCorrector_m`/`DepthCorrector_m` and `Course_deg`/`Speed_kn`
are always written as `0.0` (MRZ carries none of these), `CenterBeam` is
approximated as `NumberBeams // 2` (MRZ has no explicit center-beam field),
and the per-beam backscatter time series is not written (no encoder yet --
`BEAM_FLAGS_ARRAY`/`QUALITY_FLAGS_ARRAY` do have encoders, but `kmall2gsf.py`
doesn't populate either, since `#MRZ` carries no per-beam flag equivalent).

## Testing

    pip install pytest
    pytest tests/

The test suite has three layers: synthetic, hand-crafted GSF byte streams
that independently verify the record framing math (bit-packing, checksum
handling, error conditions) against gsflib's documented encoding
(`test_gsfu.py`); round-trip encode/decode tests for every `write_*` method
(`test_gsfu_write.py`); and tests against the real sample `.gsf` files in
`data/GSF/` that check invariants that must hold for any valid GSF file
(every byte accounted for, every record type recognized, file starts with a
header record). `test_kmall2gsf.py` covers `kmall2gsf.py`'s pure-Python
helpers unconditionally, plus end-to-end `.kmall` → `.gsf` conversion against
real sample files when both sample data and the `KMALL` package are
available (skipped otherwise).
