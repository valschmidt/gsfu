# GSF file reader (`gsfu`)

A Python class and command line utility for indexing and reading sonar data
files in the **Generic Sensor Format (GSF)**. `gsfu.py` can index a GSF file
and print a summary of the record types it contains (`-V`), and can decode
and print any record's fields for debugging (`-p`, `-I`). **Writing/encoding
GSF files is not yet implemented** -- see "Capabilities" below for the exact
current state, and "Planned" for what's next (a `.kmall` → `.gsf` converter).

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
                       CSV, one row per beam. Currently decoded only for the KMALL
                       (Kongsberg SIS 5) sensor-imagery format.
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
the KMALL vendor-specific subrecord (id 156, shown as `KMALL.*` scalars and
the `TxSectors` table above) is fully field-decoded; every *other* vendor
sensor-specific subrecord (`EM710_SPECIFIC`, `RESON_8101_SPECIFIC`, ... ~29
of gsf.h's `GSF_SWATH_BATHY_SUBRECORD_*_SPECIFIC` ids) is at least
identified by its proper name, but not decoded. gsflib's own optional RLE
array compression and the 2-bit packed quality-flags array are likewise
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
varies per beam. As with the ping table, only the KMALL sensor-imagery
format is currently decoded; other sensors are noted and skipped.

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
| `GSF_RECORD_SINGLE_BEAM_PING` | Single beam ping record. | Yes, fixed fields only -- the sensor-specific tail is noted, not decoded |
| `GSF_RECORD_HV_NAVIGATION_ERROR` | Horizontal/Vertical navigation error record. | Yes |
| `GSF_RECORD_ATTITUDE` | Attitude record: one or more time-tagged pitch/roll/heave/heading measurements. | Yes |

\* Within a `SWATH_BATHYMETRY_PING`'s subrecord stream: the standard
scale-factor-encoded beam arrays (depth, across/along track, travel time,
beam angle, amplitude, errors, etc.) are decoded and vectorized with numpy;
the KMALL vendor-specific subrecord (id 156 -- Kongsberg SIS 5 /
EM2040-and-newer) is fully field-decoded; every *other* vendor
sensor-specific subrecord (~29 of gsf.h's `GSF_SWATH_BATHY_SUBRECORD_*_SPECIFIC`
ids -- Reson, SeaBat, EM3-series, Klein, R2Sonic, etc.) is identified by name
but not decoded; and gsflib's optional RLE array compression, the 2-bit
packed quality-flags array, and the per-beam backscatter time series
(decoded separately, and only for KMALL, via `-I`/`print_intensity_series()`)
are noted rather than decoded. Anything a decoder can't make sense of falls
back to raw ASCII rendering with a note explaining why.

**Writing / encoding**: not yet implemented. `gsf.OpenFiletoWrite()` exists
but only opens the file -- there is currently no function anywhere in this
codebase that encodes any GSF record. The class and CLI docstrings say
"index and read" deliberately, not "read and write".

## Planned

A `.kmall` → `.gsf` conversion utility (`kmall2gsf.py`), reading Kongsberg
`.kmall` files via the sibling [`kmall`](https://github.com/valschmidt/kmall)
package and writing `PROCESSING_PARAMETERS`, `SOUND_VELOCITY_PROFILE`,
`SWATH_BATHYMETRY_PING` (with the `KMALL_SPECIFIC` sensor-specific
subrecord), and `ATTITUDE` records. This is what motivates adding
write/encode support to the `gsf` class described above -- currently in
design, not yet implemented.

## Testing

    pip install pytest
    pytest tests/

The test suite has two layers: synthetic, hand-crafted GSF byte streams that
independently verify the record framing math (bit-packing, checksum
handling, error conditions) against gsflib's documented encoding, and tests
against the real sample `.gsf` files in `data/GSF/` that check invariants
that must hold for any valid GSF file (every byte accounted for, every
record type recognized, file starts with a header record).
