# GSF file reader (`gsfu`)

A Python class and command line utility for indexing, reading, and writing
sonar data files in the **Generic Sensor Format (GSF)**. The `gsfu.py`
utility can index a GSF file and print a summary of the record types it
contains (count and bytes consumed by each) -- more functionality (reading
and writing individual records) is planned.

    ./GSFU/gsfu.py -h
    usage: gsfu.py [-h] [-f GSF_FILENAME] [-V] [-v]

    A python script (and class) for indexing, reading, and writing Generic
    Sensor Format (GSF) data files.

    options:
      -h, --help        show this help message and exit
      -f GSF_FILENAME    The path and filename to parse.
      -V                 Index the file and print a summary of its record types
                         (count and bytes consumed by each).
      -v                 Increasingly verbose output (e.g. -v -vv), for
                         debugging use -vv

## Why a pure-Python implementation

The [UK Hydrographic Office's `gsfpy`](https://github.com/UKHO/gsfpy) package
wraps the reference GSF C library (`libgsf`) via `ctypes`. It's a fine choice
when a prebuilt `libgsf` shared library is available for your platform, but
the shared libraries bundled in the `gsfpy` wheel are Linux-only -- they
cannot be loaded on macOS or Windows, and even importing `gsfpy` on those
platforms raises at import time. That made it unworkable as this project's
runtime dependency.

Instead, `gsfu.py` re-implements the parts of the GSF file format needed for
indexing directly in Python, using `struct` to unpack the on-disk record
framing -- the same approach [`kmall.py`](https://github.com/valschmidt/kmall)
takes for Kongsberg's `.kmall` format. This keeps the tool dependency-free
(just `pandas` and `numpy`) and portable to any platform Python runs on. The
GSF record-array beam data (depth, travel time, amplitude, etc., stored as
scaled 1/2/4-byte integers per `gsfScaleFactors`) is well suited to a
`numpy`-vectorized decode when that functionality is added, avoiding a
per-beam Python loop.

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

## Record types supported by the indexer

Every top-level GSF record type is recognized by the indexer (payloads are
not decoded yet -- only the record framing is parsed):

| Record type | Description |
|---|---|
| `GSF_RECORD_HEADER` | GSF header record. Identifies the GSF version used to create the file. |
| `GSF_RECORD_SWATH_BATHYMETRY_PING` | Data structure for a ping from a swath bathymetric system. |
| `GSF_RECORD_SOUND_VELOCITY_PROFILE` | Sound velocity profile record. |
| `GSF_RECORD_PROCESSING_PARAMETERS` | Internal record structure for processing parameters. |
| `GSF_RECORD_SENSOR_PARAMETERS` | Sensor parameters record. |
| `GSF_RECORD_COMMENT` | Comment record. |
| `GSF_RECORD_HISTORY` | History record. |
| `GSF_RECORD_NAVIGATION_ERROR` | Navigation error record. (Obsolete; replaced by `GSF_RECORD_HV_NAVIGATION_ERROR`.) |
| `GSF_RECORD_SWATH_BATHY_SUMMARY` | Swath bathymetry summary record. |
| `GSF_RECORD_SINGLE_BEAM_PING` | Single beam ping record. |
| `GSF_RECORD_HV_NAVIGATION_ERROR` | Horizontal/Vertical navigation error record. |
| `GSF_RECORD_ATTITUDE` | Attitude record: one or more time-tagged pitch/roll/heave/heading measurements. |

## Testing

    pip install pytest
    pytest tests/

The test suite has two layers: synthetic, hand-crafted GSF byte streams that
independently verify the record framing math (bit-packing, checksum
handling, error conditions) against gsflib's documented encoding, and tests
against the real sample `.gsf` files in `data/GSF/` that check invariants
that must hold for any valid GSF file (every byte accounted for, every
record type recognized, file starts with a header record).
