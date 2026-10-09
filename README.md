# GSF file reader/writer (`gsfu`)

`gsfu` is both a Python library and a command line utility for indexing,
reading, and writing sonar data files in the **Generic Sensor Format
(GSF)**. As a library, it gives Python code direct access to a GSF file's
records as ordinary dictionaries of values and `numpy` arrays, and lets a
program write new GSF files from scratch. As a command line tool, it can
index a GSF file and print a summary of the record types it contains, and
it can decode and print any record's fields for inspection and debugging.
The repository also includes [`kmall2gsf.py`](#converting-other-sonar-formats-to-gsf),
a Kongsberg `.kmall` to `.gsf` converter built on the same write API.

This is a pure-Python implementation: `gsfu.py` re-implements the GSF file
format directly in Python, using Python's `struct` module to unpack the
on-disk record framing, rather than wrapping a compiled C library. This
keeps the tool dependency-free (it needs only `pandas` and `numpy`) and
portable to any platform Python runs on. A ping's beam arrays (depth,
travel time, amplitude, and so on, which GSF stores as scaled one, two,
or four-byte integers) are decoded with `numpy` rather than a per-beam
Python loop, which was measured to be roughly one hundred times faster
than a naive per-beam approach on real, multi-hundred-beam pings.

The record framing, naming, and constants used here are ported from the
reference C implementation of the GSF library ("gsflib"), version 3.11,
distributed by Leidos, Inc. under the LGPL 2.1 from the Leidos product
page: <https://www.leidos.com/products/ocean-marine>. GSF v3.11 was
released in 2025, but the copyright notices in its source files were not
updated and still read "Copyright 2019 Leidos, Inc.". That distribution
unpacks to a single `GSF_03-11/` directory, and the source files cited
throughout this project (`gsf.h`, `gsf.c`, `gsf_dec.c`, `gsf_enc.c`, and
so on) are named relative to that directory's top level. It is the only
copy of gsflib this project was written against; other copies found
online (such as GitHub mirrors) may or may not be identical, and are not
authoritative. No gsflib source code is reused. `gsfu.py` is an
independent re-implementation, with constant names (`GSF_RECORD_HEADER`,
`RecordType`, `gsfDataID`, and so on) and comments chosen to match
gsflib, so that the code is recognizable to anyone already familiar with
the reference library.

## Installation

    pip install gsfu

The `kmall2gsf.py` converter also needs the Kongsberg `.kmall` reader,
published on PyPI as `pykmall`. To install it along with `gsfu`:

    pip install "gsfu[kmall]"

To work on `gsfu` itself, install it from a clone of this repository in
editable mode instead:

    pip install -r requirements.txt
    pip install -e ".[kmall]"

## Using `gsfu` as a command line utility

Running `gsfu.py -h` prints the full set of available options, along
with a reference list of every GSF record type:

    $ gsfu.py -h
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

`-V` indexes a file and prints a summary of the record types it
contains, without decoding any record's fields. This makes it fast even
on files with several hundred thousand records:

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

`-p` decodes and prints every record's fields to stdout, for debugging.
Scalar fields are printed as `key : value` pairs, and any per-beam data
is printed as a table, one row per beam. Passing a record type name
after `-p` restricts the output to just that type:

    $ gsfu.py -f data/GSF/0268_20240826_052757_EM712.gsf -p SWATH_BATHYMETRY_PING
    === GSF_RECORD_SWATH_BATHYMETRY_PING  offset=304484  size=25664 ===
      PingTime           : 2024-08-26T05:27:53.719285+00:00
      Longitude_deg      : -169.059375
      Latitude_deg       : -14.2062916
      NumberBeams        : 400
      ...
      # IntensityTimeSeries (21, 13287 bytes) not decoded here: use gsf.print_intensity_series() / -I
    -- Beams --
           Depth_m  AcrossTrack_m  AlongTrack_m  TravelTime_s  BeamAngle_deg  ...
    Beam
    0     1014.072       -1150.11         62.35       2.04725         -75.85  ...
    1     1013.272       -1142.42         61.84       2.03877         -75.65  ...
    -- SensorSpecific (KMALL_SPECIFIC, id=156) --
      GSFKMALLVersion             : 0
      DgmType                     : 1
      DgmVersion                  : 3
      SystemID                    : 71
      EchoSounderID               : 712
      ...
    -- TxSectors --
               TxSectorNumb  TxArrNumber  TxSubArray  SectorTransmitDelay_sec  ...
    TxSectors
    0                     0            0           0                 0.031502  ...
    1                     1            0           0                 0.000000  ...

A ping's vendor-specific metadata is printed under its own
`SensorSpecific` heading, together with any per-element table it
reports, such as the `TxSectors` table shown above for a KMALL ping.

The per-beam backscatter time series is deliberately left out of `-p`'s
output, since it can hold tens of thousands of samples per ping and
would dwarf the rest of the output. `-I` prints it separately, one CSV
row per beam, across every ping in the file:

    $ gsfu.py -f data/GSF/0268_20240826_052757_EM712.gsf -I
    # ping offset=304484 ping_time=2024-08-26T05:27:53.719285+00:00
    # Beam,SampleCount,DetectSample,StartRangeSamples,Sample0,Sample1,...
    0,140,8,38926,32666,32639,32615,32596,...
    1,17,9,34830,32690,32681,32672,32674,...

## Using `gsfu` as a library

The `gsf` class is the main entry point for using `gsfu` from Python
code. `gsf.iter_records()` walks a file's records in order and yields each
one decoded, optionally restricted to one record type. The example below
opens a file, takes its first `GSF_RECORD_SWATH_BATHYMETRY_PING` record
(decoding its backscatter intensity time series too), and prints what the
decoded record holds:

```python
from GSFU.gsfu import gsf

G = gsf("data/GSF/0264_20240826_022211_EM712.gsf")

# Each item is (record type, byte offset in the file, decoded record).
record_type, offset, d = next(G.iter_records("SWATH_BATHYMETRY_PING", decode_intensity=True))

for k, v in d.items():
    print(f"d[{k!r}]".ljust(25), type(v).__name__, getattr(v, "shape", ""))
```

This prints:

```
d['PingTime']             str
d['Longitude_deg']        float
d['Latitude_deg']         float
d['NumberBeams']          int
d['CenterBeam']           int
d['PingFlags']            int
d['TideCorrector_m']      float
d['DepthCorrector_m']     float
d['Heading_deg']          float
d['Pitch_deg']            float
d['Roll_deg']             float
d['Heave_m']              float
d['Course_deg']           float
d['Speed_kn']             float
d['Height_m']             float
d['SEP_m']                float
d['GPSTideCorrector_m']   float
d['Beams']                dict
d['Notes']                list
d['IntensityTimeSeries']  dict
d['SensorSpecificID']     int
d['SensorSpecific']       dict
```

A decoded record is one dictionary. Its fixed fields sit at the top level,
and its per-beam data sits in tables: dictionaries mapping each column
name to a `numpy` array with one element per beam.

A ping's `Beams` table always has a column for every beam array GSF
defines, one for each label in `BEAM_ARRAY_SUBRECORD_IDS`. A column is
`None` when the ping did not carry that subrecord, so
`d["Beams"][label] is None` tells you whether it was present.
`BEAM_ARRAY_SUBRECORD_IDS[label]` gives the GSF subrecord id each column
comes from. This ping carries 12 of them:

```python
for k, v in d["Beams"].items():
    if v is not None:
        print(f"d['Beams'][{k!r}]".ljust(40), type(v).__name__, getattr(v, "shape", ""))
```

```
d['Beams']['Depth_m']                    ndarray (400,)
d['Beams']['AcrossTrack_m']              ndarray (400,)
d['Beams']['AlongTrack_m']               ndarray (400,)
d['Beams']['TravelTime_s']               ndarray (400,)
d['Beams']['BeamAngle_deg']              ndarray (400,)
d['Beams']['MeanCalAmplitude_dB']        ndarray (400,)
d['Beams']['QualityFactor']              ndarray (400,)
d['Beams']['BeamFlags']                  ndarray (400,)
d['Beams']['BeamAngleForward_deg']       ndarray (400,)
d['Beams']['VerticalError_m']            ndarray (400,)
d['Beams']['HorizontalError_m']          ndarray (400,)
d['Beams']['SectorNumber']               ndarray (400,)
```

`SensorSpecific` holds the fields of the ping's vendor-specific subrecord,
and `SensorSpecificID` names the vendor format. In this file the format
is Kongsberg's KMALL, which has 72 fields, including a `TxSectors` table.

`IntensityTimeSeries` holds the ping's backscatter samples. It is present
only when `decode_intensity=True` is passed. It contains two tables:

```python
its = d["IntensityTimeSeries"]
for k, v in its.items():
    print(f"its[{k!r}]".ljust(28), type(v).__name__, getattr(v, "shape", ""))
for table in ("Beams", "Samples"):
    for k, v in its[table].items():
        print(f"its[{table!r}][{k!r}]".ljust(40), type(v).__name__, getattr(v, "shape", ""))
```

```
its['BitsPerSample']         int
its['AppliedCorrections']    int
its['Beams']                 dict
its['Samples']               dict
its['Beams']['DetectRangeSample']        ndarray (400,)
its['Samples']['Beam']                   ndarray (1736,)
its['Samples']['RangeSample']            ndarray (1736,)
its['Samples']['Value']                  ndarray (1736,)
```

The `Beams` table has one row per beam. Its one column,
`DetectRangeSample`, is the range of the beam's bottom detection, counted
in samples.

The `Samples` table has one row per sample, for every beam in the ping,
in beam order. `Value` is the sample itself. `Beam` is the number of the
beam the sample belongs to. `RangeSample` is the sample's range, counted
in samples, in the same units as `DetectRangeSample`. A beam's samples
have consecutive ranges. The number of samples in each beam, and the
range where each beam's samples start, follow from these columns and are
not stored separately. These columns let a correction be applied to every
sample in one vectorized operation. For example, a per-beam gain `gain`
applies as `gain[its["Samples"]["Beam"]]`, and a range-dependent
correction can be computed directly from `its["Samples"]["RangeSample"]`.

Any table converts to a `pandas.DataFrame` in a single call when that is
more convenient for analysis, for example
`pd.DataFrame(d["IntensityTimeSeries"]["Samples"])`. For a ping's
`Beams` table, leave out the absent columns first:
`pd.DataFrame({k: v for k, v in d["Beams"].items() if v is not None})`.

A few unusual subrecords decode to something other than one value per
beam. A zero-length beam array subrecord, which is present but holds no
values, decodes to an empty array, unless it is the last subrecord in the
ping: the reference gsflib library stops reading before such a trailing
subrecord, and so does `gsfu`, so that both see the same subrecords.
(gsflib itself never writes a zero-length subrecord, and neither does
`gsfu`: an empty column is not written.) A beam flags subrecord whose size is
not one byte per beam decodes to the bytes it holds, and the ping's
`Notes` list records the mismatch. A beam array that cannot be decoded at
all, because no scale factors are available for it or it uses gsflib's
run-length compression, stays `None`, and `Notes` says why.

`iter_records()` indexes the file on first use (`gsf.index_file()`,
which builds `G.Index`, a `pandas.DataFrame` with one row per record) and
then reads each record with a single seek, skipping records of other
types without reading them. Swath bathymetry pings are decoded with the
scale factors carried from one ping to the next, as the format requires.
Every record type comes back as a single dictionary, in the shape of that
type's `new_*()` template, such as `new_comment()` or
`new_sound_velocity_profile()`. A decoded record can be passed straight
back to the matching `write_*()` method, as described in
[Writing a GSF file](#writing-a-gsf-file). For casual
inspection of a whole file, `gsf.print_records()` (the `-p` option's
underlying method) prints every record as text.

To get every attitude measurement in a file at once, use
`gsf.read_attitude()`. It returns one dictionary in the same shape as a
single decoded attitude record (`Time`, `Pitch_deg`, `Roll_deg`,
`Heave_m`, `Heading_deg`), but covering the whole file, and it reads the
records together with vectorized `numpy` operations rather than one at a
time, so it stays fast even for files that store high-rate attitude as
one measurement per record:

```python
G = gsf("data/GSF/0264_20240826_022211_EM712.gsf")
attitude = G.read_attitude()
print(attitude["NumMeasurements"], attitude["Time"][0], attitude["Roll_deg"].std())
```

## Writing a GSF file

Every record type is written from a single dictionary, the same shape the
reader returns. Each type has a template function, `new_*()`, that
returns that dictionary with every field name already present and set to
its default ("not available" where GSF defines such a value), so you fill
in only what you know and pass it to the matching `write_*()` method:

```python
from GSFU.gsfu import gsf, new_attitude, new_comment, new_swath_bathymetry_ping

G = gsf("out.gsf")
G.write_header()  # always the first record

comment = new_comment()
comment.update(CommentTime=1724650073.7, Comment="written by gsfu")
G.write_comment(comment)

attitude = new_attitude()
attitude.update(Time=[1724650073.70, 1724650073.71], Pitch_deg=[0.12, 0.10],
                Roll_deg=[-0.5, -0.4], Heave_m=[0.02, 0.01], Heading_deg=[210.5, 210.6])
G.write_attitude(attitude)

ping = new_swath_bathymetry_ping()
ping.update(PingTime=1724650073.72, Longitude_deg=-129.98, Latitude_deg=45.93, NumberBeams=3)
ping["Beams"] = {"Depth_m": [1010.2, 1004.8, 1011.9], "AcrossTrack_m": [-850.0, 0.0, 850.0]}
G.write_swath_bathymetry_ping(ping)

G.closeFile()
```

A field left at its template default is written as "not available". A
required field left as `None`, such as a ping's `PingTime`, raises a
`ValueError` that names it. Times may be POSIX seconds, `datetime` objects, or
ISO 8601 strings. Because reading and writing use the same dictionaries,
a record read with `iter_records()` can be written back unchanged, and
copying a file record by record this way reproduces it byte for byte.

| Record type | Template | Write method |
|---|---|---|
| `GSF_RECORD_HEADER` | `new_header()` | `write_header()` |
| `GSF_RECORD_SWATH_BATHYMETRY_PING` | `new_swath_bathymetry_ping()` | `write_swath_bathymetry_ping()` |
| `GSF_RECORD_SOUND_VELOCITY_PROFILE` | `new_sound_velocity_profile()` | `write_sound_velocity_profile()` |
| `GSF_RECORD_PROCESSING_PARAMETERS` | `new_name_value_parameters()` | `write_processing_parameters()` |
| `GSF_RECORD_SENSOR_PARAMETERS` | `new_name_value_parameters()` | `write_sensor_parameters()` |
| `GSF_RECORD_COMMENT` | `new_comment()` | `write_comment()` |
| `GSF_RECORD_HISTORY` | `new_history()` | `write_history()` |
| `GSF_RECORD_NAVIGATION_ERROR` | `new_navigation_error()` | `write_navigation_error()` |
| `GSF_RECORD_SWATH_BATHY_SUMMARY` | `new_swath_bathy_summary()` | `write_swath_bathy_summary()` |
| `GSF_RECORD_SINGLE_BEAM_PING` | `new_single_beam_ping()` | `write_single_beam_ping()` |
| `GSF_RECORD_HV_NAVIGATION_ERROR` | `new_hv_navigation_error()` | `write_hv_navigation_error()` |
| `GSF_RECORD_ATTITUDE` | `new_attitude()` | `write_attitude()` |

A ping's vendor sensor-specific subrecord (its `SensorSpecific`
dictionary, identified by `SensorSpecificID`) has a template of its own
for every supported vendor family, named after it: for example
`new_kmall_specific()`, `new_em4_specific()`, `new_reson7125_specific()`,
or, for single-beam pings, `new_echotrac_specific()`. Tables inside them
have row templates, such as `new_kmall_tx_sector()` for KMALL's
`TxSectors`. See [`convert.md`](convert.md) for a complete worked example.

## Converting other sonar formats to GSF

See [`convert.md`](convert.md) for a complete, worked example of building
a GSF file from another sonar format using the write API. It uses the
conversion of a Kongsberg `.kmall` file to GSF, implemented in
`GSFU/kmall2gsf.py`, as its running example, and explains how to build up
a ping record, how scale factors are chosen, and how to mark a field as
not available rather than write a misleading zero.

## Known limitations

This library does not decode gsflib's optional run-length-encoded beam
array compression. A compressed array's compression flag is read and
reported, but the array itself is left undecoded, since none of the
sample GSF files used to build and test this library are compressed.

## Scale factors

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

## Bugs found in the reference gsflib C library

Porting `gsf_dec.c`/`gsf_enc.c` field-by-field surfaced a handful of real defects
in the reference library itself (checked against the GSF v3.11 source, not
assumed). This codebase does not replicate them; each is called out in the
relevant function's docstring, and summarized here:

- **`gsfEncodeNavigationError()` has a rounding bug for negative values.** It
  rounds both fields with an unconditional `+ 0.501` and no sign check, instead
  of the sign-aware `+/-0.501` convention used everywhere else in `gsf_enc.c`.
  For a negative error value this systematically rounds toward zero instead of
  to the nearest representable value -- e.g. a longitude error of -1.29m
  encodes, via the reference's own formula, to -12 (in 1/10m units) rather than
  the correctly-rounded -13. `_encode_navigation_error()` uses this module's
  standard sign-correct `_gsf_round()` instead.
- **`DecodeReson7100Specific()` (the Reson 7125 decoder) misreads
  `tx_pulse_reserved`.** It reads that field from a stale `stemp` -- a leftover
  2-byte local variable from an earlier field -- instead of the 4-byte `ltemp`
  it had just loaded for this field. Byte-position tracking is unaffected (the
  pointer still advances the correct 4 bytes), only the decoded *value* for
  this one field is wrong. Low real-world impact, since the field is documented
  as reserved/unused, but `_decode_reson7125_specific()` decodes the actual
  wire bytes rather than replicating the misread.
- **`gsfEncodeEM3Specific()` can never write a second (EM3000D dual-head)
  run-time block.** It hardcodes `run_time_id = 1` unconditionally; the code
  path that would set bit 1 to include a second head's run-time parameters is
  present in the source but entirely commented out -- dead code, a real
  limitation of gsflib as currently shipped, not a deliberate design choice
  (the wire format and `DecodeEM3Specific()` both fully support it).
  `_encode_em3_specific()` writes whatever the caller actually supplies (zero,
  one, or two heads).

Separately, a number of C encoder functions (`EncodeSeaBat8101Specific`,
`EncodeReson7100Specific`, `EncodeResonTSeriesSpecific`,
`EncodeGeoSwathPlusSpecific`, `EncodeR2SonicSpecific`, and the run-time/PU-status
field writes shared by `EncodeEM4Specific`/`EncodeEM3RawSpecific`/
`EncodeEM3Specific`) round some fields with a plain truncating cast -- no
rounding offset at all, unlike every other scaled field in the same function,
which do round. This introduces a small, one-directional bias (always toward
zero) rather than rounding to the nearest representable value. It's minor
enough that it's unlikely to matter in practice (the affected fields are all
non-negative in normal use, and the bias is at most a fraction of one
quantization step) -- but rather than replicate it field-by-field, every
encoder in this module rounds every scaled field with the same, consistent,
correctly-rounding `_gsf_round()`. `gsfEncodeHVNavigationError()` similarly
rounds `vertical_error` with a plain `+/- 0.5` instead of the `+/- 0.501` used
for `horizontal_error` right next to it in the same function -- functionally
identical to `_gsf_round()` except exactly on a 0.5 fractional boundary, so
treated the same way for consistency.

One wire-format quirk is replicated rather than corrected, since
correcting it would make the written files differ from gsflib's own:
`EncodeBRBIntensity()` counts its own four-byte identifier word in the
size field of the per-beam intensity time series subrecord, while every
other ping subrecord's size field counts only the bytes that follow the
identifier word. `gsf_dec.c` never reads that size field, so gsflib never
notices; `gsfu.py` writes the size the same way gsflib does, and allows
for it when reading. Re-encoding the intensity time series subrecords in
the EM712 sample files reproduces the original bytes exactly.
