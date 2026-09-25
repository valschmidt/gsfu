# GSF file reader/writer (`gsfu`)

`gsfu` is both a Python library and a command line utility for indexing,
reading, and writing sonar data files in the **Generic Sensor Format
(GSF)**. As a library, it gives Python code direct access to a GSF file's
records as ordinary dictionaries and `pandas.DataFrame`s, and lets a
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
reference C implementation of the GSF library ("gsflib"), copyright 2019
Leidos, Inc., distributed under the LGPL 2.1:
<https://github.com/Spatialnetics/gsflib> (`source/gsf/gsf.h`,
`source/gsf/gsf.c`). No gsflib source code is reused. `gsfu.py` is an
independent re-implementation, with constant names (`GSF_RECORD_HEADER`,
`RecordType`, `gsfDataID`, and so on) and comments chosen to match
gsflib, so that the code is recognizable to anyone already familiar with
the reference library.

## Installation

    pip install -r requirements.txt
    pip install -e .

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
      # IntensityTimeSeries (21, 13291 bytes) not decoded here: use gsf.print_intensity_series() / -I
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
code. The example below opens a file, indexes it, finds the first
`GSF_RECORD_SWATH_BATHYMETRY_PING` record, reads and decodes it, and
prints its keys and a few of its fields:

```python
from GSFU.gsfu import gsf, _decode_swath_bathymetry_ping, _gsf_major_version

G = gsf("data/GSF/0264_20240826_022211_EM712.gsf")
G.index_file()  # builds G.Index, a pandas.DataFrame, one row per record

# Find the first swath bathymetry ping in the file.
pings = G.Index[G.Index["RecordType"] == "GSF_RECORD_SWATH_BATHYMETRY_PING"]
offset = int(pings.iloc[0]["ByteOffset"])

# Seek to it and read its raw payload.
G.OpenFiletoRead()
G.FID.seek(offset)
data_size, _read_size, data_id = G.read_record_header()
if data_id.checksumFlag:
    G.FID.seek(4, 1)  # skip the optional four-byte checksum word
payload = G.FID.read(data_size)

# Decode the payload into a dictionary of the ping's fields.
major_version = _gsf_major_version(G.gsfVersion)
record = _decode_swath_bathymetry_ping(payload, major_version, scale_factors={})

print(list(record.keys()))
print(record["PingTime"], record["Latitude_deg"], record["Longitude_deg"])
```

This prints the following:

```
['PingTime', 'Longitude_deg', 'Latitude_deg', 'NumberBeams', 'CenterBeam', 'PingFlags', 'TideCorrector_m', 'DepthCorrector_m', 'Heading_deg', 'Pitch_deg', 'Roll_deg', 'Heave_m', 'Course_deg', 'Speed_kn', 'Height_m', 'SEP_m', 'GPSTideCorrector_m', 'Beams', 'SensorSpecificID', 'SensorSpecific', 'Notes']
2024-08-26T02:22:13.621820+00:00 -14.2136476 -169.0929353
```

Every decoded ping is a single dictionary. Its fixed scalar fields, such
as `PingTime`, `Latitude_deg`, and `Longitude_deg`, sit directly at the
top level. Its per-beam arrays sit under `Beams`, as a `pandas.DataFrame`
with one row per beam. If the ping carries a vendor-specific subrecord,
its fields sit under `SensorSpecific`, alongside `SensorSpecificID`
naming which vendor format it is.

This example calls `_decode_swath_bathymetry_ping()` directly, since
`gsfu.py` does not yet have a public convenience method for decoding a
single record at a known file offset. For casual inspection of a whole
file, `gsf.print_records()` (the `-p` option's underlying method) or
`gsf.index_file()` will usually be more convenient than reading a single
record by hand as shown above.

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

The per-beam backscatter intensity time series (the
`INTENSITY_SERIES_ARRAY` subrecord) can be decoded but not yet written;
there is no encoder for it in the `write_*` API yet.

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
