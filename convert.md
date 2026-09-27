# Writing GSF files with `gsfu`

This is a worked tutorial for writing your own GSF files with `gsfu.py`'s
`gsf` class, aimed at someone who has a source of sonar data (their own
reader, a different vendor format, synthetic data for testing) and wants to
get it into a `.gsf` file that other software can read. It walks through the
same four steps `kmall2gsf.py` (in this repo) uses to convert Kongsberg
`.kmall` files: open a file for writing, build a record-equivalent `dict`,
populate it with real values, and hand it to a `write_*` method to encode.

If you just want to convert a `.kmall` file, skip to
[`kmall2gsf.py`](README.md#converting-other-sonar-formats-to-gsf) in the
README -- this document is about the lower-level `write_*` API it's built on.

## Reference frame conventions

Before populating any of the fields below, it's worth being precise about
which direction "positive" means for each of them -- getting a sign backward
produces a file that's silently wrong rather than one that fails to write.
The conventions here are gsf_spec.pdf section 3.4's ("Ship-based coordinate
system") and 3.6.7's ("Angular measures"), and every `write_*` method in
`gsfu.py` follows them exactly -- there is no code-side correction or
sign-flip anywhere in this codebase, so a value you hand to `write_*` is
written to disk with the same sign you gave it.

**Ship-based Cartesian axes** (used for beam offsets -- `Depth_m`,
`AcrossTrack_m`, `AlongTrack_m` -- and for heave):

* **x** points in the vessel's direction of travel (forward/bow).
* **y** points to **starboard** (right, facing forward) -- chosen so
  x/y/z form a right-handed system.
* **z** points **down** -- consistent with depths being positive numbers.

Concretely:

* `Depth_m` is positive and increases with depth below the water surface.
* `AcrossTrack_m` is positive to **starboard**, negative to **port**.
* `AlongTrack_m` is positive **forward** of the ping's reference position,
  negative **astern** of it.
* `Heave_m` (both the ping header field and the `ATTITUDE` record's
  `heave_m`) is positive when the vessel moves **down** relative to the
  mean/reference surface -- i.e. it shares the z-axis's down-positive sense.

**Angular measures** (heading, course, yaw, roll, pitch -- all in degrees,
stored internally as hundredths of a degree):

* `Heading_deg`/`heading_deg` and `Course_deg` are measured from true
  North and increase as the vessel turns to **starboard** (i.e. standard
  compass convention: 090 is due east). Heading is the direction the bow
  points; course is the direction of travel through the water -- they
  differ under drift/crab.
* `Roll_deg`/`roll_deg` increases as the vessel's **starboard side moves
  down**. Valid range -180.00 to +180.00.
* `Pitch_deg`/`pitch_deg` increases as the **bow moves up**. Valid range
  -180.00 to +180.00.

**Geographic position**: `Latitude_deg` is positive in the Northern
Hemisphere, `Longitude_deg` is positive in the Eastern Hemisphere (i.e.
ordinary signed decimal degrees -- no special handling needed).

**Height/separation fields** (`Height_m`, `SEP_m`): positive `Height_m` is
**above** the ellipsoid; positive `SEP_m` (ellipsoid-to-chart-datum
separation) indicates the chart datum is **above** the ellipsoid. Both are
the opposite sense from depth (up-positive, not down-positive) -- easy to
get backward if you're not watching for it.

## 1. Open a file for writing

```python
from GSFU.gsfu import gsf

G = gsf("my_survey.gsf")
G.write_header()          # must be the first record in any GSF file
```

`write_header()` opens the file (if it isn't already open) and writes the
`GSF_RECORD_HEADER` record identifying the GSF version. It also remembers
that version on `G.gsfVersion`, which `write_swath_bathymetry_ping()` needs
later to decide the ping record's exact layout (every version this codebase
writes, `GSF_VERSION` = `"GSF-v03.11"` by default, uses the newer 3-field
layout with `Height_m`/`SEP_m`/`GPSTideCorrector_m`).

Every other record type is optional, and you can interleave them in any
order after the header -- a typical file writes `PROCESSING_PARAMETERS`
once near the top, then one `SOUND_VELOCITY_PROFILE` per cast, then
`ATTITUDE` and `SWATH_BATHYMETRY_PING` records interleaved (or all the
attitude first, then all the pings -- GSF doesn't require any particular
interleaving, only that the header comes first).

## 2. Build a record-equivalent dict, and 3. populate it

Every `write_*` method takes a single `dict` -- the record -- holding plain
Python values: scalars, and array-likes (lists or numpy arrays) for
anything per-measurement/per-beam. There's no separate "record object" to
instantiate: the dict *is* the record, using the same field names
`gsfu.py -p` prints and the corresponding `_decode_*` function returns, so
a record read with `gsf.iter_records()` can be passed straight back to its
`write_*` method. Each record type has a `new_*()` function (for example
`new_sound_velocity_profile()`) that returns that dict with every field
name already present, so you only fill in what you know. That symmetry is
deliberate -- if you've ever looked at `-p`'s output for a record type, you
already know the field names its `write_*` counterpart expects.

### Processing parameters

```python
G.write_processing_parameters({
    "ParamTime": 1724650073.719285,
    "Parameters": {
        "REFERENCE": "TRUE HEADING",
        "SYSTEM_DRAFT": "0.500",
        "PLATFORM_TYPE": "SURFACE_SHIP",
    },
})
```

`Parameters` is just `{name: value}` -- both strings. `ParamTime` is a POSIX
epoch timestamp (float), a `datetime`, or an ISO8601 string; all three are
accepted everywhere a time field appears in this API (see `_gsf_epoch()` in
`gsfu.py` if you want the details). `write_sensor_parameters()` takes the
identical record, for `GSF_RECORD_SENSOR_PARAMETERS`.

### Sound velocity profile

```python
G.write_sound_velocity_profile({
    "ObservationTime": 1724650073.719285,
    "ApplicationTime": 1724650073.719285,
    "Latitude_deg": 45.925417,
    "Longitude_deg": -129.981855,
    "Depth_m": [0.0, 5.0, 10.0, 50.0, 100.0],
    "SoundSpeed_mPerSec": [1500.1, 1499.8, 1498.2, 1495.0, 1490.3],
})
```

`Depth_m` and `SoundSpeed_mPerSec` are parallel arrays, one entry per cast
sample, and must be the same length.

### Attitude

```python
G.write_attitude({
    "Time": [1724650073.70, 1724650073.75, 1724650073.80],
    "Pitch_deg": [0.12, 0.10, 0.09],
    "Roll_deg": [-0.5, -0.4, -0.3],
    "Heave_m": [0.02, 0.01, 0.00],
    "Heading_deg": [210.5, 210.6, 210.6],
})
```

One `GSF_RECORD_ATTITUDE` record can carry many time-tagged samples -- all
five arrays must be the same length, and one record can span at most
65.535 s, since each sample's time is stored as a 16-bit millisecond offset
from the record's first sample. Batch attitude samples (e.g. one record per
second) rather than writing one record per sample: reading a GSF file costs
mostly per record, not per measurement, so 100 Hz attitude written one
sample per record reads roughly 100 times slower than the same data in
one-second records. The one cost of batching is time resolution: a
record stores its first sample's time to the nanosecond but each later
sample's as a whole-millisecond offset from it, so those times are
rounded to within 0.5 ms. `kmall2gsf.py` writes one record per second by
default (`-t/--attitude-record-seconds` changes the window). Ping
attitude is interpolated from the original, unrounded samples either way.

### Swath bathymetry ping

This is the record most worth walking through carefully, since it has the
most moving parts: fixed scalar fields, a table of per-beam arrays, and an
optional vendor-specific block. It also has by far the most scalar field
names to keep straight (17, plus ~65 more if you add the KMALL
vendor-specific block below) -- rather than writing that dict out by hand
from memory, start from `new_swath_bathymetry_ping()`, which
returns every valid key name already present, so there's nothing to look
up and no name to mistype:

```python
from GSFU.gsfu import new_swath_bathymetry_ping

record = new_swath_bathymetry_ping()
record.update(
    PingTime=1724650073.719285,
    Longitude_deg=-129.981855,
    Latitude_deg=45.925417,
    NumberBeams=3,
    CenterBeam=1,
    Heading_deg=210.5,
    Pitch_deg=0.10,
    Roll_deg=-0.4,
    Heave_m=0.01,
    Height_m=-32.1,   # antenna height above the ellipsoid, if known
)

record["Beams"] = {
    "Depth_m":         [7.342, 7.356, 7.370],
    "AcrossTrack_m":   [-3.89, 0.0, 3.89],
    "AlongTrack_m":    [0.0, 0.0, 0.0],
    "TravelTime_s":    [0.01033, 0.01033, 0.01032],
    "BeamAngle_deg":   [-45.0, 0.0, 45.0],
    "QualityFactor":   [50, 52, 49],
}

G.write_swath_bathymetry_ping(record)
```

`new_swath_bathymetry_ping()` sets the four required fields
(`PingTime`, `Longitude_deg`, `Latitude_deg`, `NumberBeams`) to `None` as a
placeholder -- `write_swath_bathymetry_ping()` raises `ValueError` naming
whichever one you forget to overwrite -- and pre-fills every optional
field with its **GSF_NULL_\* sentinel** (or, for `CenterBeam`, `PingFlags`,
and `GPSTideCorrector_m`, `0`/`0.0` -- gsf.h defines no sentinel for those
three). Fields you never touch are written as "not available", not as a
misleading `0`/`0.0` -- see "Marking a field as not available" below for
why that distinction matters. (Building `record` as a plain `dict` literal
the way earlier GSF-writing tools do also still works exactly as before --
the template is a convenience, not a required calling convention.)

Every array in `record["Beams"]` must have exactly `NumberBeams` entries. The
dict *key* is what selects which subrecord gets written and, in turn, its
scale factor -- `_beam_array_subrecord_id()` resolves each label (e.g.
`"Depth_m"`, `"BeamAngle_deg"`) to its `GSF_SWATH_BATHY_SUBRECORD_*` id.
Any key `write_swath_bathymetry_ping()` doesn't recognize raises
`KeyError` -- see `_PING_ARRAY_SUBRECORDS` in `gsfu.py` for the full list of
recognized labels.

To add the KMALL (Kongsberg SIS 5) vendor-specific subrecord, the same
template pattern applies -- `new_kmall_specific()` for the scalar block,
`new_kmall_tx_sector()` for each row of the `TxSectors` table -- assigned
to `record["SensorSpecificID"]`/`record["SensorSpecific"]`:

```python
from GSFU.gsfu import new_kmall_specific, new_kmall_tx_sector

kmall_specific = new_kmall_specific()
kmall_specific.update(EchoSounderID=712, PingRate_Hz=1.2)

sector0 = new_kmall_tx_sector()
sector0.update(TxSectorNumb=0, CentreFreq_Hz=70000.0, TiltAngleReTx_deg=0.0)
sector1 = new_kmall_tx_sector()
sector1.update(TxSectorNumb=1, CentreFreq_Hz=71000.0, TiltAngleReTx_deg=15.0)

record["SensorSpecificID"] = 156
# A table is a dict of column name -> one value per row.
tx_sectors = {name: [sector0[name], sector1[name]] for name in sector0}
record["SensorSpecific"] = {**kmall_specific, "TxSectors": tx_sectors}

G.write_swath_bathymetry_ping(record)
```

Note the spelling: `CentreFreq_Hz`, not `CenterFreq_Hz` -- one of the ~65
vendor-specific field names carried over from the originating `.kmall`
format's own (British) spelling, and exactly the kind of thing
`new_kmall_specific()`/`new_kmall_tx_sector()` save you from having to get
right from memory. `kmall_specific` is a flat dict of scalar fields
(matching the names `-p` prints under `-- SensorSpecific (KMALL_SPECIFIC,
id=156) --`); `record["SensorSpecific"]["TxSectors"]` is a table -- a dict
of column name -> array, one element per transmit sector (up to
`GSF_MAX_KMALL_SECTORS` = 9) -- the exact shape
`_decode_swath_bathymetry_ping()` returns, so a decoded KMALL ping's
`TxSectors` table can be assigned back in unmodified. A `pandas.DataFrame`
(for example `pd.DataFrame([sector0, sector1])`) is accepted too. Both
`SensorSpecificID`/`SensorSpecific` are optional -- omit them entirely for a
non-KMALL system, or if you don't need the vendor-specific block. Unlike
the ping scalars above, gsf.h defines no null-value convention for these
vendor-specific fields, so both templates default every field to plain
`0`/`0.0` -- see "Marking a field as not available" below.

## 4. Scale factors

GSF stores each beam array as scaled integers (1, 2, or 4 bytes), not raw
floats, to keep files compact -- `value_on_disk = round((value + offset) *
multiplier)`. You don't need to think about this at all for the common
case: `write_swath_bathymetry_ping()` uses `DEFAULT_PING_SCALE_FACTORS`
automatically, one multiplier/offset/width per subrecord id, chosen (by
inspecting how real GSF-writing software sets scale factors, per
`gsf_spec.pdf`) to comfortably cover depths from about 1m to 10,000m without
overflowing their field width.

If your data needs different precision or range -- deeper water, unusually
high-precision measurements -- pass your own table:

```python
from GSFU.gsfu import DEFAULT_PING_SCALE_FACTORS

my_scale_factors = dict(DEFAULT_PING_SCALE_FACTORS)
my_scale_factors[1] = (10000.0, 0.0, 4, False)  # Depth_m: 4 bytes, 0.1mm precision

G.write_swath_bathymetry_ping(record, scale_factors=my_scale_factors)
```

Each entry is `subrecordID: (multiplier, offset, field_width_bytes,
signed)`. `write_swath_bathymetry_ping()` raises `ValueError` if a scaled
value doesn't fit its field width (e.g. a value too large for the
multiplier/width you chose) -- that's your signal the scale factor needs
adjusting, not the data.

## Marking a field as not available (null values)

Many of the fields above are perfectly valid at `0` -- a `Speed_kn` of
`0.0` is a vessel stopped in the water; a `Roll_deg` of `0.0` is a level
ship; a `Course_deg` of `0.0` is a real heading, due north. If a source
system doesn't measure or compute one of these values at all, writing `0`
in its place would silently claim "measured, and the answer was zero" --
indistinguishable, to anything reading the file back later, from a real
zero measurement. gsf.h avoids this by defining an explicit "not available"
sentinel for most scalar ping-header fields, chosen from outside (or at the
extreme, implausible edge of) the field's valid range:

| Field | GSF_NULL_\* constant | Value |
|---|---|---|
| `Latitude_deg` | `GSF_NULL_LATITUDE` | 91.0 |
| `Longitude_deg` | `GSF_NULL_LONGITUDE` | 181.0 |
| `Heading_deg`/`heading_deg` | `GSF_NULL_HEADING` | 361.0 |
| `Course_deg` | `GSF_NULL_COURSE` | 361.0 |
| `Speed_kn` | `GSF_NULL_SPEED` | 99.0 |
| `Pitch_deg`/`pitch_deg` | `GSF_NULL_PITCH` | 99.0 |
| `Roll_deg`/`roll_deg` | `GSF_NULL_ROLL` | 99.0 |
| `Heave_m`/`heave_m` | `GSF_NULL_HEAVE` | 99.0 |
| `DepthCorrector_m` | `GSF_NULL_DEPTH_CORRECTOR` | 99.99 |
| `TideCorrector_m` | `GSF_NULL_TIDE_CORRECTOR` | 99.99 |
| `HorizontalError_m` | `GSF_NULL_HORIZONTAL_ERROR` | -1.00 |
| `VerticalError_m` | `GSF_NULL_VERTICAL_ERROR` | -1.00 |
| `Height_m` | `GSF_NULL_HEIGHT` | 9999.99 |
| `SEP_m` | `GSF_NULL_SEP` | 9999.99 |

All are importable from `GSFU.gsfu` (`from GSFU.gsfu import GSF_NULL_SPEED`,
etc.), and `new_swath_bathymetry_ping()` (above) already fills in
every one of them for you -- if you build `record` that way, a field you
never overwrite is automatically "not available", not `0`/`0.0`. Building
`record` as a plain dict instead? Set the constant explicitly rather than
omitting the key, so the "not available" intent is visible in your own
code too:

```python
from GSFU.gsfu import GSF_NULL_COURSE, GSF_NULL_SPEED

record = {
    "PingTime": ping_time, "Longitude_deg": lon, "Latitude_deg": lat,
    "NumberBeams": n,
    "Course_deg": GSF_NULL_COURSE,  # course-made-good not computed by this source
    "Speed_kn": GSF_NULL_SPEED,
}
```

Two fields have no defined null: `CenterBeam` and `GPSTideCorrector_m`
default to plain `0` if omitted, since gsf.h defines no sentinel for
either -- there's no better option than `0` available for these two.
`DRAFT` and `SEP_UNCERTAINTY` (both metadata-file-level fields, not part of
the ping record this API writes) are likewise stuck at `0.0` in gsf.h
itself, for the same reason.

**Per-beam arrays are different.** `Depth_m`, `AcrossTrack_m`,
`TravelTime_s`, and the rest of the beam-array columns in `record["Beams"]`
have *no* meaningful null value -- gsf.h defines all of their nulls as plain
`0.0`, with an explicit warning that a `0.0` beam value does **not** by
itself mean "no data". To mark individual beams unusable, write a
`'BeamFlags'` column in `record["Beams"]` (one `GSF_IGNORE_BEAM`-or-not byte
per beam) instead of trying to signal it through the depth/travel-time/etc.
values themselves:

```python
from GSFU.gsfu import GSF_IGNORE_BEAM

record["Beams"]["BeamFlags"] = [0, GSF_IGNORE_BEAM, 0]  # beam 1 of 3 is unusable
```

And to flag an entire ping as unusable (rather than one beam within it),
set the low bit of `PingFlags`:

```python
from GSFU.gsfu import GSF_IGNORE_PING

record["PingFlags"] = GSF_IGNORE_PING
```

## Closing the file

```python
G.closeFile()
```

## Putting it together

```python
from GSFU.gsfu import gsf, new_swath_bathymetry_ping

G = gsf("my_survey.gsf")
G.write_header()
G.write_processing_parameters({"ParamTime": t0, "Parameters": {"REFERENCE": "TRUE HEADING"}})
G.write_sound_velocity_profile({
    "ObservationTime": t0, "ApplicationTime": t0,
    "Latitude_deg": lat, "Longitude_deg": lon,
    "Depth_m": svp_depths, "SoundSpeed_mPerSec": svp_speeds})

for ping_time, lat, lon, beams in my_pings:
    G.write_attitude(...)  # or batch these separately, see above
    record = new_swath_bathymetry_ping()
    record.update(
        PingTime=ping_time, Latitude_deg=lat, Longitude_deg=lon,
        NumberBeams=len(beams["Depth_m"]))
    record["Beams"] = beams
    G.write_swath_bathymetry_ping(record)

G.closeFile()
```

## Re-encoding a record you decoded

Because `write_*`'s `record` dict shape matches what the `_decode_*`
functions return, you can round-trip a record through `gsfu.py`'s own
reader unmodified -- no repackaging, even for a vendor sensor-specific
subrecord's own tables (`SensorSpecific["TxSectors"]` etc., dicts of
`numpy` arrays on both sides):

```python
from GSFU.gsfu import gsf, _decode_swath_bathymetry_ping

src = gsf("existing.gsf")
src.index_file()
row = src.Index[src.Index["RecordType"] == "GSF_RECORD_SWATH_BATHYMETRY_PING"].iloc[0]
src.FID.seek(int(row["ByteOffset"]))
dataSize, _readSize, data_id = src.read_record_header()
if data_id.checksumFlag:
    src.FID.seek(4, 1)  # skip the optional 4-byte checksum word
payload = src.FID.read(dataSize)
record = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

dst = gsf("copy.gsf")
dst.write_header()
dst.write_swath_bathymetry_ping(record)
dst.closeFile()
```

This is exactly the strategy used to generate the round-trip compatibility
test files in this project's test suite.
