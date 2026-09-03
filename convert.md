# Writing GSF files with `gsfu`

This is a worked tutorial for writing your own GSF files with `gsfu.py`'s
`gsf` class, aimed at someone who has a source of sonar data (their own
reader, a different vendor format, synthetic data for testing) and wants to
get it into a `.gsf` file that other software can read. It walks through the
same four steps `kmall2gsf.py` (in this repo) uses to convert Kongsberg
`.kmall` files: open a file for writing, build a record-equivalent `dict`,
populate it with real values, and hand it to a `write_*` method to encode.

If you just want to convert a `.kmall` file, skip to
[`kmall2gsf.py`](README.md#kmall--gsf-conversion-kmall2gsfpy) in the README
-- this document is about the lower-level `write_*` API it's built on.

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

Each `write_*` method other than `write_header()` takes plain Python
values -- `dict`s of scalars, and array-likes (lists or numpy arrays) for
anything per-measurement/per-beam. There's no separate "record object" to
instantiate: the dict *is* the record, using the same field names
`gsfu.py -p` prints and the corresponding `_decode_*` function returns. That
symmetry is deliberate -- if you've ever looked at `-p`'s output for a
record type, you already know the field names its `write_*` counterpart
expects.

### Processing parameters

```python
params = {
    "REFERENCE": "TRUE HEADING",
    "SYSTEM_DRAFT": "0.500",
    "PLATFORM_TYPE": "SURFACE_SHIP",
}
G.write_processing_parameters(params, param_time=1724650073.719285)
```

`params` is just `{name: value}` -- both strings. `param_time` is a POSIX
epoch timestamp (float), a `datetime`, or an ISO8601 string; all three are
accepted everywhere a "time" parameter appears in this API (see
`_gsf_epoch()` in `gsfu.py` if you want the details). `write_sensor_parameters()`
has the identical signature, for `GSF_RECORD_SENSOR_PARAMETERS`.

### Sound velocity profile

```python
G.write_sound_velocity_profile(
    observation_time=1724650073.719285,
    application_time=1724650073.719285,
    latitude_deg=45.925417,
    longitude_deg=-129.981855,
    depth_m=[0.0, 5.0, 10.0, 50.0, 100.0],
    sound_speed_mPerSec=[1500.1, 1499.8, 1498.2, 1495.0, 1490.3],
)
```

`depth_m` and `sound_speed_mPerSec` are parallel arrays, one entry per cast
sample, and must be the same length.

### Attitude

```python
G.write_attitude(
    attitude_time=[1724650073.70, 1724650073.75, 1724650073.80],
    pitch_deg=[0.12, 0.10, 0.09],
    roll_deg=[-0.5, -0.4, -0.3],
    heave_m=[0.02, 0.01, 0.00],
    heading_deg=[210.5, 210.6, 210.6],
)
```

One `GSF_RECORD_ATTITUDE` record can carry many time-tagged samples -- all
five arrays must be the same length. In practice you'd batch attitude
samples (e.g. one record per second, or per some fixed sample count) rather
than writing one record per sample.

### Swath bathymetry ping

This is the record most worth walking through carefully, since it has the
most moving parts: fixed scalar fields, a table of per-beam arrays, and an
optional vendor-specific block.

```python
scalars = {
    "PingTime": 1724650073.719285,
    "Longitude_deg": -129.981855,
    "Latitude_deg": 45.925417,
    "NumberBeams": 3,
    "CenterBeam": 1,
    "Heading_deg": 210.5,
    "Pitch_deg": 0.10,
    "Roll_deg": -0.4,
    "Heave_m": 0.01,
    "Height_m": -32.1,   # antenna height above the ellipsoid, if known
}

beams = {
    "Depth_m":         [7.342, 7.356, 7.370],
    "AcrossTrack_m":   [-3.89, 0.0, 3.89],
    "AlongTrack_m":    [0.0, 0.0, 0.0],
    "TravelTime_s":    [0.01033, 0.01033, 0.01032],
    "BeamAngle_deg":   [-45.0, 0.0, 45.0],
    "QualityFactor":   [50, 52, 49],
}

G.write_swath_bathymetry_ping(scalars, beams)
```

Only `PingTime`, `Longitude_deg`, `Latitude_deg`, and `NumberBeams` are
required in `scalars`; everything else (`CenterBeam`, `PingFlags`,
`TideCorrector_m`, `DepthCorrector_m`, `Heading_deg`, `Pitch_deg`,
`Roll_deg`, `Heave_m`, `Course_deg`, `Speed_kn`, `Height_m`, `SEP_m`,
`GPSTideCorrector_m`) defaults to `0`/`0.0` if you leave it out.

Every array in `beams` must have exactly `NumberBeams` entries. The
dict *key* is what selects which subrecord gets written and, in turn, its
scale factor -- `_beam_array_subrecord_id()` resolves each label (e.g.
`"Depth_m"`, `"BeamAngle_deg"`) to its `GSF_SWATH_BATHY_SUBRECORD_*` id.
Any key `write_swath_bathymetry_ping()` doesn't recognize raises
`KeyError` -- see `_PING_ARRAY_SUBRECORDS` in `gsfu.py` for the full list of
recognized labels.

To add the KMALL (Kongsberg SIS 5) vendor-specific subrecord:

```python
kmall_specific = {
    "EchoSounderID": 712,
    "PingRate_Hz": 1.2,
    # ... see _decode_kmall_specific()'s docstring in gsfu.py for the
    # full field list; anything omitted defaults to 0/0.0.
}
tx_sectors = [
    {"TxSectorNumb": 0, "CenterFreq_Hz": 70000.0, "TiltAngleReTx_deg": 0.0},
    {"TxSectorNumb": 1, "CenterFreq_Hz": 71000.0, "TiltAngleReTx_deg": 15.0},
]

G.write_swath_bathymetry_ping(
    scalars, beams, kmall_specific=kmall_specific, tx_sectors=tx_sectors)
```

`kmall_specific` is a flat dict of scalar fields (matching the `KMALL.*`
names `-p` prints, minus the `KMALL.` prefix); `tx_sectors` is a list of
per-sector dicts, one per transmit sector (up to `GSF_MAX_KMALL_SECTORS` =
9). Both are optional -- omit them entirely for a non-KMALL system, or if
you don't need the vendor-specific block.

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

G.write_swath_bathymetry_ping(scalars, beams, scale_factors=my_scale_factors)
```

Each entry is `subrecordID: (multiplier, offset, field_width_bytes,
signed)`. `write_swath_bathymetry_ping()` raises `ValueError` if a scaled
value doesn't fit its field width (e.g. a value too large for the
multiplier/width you chose) -- that's your signal the scale factor needs
adjusting, not the data.

## Closing the file

```python
G.closeFile()
```

## Putting it together

```python
from GSFU.gsfu import gsf

G = gsf("my_survey.gsf")
G.write_header()
G.write_processing_parameters({"REFERENCE": "TRUE HEADING"}, param_time=t0)
G.write_sound_velocity_profile(
    observation_time=t0, application_time=t0,
    latitude_deg=lat, longitude_deg=lon,
    depth_m=svp_depths, sound_speed_mPerSec=svp_speeds)

for ping_time, lat, lon, beams in my_pings:
    G.write_attitude(...)  # or batch these separately, see above
    G.write_swath_bathymetry_ping(
        {"PingTime": ping_time, "Latitude_deg": lat, "Longitude_deg": lon,
         "NumberBeams": len(beams["Depth_m"])},
        beams)

G.closeFile()
```

## Re-encoding a record you decoded

Because `write_*`'s dict shapes match what the `_decode_*` functions
return, you can round-trip a record through `gsfu.py`'s own reader:

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
scalars, tables, notes = _decode_swath_bathymetry_ping(payload, major_version=3, scale_factors={})

dst = gsf("copy.gsf")
dst.write_header()
dst.write_swath_bathymetry_ping(scalars, tables["Beams"])
dst.closeFile()
```

This is exactly the strategy used to generate the round-trip compatibility
test files in this project's test suite.
