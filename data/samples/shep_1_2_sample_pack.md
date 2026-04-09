# SHEP 1.2 Sample Pack

I made this sample pack to test the promoted `SHEP 1.2` Kepler detector with three different situations:

- a confirmed positive target
- a hard false-positive target
- a quiet control target

All three CSV files came from real cached Kepler mission light-curve files in [data/cache](C:\Users\belvo\OneDrive\Desktop\Research Project\data\cache). For each sample:

- I used `PDCSAP_FLUX`
- I kept only rows with `SAP_QUALITY == 0`
- I sorted the combined rows by time
- I normalized the flux by the sample median so the app graph is easier to read

The uploaded CSV format is still just:

```text
time,flux
```

The `time` values are still Kepler mission times in `BJD - 2454833`.

## Included Files

### 1. `shep_1_2_kepler10_positive.csv`
- Target: `Kepler-10`
- Type: confirmed positive system
- Rows: `5267`
- Time span: about `120.54` to `258.47`
- Why I picked it:
  - it is a clean, famous confirmed Kepler system
  - it is easier to cross-check online than some of the multi-planet systems
  - it is a good sanity test for whether `SHEP 1.2` reacts strongly to a real positive target
- Best online cross-check:
  - NASA Exoplanet Archive overview: [Kepler-10 b](https://exoplanetarchive.ipac.caltech.edu/overview/Kepler-10%20b)
- Useful reference values from the archive:
  - `Kepler-10 b` period: about `0.83749` days
  - `Kepler-10 c` period: about `45.29422` days

### 2. `shep_1_2_kic10621666_false_positive.csv`
- Target: `KIC 10621666`
- Type: hard KOI false-positive host
- Rows: `26130`
- Time span: about `131.51` to `905.93`
- Why I picked it:
  - this is the kind of target that can look transit-like and still fool the detector
  - it is a good stress test for false-positive rejection
  - it helps compare whether the model is really improving or just flagging anything dip-like
- KOI rows tied to this target in the Stage 3 catalog:
  - `K01636.01` - `FALSE POSITIVE`
  - `K01636.02` - `FALSE POSITIVE`
- Best online cross-check:
  - NASA Exoplanet Archive KOI query for this KepID: [KIC 10621666 KOI rows](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration+from+q1_q17_dr25_koi+where+kepid%3D10621666&format=csv)

### 3. `shep_1_2_kic6129514_quiet_control.csv`
- Target: `KIC 6129514`
- Type: quiet control target
- Rows: `28892`
- Time span: about `131.51` to `1000.27`
- Why I picked it:
  - this is a good negative/control sample for checking whether the model stays calm on a non-KOI target
  - it helps test whether the app flags noise when it should not
- Stage 3 control list:
  - this target is in [kepler_stage3_control_targets.txt](C:\Users\belvo\OneDrive\Desktop\Research Project\data\kepler_stage3_control_targets.txt)
- Best online cross-check:
  - NASA Exoplanet Archive KOI query for this KepID: [KIC 6129514 KOI rows](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration+from+q1_q17_dr25_koi+where+kepid%3D6129514&format=csv)
  - MAST Kepler archive home: [Kepler at MAST](https://archive.stsci.edu/kepler/)

## How I Would Use These

If I were testing `SHEP 1.2`, I would upload them in this order:

1. `shep_1_2_kepler10_positive.csv`
2. `shep_1_2_kic10621666_false_positive.csv`
3. `shep_1_2_kic6129514_quiet_control.csv`

That gives a quick feel for:

- how strongly the model reacts to a known positive
- whether it overreacts to a known false positive
- whether it stays conservative on a quiet control
