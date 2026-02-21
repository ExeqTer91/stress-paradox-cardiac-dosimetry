# Stress Paradox: Cardiac RF Dosimetry

Time-resolved cardiac RF dosimetry from 24h Holter ECG recordings.
Transforms static 3D FDTD simulation results into temporally-varying cardiac SAR profiles
based on cardiovascular state (heart rate as autonomic proxy).

## Structure

```
holter_temporal/          # Temporal dosimetry analysis
  temporal_dosimetry.py   # Main script (Phases 1-4)
  temporal_dosimetry_data.csv    # 4,582 five-minute windows, 18 subjects
  temporal_dosimetry_results.json # Summary statistics
figures/                  # Publication figures
  fig_temporal_dosimetry.png     # 4-panel: HR, SAR, states, variation
  fig_daily_exposure_profile.png # Hour-by-hour SAR x building
```

## Quick Start

```bash
pip install -r requirements.txt
cd holter_temporal
python temporal_dosimetry.py all
```

## Key Results (N=18 healthy subjects, nsrdb)

| Metric | Value |
|--------|-------|
| State distribution | 12.0% vasodilation, 52.9% rest, 35.1% vasoconstriction |
| Population SAR factor | 1.0315 (+3.15% above standard testing) |
| Individual range | 0.919 to 1.104 |
| Max continuous vasoconstriction | 670 min (11.2h) |
| Modern building 24h SAR | 0.723 (vs 1.0 standard) |
| Traditional building 24h SAR | 0.157 (4.6x lower than modern) |
| Worst period SAR | 1.165 (outdoor cold commute) |

## Method

Heart rate is used as a direct, uncontroversial physiological proxy for autonomic state:
- HR < 60 bpm -> vasodilation-dominant -> SAR factor 0.781 (from 3D FDTD)
- HR 60-80 bpm -> resting -> SAR factor 1.000
- HR > 80 bpm -> vasoconstriction-dominant -> SAR factor 1.165 (from 3D FDTD)

No LF/HF frequency-domain assumptions. Just HR -> state -> SAR factor from validated 3D FDTD.

## Data Source

MIT-BIH Normal Sinus Rhythm Database (nsrdb): N=18, 24h Holter, healthy subjects
https://physionet.org/content/nsrdb/1.0.0/

## License

MIT
