#!/usr/bin/env python3
"""
Temporal Cardiac Dosimetry from 24h Holter ECG
Transform static FDTD results into time-resolved cardiac RF dosimetry.
Uses HR as direct physiological proxy for autonomic state → SAR factor.
"""
import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
                     'figure.dpi': 300, 'savefig.dpi': 300})

OUTPUT_DIR = 'temporal_dosimetry_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAR_FACTORS = {
    'vasodilation': 0.781,
    'rest': 1.000,
    'vasoconstriction': 1.165,
}

def log(msg):
    print(f"  {msg}", flush=True)


def classify_state(hr):
    if hr < 60:
        return 'vasodilation'
    elif hr > 80:
        return 'vasoconstriction'
    else:
        return 'rest'


def extract_hr_timeseries_from_rr(rr, timestamps, subject_id, window_sec=300):
    valid = (rr > 0.3) & (rr < 2.0)
    rr = rr[valid]
    timestamps = timestamps[valid]

    total_duration = timestamps[-1] - timestamps[0]
    n_windows = int(total_duration / window_sec)

    results = []
    t0 = timestamps[0]
    for i in range(n_windows):
        t_start = t0 + i * window_sec
        t_end = t0 + (i + 1) * window_sec

        mask = (timestamps >= t_start) & (timestamps < t_end)
        if mask.sum() < 10:
            continue

        window_rr = rr[mask]
        mean_hr = 60.0 / np.mean(window_rr)
        sdnn = np.std(window_rr) * 1000
        rmssd = np.sqrt(np.mean(np.diff(window_rr)**2)) * 1000 if len(window_rr) > 1 else 0

        state = classify_state(mean_hr)
        sar_factor = SAR_FACTORS[state]

        results.append({
            'subject': subject_id,
            'time_hours': ((t_start + t_end) / 2 - t0) / 3600,
            'mean_hr': mean_hr,
            'sdnn_ms': sdnn,
            'rmssd_ms': rmssd,
            'state': state,
            'sar_factor': sar_factor,
            'n_beats': int(mask.sum()),
        })

    return pd.DataFrame(results)


def extract_hr_timeseries_wfdb(record_path, subject_id, window_sec=300):
    import wfdb
    try:
        ann = wfdb.rdann(record_path, 'atr')
    except Exception:
        ann = wfdb.rdann(record_path, 'ecg')

    normal_mask = np.isin(ann.symbol, ['N', '.'])
    r_peaks = ann.sample[normal_mask]
    fs = ann.fs

    rr = np.diff(r_peaks) / fs
    rr_times = r_peaks[1:] / fs

    return extract_hr_timeseries_from_rr(rr, rr_times, subject_id, window_sec)


def phase1_extract():
    log("=== PHASE 1: Extract HR Time-Series ===")

    pkl_path = 'ecg_data/nsrdb.pkl'
    if os.path.exists(pkl_path):
        log("Using cached nsrdb RR intervals")
        with open(pkl_path, 'rb') as f:
            records = pickle.load(f)

        all_data = []
        for rec in records:
            subject_id = rec['subject_id']
            rr = rec['rr']
            timestamps = rec['timestamps']
            df = extract_hr_timeseries_from_rr(rr, timestamps, subject_id)
            if len(df) > 20:
                all_data.append(df)
                log(f"  {subject_id}: {len(df)} windows, mean HR = {df['mean_hr'].mean():.1f}")
            else:
                log(f"  {subject_id}: too few windows ({len(df)}), skipping")

        log(f"Total: {len(all_data)} subjects from nsrdb")
    else:
        log("Downloading nsrdb via wfdb...")
        import wfdb
        data_dir = Path('data/nsrdb')
        data_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database('nsrdb', str(data_dir))
        record_list = wfdb.get_record_list('nsrdb')

        all_data = []
        for rec_name in record_list:
            try:
                df = extract_hr_timeseries_wfdb(str(data_dir / rec_name), rec_name)
                if len(df) > 20:
                    all_data.append(df)
                    log(f"  {rec_name}: {len(df)} windows, mean HR = {df['mean_hr'].mean():.1f}")
            except Exception as e:
                log(f"  {rec_name}: FAILED ({e})")

        log(f"Total: {len(all_data)} subjects from nsrdb")

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, 'temporal_dosimetry_data.csv'), index=False)
    log(f"Saved temporal_dosimetry_data.csv ({len(combined)} windows)")
    return all_data


def phase2_dosimetry(all_data):
    log("=== PHASE 2: Population-Level Temporal Dosimetry ===")

    combined = pd.concat(all_data, ignore_index=True)
    n_subjects = len(all_data)

    print("=" * 60, flush=True)
    print(f"TEMPORAL CARDIAC DOSIMETRY (N = {n_subjects} subjects)", flush=True)
    print("=" * 60, flush=True)

    state_counts = combined['state'].value_counts(normalize=True)
    log("\n1. CARDIOVASCULAR STATE DISTRIBUTION (all 5-min windows):")
    for state in ['vasodilation', 'rest', 'vasoconstriction']:
        pct = state_counts.get(state, 0) * 100
        sar = SAR_FACTORS[state]
        log(f"   {state:20s}: {pct:5.1f}%  (SAR factor = {sar:.3f})")

    pop_sar = sum(state_counts.get(s, 0) * SAR_FACTORS[s] for s in SAR_FACTORS)
    log(f"\n   Population-weighted SAR factor: {pop_sar:.4f}")
    log(f"   Deviation from standard testing: {(pop_sar - 1) * 100:+.2f}%")

    cumulative_ratios = []
    for df in all_data:
        if len(df) < 20:
            continue
        cumulative_ratios.append(df['sar_factor'].mean())

    cum_arr = np.array(cumulative_ratios)
    log(f"\n2. 24h CUMULATIVE SAR RATIO (vs REST-only assumption):")
    log(f"   Mean:   {cum_arr.mean():.4f} ({(cum_arr.mean()-1)*100:+.2f}%)")
    log(f"   Median: {np.median(cum_arr):.4f}")
    log(f"   Min:    {cum_arr.min():.4f} (most vasodilated subject)")
    log(f"   Max:    {cum_arr.max():.4f} (most vasoconstricted subject)")
    log(f"   Range:  {cum_arr.max() - cum_arr.min():.4f}")
    log(f"   SD:     {cum_arr.std():.4f}")

    vasoconst_pct_per_subject = []
    for df in all_data:
        if len(df) < 20:
            continue
        pct = (df['state'] == 'vasoconstriction').mean() * 100
        vasoconst_pct_per_subject.append(pct)

    vc_arr = np.array(vasoconst_pct_per_subject)
    log(f"\n3. TIME IN VASOCONSTRICTION (worst-case SAR) PER SUBJECT:")
    log(f"   Mean:   {vc_arr.mean():.1f}%")
    log(f"   Median: {np.median(vc_arr):.1f}%")
    log(f"   Max:    {vc_arr.max():.1f}% (most stressed subject)")
    log(f"   >20%:   {(vc_arr > 20).sum()} / {len(vc_arr)} subjects")
    log(f"   >30%:   {(vc_arr > 30).sum()} / {len(vc_arr)} subjects")

    max_consecutive = []
    for df in all_data:
        if len(df) < 20:
            continue
        states = (df['state'] == 'vasoconstriction').values
        max_run = 0
        current_run = 0
        for s in states:
            if s:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        max_consecutive.append(max_run * 5)

    mc_arr = np.array(max_consecutive)
    log(f"\n4. LONGEST CONTINUOUS VASOCONSTRICTION EPISODE:")
    log(f"   Mean:   {mc_arr.mean():.0f} min")
    log(f"   Median: {np.median(mc_arr):.0f} min")
    log(f"   Max:    {mc_arr.max():.0f} min ({mc_arr.max()/60:.1f} hours)")

    results = {
        'n_subjects': n_subjects,
        'n_windows': len(combined),
        'state_distribution': {k: round(v * 100, 2) for k, v in state_counts.to_dict().items()},
        'population_sar_factor': round(float(pop_sar), 4),
        'cumulative_sar_mean': round(float(cum_arr.mean()), 4),
        'cumulative_sar_median': round(float(np.median(cum_arr)), 4),
        'cumulative_sar_min': round(float(cum_arr.min()), 4),
        'cumulative_sar_max': round(float(cum_arr.max()), 4),
        'cumulative_sar_sd': round(float(cum_arr.std()), 4),
        'vasoconstriction_pct_mean': round(float(vc_arr.mean()), 1),
        'vasoconstriction_pct_max': round(float(vc_arr.max()), 1),
        'max_continuous_vasoconst_min': float(mc_arr.max()),
        'max_continuous_vasoconst_mean_min': round(float(mc_arr.mean()), 1),
    }

    return results


def phase3_exposure_profile():
    log("=== PHASE 3: Building × State Interaction (Daily Exposure Profile) ===")

    atten_outdoor = 0
    atten_modern = 1.2
    atten_traditional = 15

    def db_to_sar_factor(db):
        return 10 ** (-db / 10)

    schedule = [
        (0, 7, 'indoor_modern', 'vasodilation', 'Sleep'),
        (7, 8, 'outdoor', 'vasoconstriction', 'Commute (cold)'),
        (8, 12, 'indoor_modern', 'rest', 'Office AM'),
        (12, 13, 'outdoor', 'rest', 'Lunch outdoor'),
        (13, 16, 'indoor_modern', 'rest', 'Office PM'),
        (16, 18, 'indoor_modern', 'vasoconstriction', 'Office stress'),
        (18, 19, 'outdoor', 'rest', 'Commute eve'),
        (19, 23, 'indoor_modern', 'vasodilation', 'Home relax'),
        (23, 24, 'indoor_modern', 'vasodilation', 'Falling asleep'),
    ]

    atten_map = {
        'outdoor': atten_outdoor,
        'indoor_modern': atten_modern,
        'indoor_traditional': atten_traditional,
    }

    print("\n" + "=" * 70, flush=True)
    print("DAILY EXPOSURE PROFILE: Urban Office Worker, Modern Building", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Period':20s} {'Hours':>5s} {'Location':>18s} {'CV State':>16s} "
          f"{'Bldg dB':>7s} {'SAR_state':>9s} {'SAR_total':>9s}", flush=True)
    print("-" * 70, flush=True)

    total_sar_weighted_modern = 0
    total_sar_weighted_traditional = 0
    total_hours = 0
    schedule_data = []

    for h_start, h_end, location, state, label in schedule:
        hours = h_end - h_start
        bldg_db = atten_map[location]
        bldg_sar = db_to_sar_factor(bldg_db)
        state_sar = SAR_FACTORS[state]
        total_sar = bldg_sar * state_sar

        if location == 'indoor_modern':
            trad_bldg_sar = db_to_sar_factor(atten_traditional)
        else:
            trad_bldg_sar = db_to_sar_factor(atten_outdoor)
        trad_total = trad_bldg_sar * state_sar

        total_sar_weighted_modern += total_sar * hours
        total_sar_weighted_traditional += trad_total * hours
        total_hours += hours

        schedule_data.append({
            'h_start': h_start, 'h_end': h_end, 'label': label,
            'location': location, 'state': state,
            'bldg_db': bldg_db, 'state_sar': state_sar,
            'total_sar_modern': total_sar, 'total_sar_traditional': trad_total,
        })

        period = f"{h_start:02d}:00-{h_end:02d}:00"
        print(f"{period:20s} {hours:5.0f} {location:>18s} {state:>16s} "
              f"{bldg_db:7.1f} {state_sar:9.3f} {total_sar:9.3f}", flush=True)

    avg_modern = total_sar_weighted_modern / total_hours
    avg_traditional = total_sar_weighted_traditional / total_hours

    print("-" * 70, flush=True)
    log(f"\n24h time-weighted average SAR factor:")
    log(f"  Standard testing (REST, 0 dB):     1.000")
    log(f"  Modern building scenario:           {avg_modern:.4f}")
    log(f"  Traditional building scenario:      {avg_traditional:.4f}")
    log(f"  Modern / Standard ratio:            {avg_modern:.4f}")
    log(f"  Traditional / Standard ratio:       {avg_traditional:.4f}")
    log(f"  Modern / Traditional ratio:         {avg_modern/avg_traditional:.2f}")

    worst_sar = max(s['total_sar_modern'] for s in schedule_data)
    log(f"\n  Worst single period SAR factor:     {worst_sar:.3f}")

    profile_results = {
        'avg_sar_modern': round(float(avg_modern), 4),
        'avg_sar_traditional': round(float(avg_traditional), 4),
        'ratio_modern_standard': round(float(avg_modern), 4),
        'ratio_modern_traditional': round(float(avg_modern / avg_traditional), 2),
        'worst_period_sar': round(float(worst_sar), 3),
        'schedule': schedule_data,
    }

    return profile_results


def phase4_figures(all_data, dosimetry_results, profile_results):
    log("=== PHASE 4: Figures ===")

    colors = {'vasodilation': '#3498DB', 'rest': '#2ECC71', 'vasoconstriction': '#E74C3C'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    example = all_data[0]
    ax.plot(example['time_hours'], example['mean_hr'], 'k-', linewidth=0.5, alpha=0.7)
    for state, color in colors.items():
        mask = example['state'] == state
        if mask.any():
            ax.scatter(example.loc[mask, 'time_hours'], example.loc[mask, 'mean_hr'],
                       c=color, s=8, alpha=0.6, label=state, zorder=3)
    ax.axhline(60, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(80, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Heart Rate (bpm)')
    subj_id = example['subject'].iloc[0]
    ax.set_title(f'A. Example 24h heart rate (Subject {subj_id})')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0, 24)

    ax = axes[0, 1]
    ax.fill_between(example['time_hours'], example['sar_factor'],
                    alpha=0.3, color='#E74C3C')
    ax.plot(example['time_hours'], example['sar_factor'], 'k-', linewidth=0.8)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Standard testing (REST)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cardiac SAR factor (vs REST)')
    ax.set_title('B. Implied cardiac SAR modulation over 24h')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 24)
    ax.set_ylim(0.7, 1.25)

    ax = axes[1, 0]
    combined = pd.concat(all_data, ignore_index=True)
    state_pcts = combined['state'].value_counts(normalize=True) * 100
    bar_labels = ['Vasodilation\n(SAR\u00d70.78)', 'Rest\n(SAR\u00d71.00)', 'Vasoconstriction\n(SAR\u00d71.17)']
    bar_vals = [state_pcts.get('vasodilation', 0), state_pcts.get('rest', 0),
                state_pcts.get('vasoconstriction', 0)]
    bar_colors = [colors['vasodilation'], colors['rest'], colors['vasoconstriction']]
    bars = ax.bar(bar_labels, bar_vals, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('% of 5-min windows')
    ax.set_title('C. Population cardiovascular state distribution')
    for bar, pct in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')

    ax = axes[1, 1]
    cum_sars = [df['sar_factor'].mean() for df in all_data if len(df) > 20]
    ax.hist(cum_sars, bins=15, color='#8E44AD', alpha=0.7, edgecolor='white')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Standard testing')
    ax.axvline(np.mean(cum_sars), color='black', linestyle='-', linewidth=2,
               label=f'Population mean = {np.mean(cum_sars):.3f}')
    ax.set_xlabel('24h cumulative SAR factor')
    ax.set_ylabel('Number of subjects')
    ax.set_title('D. Individual variation in 24h cardiac RF exposure')
    ax.legend(fontsize=8)

    n_subj = len(all_data)
    plt.suptitle(f'Temporal Cardiac Dosimetry from 24h Holter ECG (N={n_subj})', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_temporal_dosimetry.png'), bbox_inches='tight')
    plt.close()
    log("Saved fig_temporal_dosimetry.png")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    schedule = profile_results.get('schedule', [])
    if schedule:
        for seg in schedule:
            h_start, h_end = seg['h_start'], seg['h_end']
            sar = seg['total_sar_modern']
            label = seg['label']
            color = '#E74C3C' if sar > 1.05 else '#2ECC71' if sar < 0.85 else '#F39C12'
            ax.fill_between([h_start, h_end], [sar, sar], alpha=0.4, color=color)
            ax.plot([h_start, h_end], [sar, sar], color=color, linewidth=2)
            ax.text((h_start + h_end)/2, sar + 0.02, label, ha='center', fontsize=7)

    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5,
               label='Standard SAR testing')
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Effective cardiac SAR factor\n(state \u00d7 building)')
    ax.set_title('Daily Cardiac RF Exposure Profile: Urban Office Worker')
    ax.set_xlim(0, 24)
    ax.set_ylim(0.5, 1.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_daily_exposure_profile.png'), bbox_inches='tight')
    plt.close()
    log("Saved fig_daily_exposure_profile.png")

    log("\nAll output files:")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, fn))
        log(f"  {fn} ({sz:,} bytes)")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'

    csv_path = os.path.join(OUTPUT_DIR, 'temporal_dosimetry_data.csv')

    if phase in ('extract', 'all', '1'):
        all_data = phase1_extract()
    else:
        if os.path.exists(csv_path):
            combined = pd.read_csv(csv_path)
            all_data = [g for _, g in combined.groupby('subject')]
            log(f"Loaded {len(all_data)} subjects from CSV")
        else:
            log("ERROR: No data found. Run 'extract' phase first.")
            sys.exit(1)

    if phase in ('dosimetry', 'all', '2'):
        dosimetry_results = phase2_dosimetry(all_data)
    else:
        dosimetry_results = {}

    if phase in ('profile', 'all', '3'):
        profile_results = phase3_exposure_profile()
    else:
        profile_results = {}

    if phase in ('figures', 'all', '4'):
        phase4_figures(all_data, dosimetry_results, profile_results)

    if phase == 'all' or phase in ('dosimetry', 'profile'):
        all_results = {**dosimetry_results, **profile_results}
        if 'schedule' in all_results:
            del all_results['schedule']
        with open(os.path.join(OUTPUT_DIR, 'temporal_dosimetry_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        log("Saved temporal_dosimetry_results.json")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
