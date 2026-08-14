import csv
import numpy as np
from pathlib import Path
from pandas.io.stata import StataReader

RAND_DTA = Path('/mnt/data/randtmp/randhrs1992_2022v1.dta')
MASTER = Path('/mnt/data/HRS_participant_level_cohorts_updated_with_social_connections.csv')
OUT = Path('/mnt/data/HRS_cesd7_validation_audit.csv')

COHORTS = {
    'A': {'T1': 8, 'T2': 10},
    'B': {'T1': 9, 'T2': 11},
}
NEG = ['depres', 'effort', 'sleepr', 'fsad', 'going']
POS = ['whappy', 'enlife']

reader = StataReader(RAND_DTA, convert_categoricals=False, columns=['hhidpn'])
reader.read(nrows=1)
mm = np.memmap(RAND_DTA, mode='r', dtype=reader._dtype,
               offset=reader._data_location, shape=(reader._nobs,))
var_index = {name: i for i, name in enumerate(reader._varlist)}

def arr(name):
    return mm[f's{var_index[name]}']

rand_ids = arr('hhidpn').astype(np.int64)
row_by_id = {int(hhidpn): i for i, hhidpn in enumerate(rand_ids)}

def cesd7(row_idx, wave):
    vals = {name: int(arr(f'r{wave}{name}')[row_idx]) for name in NEG + POS}
    if not all(v in (0, 1) for v in vals.values()):
        return None
    return int(sum(vals[n] for n in NEG) + sum(1 - vals[n] for n in POS))

def cesd8_valid(row_idx, wave):
    v = int(arr(f'r{wave}cesd')[row_idx])
    return 0 <= v <= 8

def valid_loneliness_base(row):
    def num(name):
        try:
            return float(row.get(name, ''))
        except Exception:
            return None
    return (
        num('T1 loneliness') is not None
        and num('T2 loneliness') is not None
        and num('self_completed_loneliness_T1') == 1
        and num('self_completed_loneliness_T2') == 1
    )

with MASTER.open(newline='', encoding='utf-8-sig') as f:
    master_rows = list(csv.DictReader(f))

audit = []
def add(section, cohort, timepoint, measure, count, denominator, notes=''):
    audit.append({
        'section': section,
        'cohort': cohort,
        'timepoint': timepoint,
        'measure': measure,
        'count': count,
        'denominator': denominator,
        'percent': round(100 * count / denominator, 1) if denominator else '',
        'notes': notes,
    })

# RAND behavior when components are missing.
for wave in [8, 9, 10, 11, 12]:
    cesd = arr(f'r{wave}cesd')
    cesdm = arr(f'r{wave}cesdm')
    for missing_n in range(0, 9):
        mask = cesdm == missing_n
        n = int(mask.sum())
        if n == 0:
            continue
        n_numeric = int((mask & (cesd >= 0) & (cesd <= 8)).sum())
        add(
            'rand_cesd_missingness_behavior', 'All RAND', f'Wave {wave}',
            f'RwCESD numeric when RwCESDM={missing_n}', n_numeric, n,
            'RAND can retain a numeric CES-D score when some components are missing; RwCESDM records the number missing.'
        )

# Availability by cohort and sample definition.
for cohort, timing in COHORTS.items():
    cohort_rows = [r for r in master_rows if r['cohort'] == cohort]
    loneliness_rows = [r for r in cohort_rows if valid_loneliness_base(r)]

    for sample_name, rows in [
        ('full_master', cohort_rows),
        ('valid_self_completed_loneliness_T1_T2', loneliness_rows),
    ]:
        denom = len(rows)
        for tp in ['T1', 'T2']:
            wave = timing[tp]
            n8 = n7 = 0
            for r in rows:
                try:
                    hid = int(float(r['hhidpn']))
                except Exception:
                    continue
                j = row_by_id.get(hid)
                if j is None:
                    continue
                n8 += int(cesd8_valid(j, wave))
                n7 += int(cesd7(j, wave) is not None)

            add(f'availability_{sample_name}', cohort, tp, 'standard_RAND_CESD8', n8, denom,
                'Standard 8-item RAND CES-D; includes felt-lonely item.')
            add(f'availability_{sample_name}', cohort, tp, 'CESD7_excluding_loneliness_complete_items', n7, denom,
                'Preferred project depression measure; requires all 7 non-loneliness components.')

        w1, w2 = timing['T1'], timing['T2']
        both8 = both7 = 0
        for r in rows:
            try:
                hid = int(float(r['hhidpn']))
            except Exception:
                continue
            j = row_by_id.get(hid)
            if j is None:
                continue
            both8 += int(cesd8_valid(j, w1) and cesd8_valid(j, w2))
            both7 += int(cesd7(j, w1) is not None and cesd7(j, w2) is not None)

        add(f'availability_{sample_name}', cohort, 'T1_and_T2', 'standard_RAND_CESD8', both8, denom)
        add(f'availability_{sample_name}', cohort, 'T1_and_T2', 'CESD7_excluding_loneliness_complete_items', both7, denom,
            'Preferred project depression measure.')

# Basic CESD-7 distribution in full master.
for cohort, timing in COHORTS.items():
    rows = [r for r in master_rows if r['cohort'] == cohort]
    for tp in ['T1', 'T2']:
        wave = timing[tp]
        scores = []
        for r in rows:
            try:
                hid = int(float(r['hhidpn']))
            except Exception:
                continue
            j = row_by_id.get(hid)
            if j is None:
                continue
            s = cesd7(j, wave)
            if s is not None:
                scores.append(s)
        if scores:
            add('cesd7_distribution', cohort, tp, 'CESD7_valid_N', len(scores), len(rows))
            audit.append({
                'section': 'cesd7_distribution', 'cohort': cohort, 'timepoint': tp,
                'measure': 'CESD7_mean', 'count': round(float(np.mean(scores)), 3),
                'denominator': len(scores), 'percent': '',
                'notes': 'Range 0-7; higher = more depressive symptoms.'
            })
            audit.append({
                'section': 'cesd7_distribution', 'cohort': cohort, 'timepoint': tp,
                'measure': 'CESD7_SD', 'count': round(float(np.std(scores, ddof=1)), 3),
                'denominator': len(scores), 'percent': '',
                'notes': ''
            })

with OUT.open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['section','cohort','timepoint','measure','count','denominator','percent','notes'])
    writer.writeheader()
    writer.writerows(audit)

print(f'Wrote {OUT}')
