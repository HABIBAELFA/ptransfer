"""Share of transitions that begin in another department (Section 2.3 of the paper).
Run on Kaggle with ECG-ViEW II attached. Expected: source 41,144 patients, target 17,291;
34.6% of source and 37.8% of target transitions begin in another department."""
import pandas as pd
from pathlib import Path

M = ['PR', 'QRS', 'QT', 'QTc', 'P_wave_axis', 'QRS_axis', 'T_wave_axis']
d = pd.read_csv(str(next(Path('/kaggle/input').rglob('Electrocardiogram*.csv'))), parse_dates=['ecgdate'])
d = d[d.ecgsource == 'M'].drop_duplicates(subset=['personid'] + M).dropna(subset=M)
d = d.sort_values(['personid', 'ecgdate']).reset_index(drop=True)
g = d.groupby('personid')
tt = pd.DataFrame({'sujet': d.personid.astype(str), 'dept': d.ecgdept, 'dept_prec': g['ecgdept'].shift(1),
                   'gap': g['ecgdate'].diff().dt.total_seconds() / 86400})
tt = tt[tt.gap.between(7, 365)]
compte = tt.groupby('dept').sujet.nunique().sort_values(ascending=False)
D_SRC, D_CIB = compte.index[0], compte.index[1]
ids_src = set(tt.loc[tt.dept == D_SRC, 'sujet'])
ids_cib = set(tt.loc[tt.dept == D_CIB, 'sujet']) - ids_src
SRC = tt[tt.sujet.isin(ids_src) & (tt.dept == D_SRC)]
CIB = tt[tt.sujet.isin(ids_cib) & (tt.dept == D_CIB)]
print(f'source patients {SRC.sujet.nunique():,}; target patients {CIB.sujet.nunique():,}')
for nom, tab, dep in [('source', SRC, D_SRC), ('target', CIB, D_CIB)]:
    print(f'{nom}: {100 * (tab.dept_prec != dep).mean():.1f}% of transitions begin in another department')
