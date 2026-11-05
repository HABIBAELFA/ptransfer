"""SUPPLEMENTARY TABLE S9 — erreurs absolues en unites physiques (ms, degres).

Pour chaque mesure, sur le jeu de test cible de l'analyse principale (5 187 patients) :
  - MAE de la persistence ("rien ne change"), moyennee par patient puis sur les patients ;
  - MAE du modele importe gele ;
  - reduction absolue (ms ou degres) avec IC95 bootstrap sur les patients ;
  - reduction relative (1 - rError), qui doit reproduire le tableau 3 du manuscrit.
Duree : environ 2 min. Aucune donnee n'est ecrite hors de /kaggle/working.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold

SMOKE_TEST = False
SORTIE = '/kaggle/working'
GRAINE_SPLIT = 11
GRAINE = 20260926
N_BOOT = 2000

MESURES_7 = ['PR', 'QRS', 'QT', 'QTc', 'P_wave_axis', 'QRS_axis', 'T_wave_axis']
AXES_3 = ['P_wave_axis', 'QRS_axis', 'T_wave_axis']
GAP_MIN, GAP_MAX = 7.0, 365.0

# [v2] grille fine : avec 8 points, les seuils et leurs IC etaient limites par la grille
TAILLES = [20, 30, 44, 60, 100, 150, 220, 300, 500, 700, 1000, 1500, 2000, 3000, 6000, 12000]
REPS = {n: (20 if n <= 300 else 10 if n <= 1000 else 5) for n in TAILLES}
N_BOOT_SEUILS = 1000
GRILLE_RETRECISSEMENT = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]


# ==========================================================================
# Socle : strictement identique au script A
# ==========================================================================
def enrouler(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def moyenne_circulaire(x):
    a = np.deg2rad(np.asarray(x, float))
    return float(np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def sd_circulaire(x):
    a = np.deg2rad(np.asarray(x, float))
    R = min(max(np.hypot(np.sin(a).mean(), np.cos(a).mean()), 1e-12), 1 - 1e-12)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def stats_niveau(tab, mesures, axes):
    ctr, ech = {}, {}
    for m in mesures:
        v = tab[f'src_{m}'].to_numpy(float)
        if m in axes:
            ctr[m], ech[m] = moyenne_circulaire(v), sd_circulaire(v)
        else:
            ctr[m], ech[m] = float(v.mean()), float(v.std(ddof=1))
    return ctr, ech


def stats_delta(tab, mesures):
    return {m: float(tab[f'd_{m}'].std(ddof=1)) for m in mesures}


def X(tab, ctr, ech, mesures, axes):
    cols = []
    for m in mesures:
        v = tab[f'src_{m}'].to_numpy(float) - ctr[m]
        if m in axes:
            v = enrouler(v)
        cols.append(v / ech[m])
    cols.append(tab.log_gap.to_numpy())
    return np.column_stack(cols)


def construire_transitions(mesures, axes):
    chemin = str(next(Path('/kaggle/input').rglob('Electrocardiogram*.csv')))
    d = pd.read_csv(chemin, parse_dates=['ecgdate'])
    d = d[d.ecgsource == 'M']
    d = d.drop_duplicates(subset=['personid'] + mesures).dropna(subset=mesures)
    d = d.sort_values(['personid', 'ecgdate']).reset_index(drop=True)
    g = d.groupby('personid')
    tt = pd.DataFrame({'sujet': d.personid.astype(str), 'dept': d.ecgdept.to_numpy()})
    tt['gap_j'] = g['ecgdate'].diff().dt.total_seconds() / 86400.0
    for m in mesures:
        tt[f'src_{m}'] = g[m].shift(1)
        tt[f'tgt_{m}'] = d[m].to_numpy()
    besoin = [c for c in tt.columns if c.startswith(('src_', 'tgt_'))] + ['gap_j']
    tt = tt.dropna(subset=besoin)
    tt = tt[(tt.gap_j >= GAP_MIN) & (tt.gap_j <= GAP_MAX)].reset_index(drop=True)
    for m in mesures:
        delta = tt[f'tgt_{m}'] - tt[f'src_{m}']
        tt[f'd_{m}'] = enrouler(delta) if m in axes else delta
    tt['log_gap'] = np.log1p(tt.gap_j)
    return tt


def donnees_synthetiques(rng):
    lignes = []
    for dept, n_pat in [('O', 2500), ('I', 1400), ('E', 300)]:
        for k in range(n_pat):
            for _ in range(1 + rng.poisson(0.6)):
                lignes.append({'sujet': f'{dept}{k}', 'dept': dept})
    tt = pd.DataFrame(lignes)
    n = len(tt)
    tt['gap_j'] = rng.uniform(GAP_MIN, GAP_MAX, n)
    tt['log_gap'] = np.log1p(tt.gap_j)
    lam = {'PR': .29, 'QRS': .14, 'QT': .36, 'QTc': .42,
           'P_wave_axis': .46, 'QRS_axis': .11, 'T_wave_axis': .22}
    for m, mu in {'PR': 160, 'QRS': 95, 'QT': 400, 'QTc': 440,
                  'P_wave_axis': 50, 'QRS_axis': 30, 'T_wave_axis': 40}.items():
        x = mu + 25 * rng.standard_normal(n)
        tt[f'src_{m}'] = x
        tt[f'd_{m}'] = -lam[m] * (x - mu) + 25 * rng.standard_normal(n)
        tt[f'tgt_{m}'] = tt[f'src_{m}'] + tt[f'd_{m}']
    return tt


t0 = time.time()
if SMOKE_TEST:
    print('*** SMOKE TEST : donnees synthetiques, chiffres sans valeur ***')
    t = donnees_synthetiques(np.random.default_rng(0))
    TAILLES = [20, 100]
    REPS = {20: 3, 100: 3}
    N_BOOT = 200
    N_BOOT_SEUILS = 100
elif not (isinstance(globals().get('t'), pd.DataFrame) and len(globals()['t']) > 130000):
    t = construire_transitions(MESURES_7, AXES_3)

EV = t.copy()
EV['sujet'] = EV.sujet.astype(str)
compte = EV.groupby('dept').sujet.nunique().sort_values(ascending=False)
D_SRC, D_CIB = compte.index[0], compte.index[1]
ids_src = set(EV.loc[EV.dept == D_SRC, 'sujet'])
ids_cib = set(EV.loc[EV.dept == D_CIB, 'sujet']) - ids_src
SRC = EV[EV.sujet.isin(ids_src) & (EV.dept == D_SRC)].reset_index(drop=True)
CIB = EV[EV.sujet.isin(ids_cib) & (EV.dept == D_CIB)].reset_index(drop=True)

rng_split = np.random.default_rng(GRAINE_SPLIT)
pat = np.array(sorted(CIB.sujet.unique()))
rng_split.shuffle(pat)
n_te = int(0.30 * len(pat))
TEST_IDS, POOL_IDS = set(pat[:n_te]), pat[n_te:]
TE = CIB[CIB.sujet.isin(TEST_IDS)].reset_index(drop=True)
POOL = CIB[~CIB.sujet.isin(TEST_IDS)].reset_index(drop=True)
if not SMOKE_TEST:
    assert SRC.sujet.nunique() == 41144 and TE.sujet.nunique() == 5187
TAILLES = [n for n in TAILLES if n <= len(POOL_IDS)]

CTR_S, ECH_NIV_S = stats_niveau(SRC, MESURES_7, AXES_3)
ECH_SRC = stats_delta(SRC, MESURES_7)
ECH_MET = stats_delta(TE, MESURES_7)
Xsrc = lambda tab: X(tab, CTR_S, ECH_NIV_S, MESURES_7, AXES_3)
Xte = Xsrc(TE)
SUJETS_TE = TE.sujet.to_numpy()
PATIENTS_TE = np.array(sorted(TE.sujet.unique()))


def gbm():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=15, random_state=0)



UNITE = {m: ('deg' if m in AXES_3 else 'ms') for m in MESURES_7}
print('entrainement du modele source...')
Xs = Xsrc(SRC)
MOD = {m: gbm().fit(Xs, SRC[f'd_{m}'].to_numpy() / ECH_SRC[m]) for m in MESURES_7}
lignes = []
rng = np.random.default_rng(GRAINE)
for m in MESURES_7:
    d = TE[f'd_{m}'].to_numpy(float)
    p = MOD[m].predict(Xte) * ECH_SRC[m]
    e0 = np.abs(enrouler(d) if m in AXES_3 else d)
    e1 = np.abs(enrouler(d - p) if m in AXES_3 else d - p)
    E0 = pd.Series(e0, index=TE.sujet.to_numpy()).groupby(level=0).mean()
    E1 = pd.Series(e1, index=TE.sujet.to_numpy()).groupby(level=0).mean().reindex(E0.index)
    a0, a1 = E0.to_numpy(), E1.to_numpy()
    idx = rng.integers(0, len(a0), (2000, len(a0)))
    red = a0[idx].mean(1) - a1[idx].mean(1)
    lignes.append({'mesure': m, 'unite': UNITE[m], 'MAE_persistence': a0.mean(), 'MAE_gele': a1.mean(),
                   'reduction': a0.mean() - a1.mean(), 'ic_bas': np.percentile(red, 2.5),
                   'ic_haut': np.percentile(red, 97.5), 'rError': a1.mean() / a0.mean()})
T = pd.DataFrame(lignes).set_index('mesure')
print(T.round(4).to_string())
T.to_csv(f'{SORTIE}/S9_erreurs_absolues.csv')
print('\nLa colonne rError doit valoir 0,8938 (QT), 0,8698 (QTc)... comme au tableau 3.')
print(f'duree : {(time.time() - t0) / 60:.1f} min')
