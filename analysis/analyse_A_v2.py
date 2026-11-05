"""ANALYSE A (v2, relue) — references lineaires, R2 par mesure, n*, et controle de n=20.

Repond aux points 1, 2 et au nouveau centre de gravite de la relecture.

  A1  Par mesure, sur la SOURCE seule : lambda, sigma_x, sigma_e, R2.
  A2  REFERENCE LINEAIRE EMPIRIQUE : la regression lineaire ajustee sur la
      source est evaluee sur le test cible avec NOTRE metrique (Eq. 4-5).
      C'est exact, cela ne suppose rien de gaussien, et cela remplace la
      forme close R_min = sigma_e / sqrt(lambda^2 sigma_x^2 + sigma_e^2),
      qui n'est qu'une reecriture de sqrt(1 - R2) et que le modele gele
      franchit pour le QTc (0,8698 contre 0,8760).
  A3  n* = k (1 - R2) / R2, en transitions puis en patients, contre les
      seuils observes du tableau 3.
  A4  R2 contre lambda comme predicteur du seuil, avec IC bootstrap sur les
      correlations de rang.
  A5  CONTROLE DE DEGENERESCENCE a n=20 : si les arbres ne coupent pas, le
      modele "local" est une CONSTANTE, et son inferiorite a la persistance
      vient de l'intercept non retreci, pas de l'adaptation. Ce point touche
      le titre de l'article.

Duree : 5 a 15 min. Aucune donnee n'est ecrite hors de /kaggle/working.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

SMOKE_TEST = False       # True : donnees synthetiques, verifie que la cellule tourne
SORTIE = '/kaggle/working'
GRAINE_SPLIT = 11
GRAINE = 20260926
N_BOOT = 10000

MESURES_7 = ['PR', 'QRS', 'QT', 'QTc', 'P_wave_axis', 'QRS_axis', 'T_wave_axis']
AXES_3 = ['P_wave_axis', 'QRS_axis', 'T_wave_axis']
GAP_MIN, GAP_MAX = 7.0, 365.0

# seuils observes, tableau 3 du manuscrit
SEUILS_OBSERVES = {'QT': 44, 'QTc': 60, 'P_wave_axis': 150, 'PR': 220,
                   'T_wave_axis': 500, 'QRS': 1000, 'QRS_axis': 3000}


# ==========================================================================
# Socle identique aux analyses precedentes : mêmes cohortes, meme decoupage
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


def err_sujet(tab, pred, ech_metrique, mesures, axes):
    parts = {}
    for i, m in enumerate(mesures):
        r = tab[f'd_{m}'].to_numpy() - pred[:, i]
        if m in axes:
            r = enrouler(r)
        parts[m] = pd.Series(np.abs(r) / ech_metrique[m],
                             index=tab.sujet.to_numpy()).groupby(level=0).mean()
    return sum(parts.values()) / len(mesures), parts


def construire_transitions(mesures, axes):
    chemin = str(next(Path('/kaggle/input').rglob('Electrocardiogram*.csv')))
    print(f'construction depuis {chemin}')
    d = pd.read_csv(chemin, parse_dates=['ecgdate'])
    d = d[d.ecgsource == 'M']
    d = d.drop_duplicates(subset=['personid'] + mesures)
    d = d.dropna(subset=mesures)
    d = d.sort_values(['personid', 'ecgdate']).reset_index(drop=True)
    g = d.groupby('personid')
    tt = pd.DataFrame({'sujet': d.personid.astype(str), 'dept': d.ecgdept.to_numpy()})
    tt['dept_prec'] = g['ecgdept'].shift(1).to_numpy()   # [v2] service de l ECG precedent
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
    N_BOOT = 500
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
    assert SRC.sujet.nunique() == 41144, SRC.sujet.nunique()
    assert TE.sujet.nunique() == 5187, TE.sujet.nunique()
    assert len(POOL_IDS) == 12104, len(POOL_IDS)
TRANS_PAR_PATIENT = len(POOL) / len(POOL_IDS)

# [v2] Le manuscrit dit "only within-department transitions retained", mais le
# filtre ci-dessus ne regarde que le service de l ECG d ARRIVEE. On mesure l ecart
# sans rien changer (les asserts d effectifs restent valides).
if 'dept_prec' in EV.columns:
    for nom, tab, dep in [('source', SRC, D_SRC), ('cible', CIB, D_CIB)]:
        croise = float((tab.dept_prec != dep).mean())
        print(f'[v2] {nom} : {100 * croise:.1f} % des transitions partent d un autre service')
    print('[v2] Si > 0 : corriger le code (filtrer dept_prec == dept) OU le manuscrit.')
else:
    print('[v2] dept_prec absent (t reutilise d une session precedente ?) : '
          'relancer construire_transitions pour verifier le point intra-service.')
print(f'source {SRC.sujet.nunique():,} / test {TE.sujet.nunique():,} / '
      f'reservoir {len(POOL_IDS):,} patients')
print(f'transitions par patient dans le reservoir : {TRANS_PAR_PATIENT:.3f}')

CTR_S, ECH_NIV_S = stats_niveau(SRC, MESURES_7, AXES_3)
ECH_SRC = stats_delta(SRC, MESURES_7)
ECH_MET = stats_delta(TE, MESURES_7)
Xsrc = lambda tab: X(tab, CTR_S, ECH_NIV_S, MESURES_7, AXES_3)
Xte = Xsrc(TE)
e_pers, parts_pers = err_sujet(TE, np.zeros((len(TE), 7)), ECH_MET,
                               MESURES_7, AXES_3)


def evaluer(pred):
    e, parts = err_sujet(TE, pred, ECH_MET, MESURES_7, AXES_3)
    out = {'composite': float(e.mean() / e_pers.mean())}
    for m in MESURES_7:
        out[m] = float(parts[m].mean() / parts_pers[m].reindex(parts[m].index).mean())
    return out


# ==========================================================================
# A1 + A2 : statistiques source et reference lineaire empirique
# ==========================================================================
print('\n' + '=' * 78)
print('A1-A2  STATISTIQUES SOURCE ET REFERENCE LINEAIRE EMPIRIQUE')
print('=' * 78)

pred_uni = np.zeros((len(TE), 7))
pred_multi = np.zeros((len(TE), 7))
lignes = []
for i, m in enumerate(MESURES_7):
    x_src = SRC[f'src_{m}'].to_numpy(float)
    d_src = SRC[f'd_{m}'].to_numpy(float)
    if m in AXES_3:
        x_src = enrouler(x_src - moyenne_circulaire(x_src))
        d_src = enrouler(d_src)
    else:
        x_src = x_src - x_src.mean()

    # --- univarie : d ~ niveau propre
    A = np.column_stack([np.ones_like(x_src), x_src])
    coef_u, *_ = np.linalg.lstsq(A, d_src, rcond=None)
    res_u = d_src - A @ coef_u
    lam = -coef_u[1]
    sigma_x = float(x_src.std(ddof=1))
    sigma_e = float(res_u.std(ddof=1))
    var_d = float(d_src.var(ddof=1))
    R2 = 1.0 - res_u.var(ddof=1) / var_d
    R_min_forme_close = sigma_e / np.sqrt(lam ** 2 * sigma_x ** 2 + sigma_e ** 2)

    # application a la cible
    x_te = TE[f'src_{m}'].to_numpy(float)
    if m in AXES_3:
        x_te = enrouler(x_te - moyenne_circulaire(SRC[f'src_{m}'].to_numpy(float)))
    else:
        x_te = x_te - SRC[f'src_{m}'].to_numpy(float).mean()
    pred_uni[:, i] = np.column_stack([np.ones_like(x_te), x_te]) @ coef_u

    # --- multivarie : d ~ les huit entrees standardisees
    A8 = np.column_stack([np.ones(len(SRC)), Xsrc(SRC)])
    coef_m, *_ = np.linalg.lstsq(A8, d_src, rcond=None)
    res_m = d_src - A8 @ coef_m
    R2_multi = 1.0 - res_m.var(ddof=1) / var_d
    pred_multi[:, i] = np.column_stack([np.ones(len(TE)), Xte]) @ coef_m

    lignes.append({'mesure': m, 'lambda': lam, 'sigma_x': sigma_x,
                   'sigma_e': sigma_e, 'R2_uni': R2, 'R2_multi': R2_multi,
                   'R_min_forme_close': R_min_forme_close})

ref_uni = evaluer(pred_uni)
ref_multi = evaluer(pred_multi)
STATS = pd.DataFrame(lignes).set_index('mesure')
STATS['ref_lineaire_uni'] = [ref_uni[m] for m in STATS.index]
STATS['ref_lineaire_multi'] = [ref_multi[m] for m in STATS.index]

print(STATS.round(4).to_string())
print(f'\ncomposite : reference univariee {ref_uni["composite"]:.4f}, '
      f'multivariee {ref_multi["composite"]:.4f}')
print('\nLa colonne R_min_forme_close est ce que l article appelait un plafond.')
print('Comparer a ref_lineaire_uni : si elles different, la forme close est')
print('biaisee par l hypothese gaussienne et il faut publier la seconde.')

# ==========================================================================
# A3 : n* = k (1 - R2) / R2
# ==========================================================================
print('\n' + '=' * 78)
print('A3  n* = k (1 - R2) / R2  CONTRE LES SEUILS OBSERVES')
print('=' * 78)
print('Verifie par simulation : la formule est optimiste d environ 20 %')
print('(rapport observe/predit median 1,21 sur grille discrete).')
print()
for k, etiquette, colonne in [(2, 'univarie (intercept + pente)', 'R2_uni'),
                              (9, 'multivarie (intercept + 8 entrees)', 'R2_multi')]:
    print(f'--- k = {k}, {etiquette}')
    print(f'{"mesure":<14} {"R2":>7} {"n* trans":>9} {"n* pat":>8} '
          f'{"observe":>8} {"rapport":>8}')
    for m in STATS.index:
        R2 = STATS.loc[m, colonne]
        if R2 <= 0:
            print(f'{m:<14} {R2:>7.4f}   R2 <= 0, pas de croisement attendu')
            continue
        n_trans = k * (1 - R2) / R2
        n_pat = n_trans / TRANS_PAR_PATIENT
        obs = SEUILS_OBSERVES[m]
        print(f'{m:<14} {R2:>7.4f} {n_trans:>9.1f} {n_pat:>8.1f} '
              f'{obs:>8} {obs / n_pat:>7.1f}x')
    print()
# [v2] Esperance EXACTE en plan aleatoire gaussien, MCO avec intercept et p
# predicteurs (Stein 1960 ; Browne 1975 ; Breiman & Freedman 1983) :
#   E[EQM prediction] = sigma_e^2 (n+1)(n-2) / (n (n-p-2))
# La persistance a pour EQM var(d) + moyenne(d)^2 = sigma_e^2/(1-R2) + moy^2.
# n* exact = plus petit n qui rend la premiere inferieure a la seconde.
def n_etoile_exact(R2, p, derive2):
    """derive2 = moyenne(d)^2 / var(d). Renvoie n* en transitions."""
    if R2 <= 0 and derive2 <= 0:
        return np.nan
    # EQM persistance / sigma_e^2 = (var(d) + moy^2) / ((1-R2) var(d))
    cible = (1.0 + derive2) / max(1.0 - R2, 1e-12)
    for n in range(p + 3, 10 ** 7):
        if (n + 1) * (n - 2) / (n * (n - p - 2)) < cible:
            return float(n)
    return np.nan

DERIVE2 = {}
for m in MESURES_7:
    dd = SRC[f'd_{m}'].to_numpy(float)
    DERIVE2[m] = float(dd.mean() ** 2 / dd.var(ddof=1))
N_ETOILE = []
print('--- n* EXACT (plan aleatoire, intercept, derive incluse), en patients')
print(f'{"mesure":<14} {"approx k=2":>11} {"exact p=1":>10} {"approx k=9":>11} {"exact p=8":>10} {"obs GBM":>8}')
for m in STATS.index:
    r_u, r_m = STATS.loc[m, 'R2_uni'], STATS.loc[m, 'R2_multi']
    a2 = 2 * (1 - r_u) / r_u / TRANS_PAR_PATIENT if r_u > 0 else np.nan
    a9 = 9 * (1 - r_m) / r_m / TRANS_PAR_PATIENT if r_m > 0 else np.nan
    e1 = n_etoile_exact(r_u, 1, DERIVE2[m]) / TRANS_PAR_PATIENT
    e8 = n_etoile_exact(r_m, 8, DERIVE2[m]) / TRANS_PAR_PATIENT
    N_ETOILE.append({'mesure': m, 'approx_k2': a2, 'exact_p1': e1,
                     'approx_k9': a9, 'exact_p8': e8, 'derive2': DERIVE2[m]})
    print(f'{m:<14} {a2:>11.1f} {e1:>10.1f} {a9:>11.1f} {e8:>10.1f} {SEUILS_OBSERVES[m]:>8}')
N_ETOILE = pd.DataFrame(N_ETOILE).set_index('mesure')
print('Ces n* decrivent un modele LINEAIRE local. Le test valide est contre le')
print('seuil du lineaire local (section A6), pas contre le seuil GBM.')
print()
print('Un rapport a peu pres CONSTANT signifie que la formule predit le seuil')
print('a un facteur d echelle pres, attribuable au surcout du boosting. Un')
print('rapport qui derive avec R2 signifie qu elle ne le predit pas.')

# ==========================================================================
# A4 : R2 contre lambda comme predicteur
# ==========================================================================
print('=' * 78)
print('A4  R2 CONTRE LAMBDA COMME PREDICTEUR DU SEUIL')
print('=' * 78)
seuils = np.array([SEUILS_OBSERVES[m] for m in STATS.index], float)
rng = np.random.default_rng(GRAINE)


def spearman_ic(x, y, n_boot=N_BOOT):
    rho, p = stats.spearmanr(x, y)
    tirages = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 3:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            tirages.append(r)
    lo, hi = np.percentile(tirages, [2.5, 97.5])
    return rho, p, lo, hi


for nom, valeurs in [('lambda', STATS['lambda'].to_numpy()),
                     ('R2 univarie', STATS['R2_uni'].to_numpy()),
                     ('R2 multivarie', STATS['R2_multi'].to_numpy())]:
    rho, p, lo, hi = spearman_ic(valeurs, seuils)
    print(f'  {nom:<16} rho = {rho:+.3f}  p = {p:.4f}  '
          f'IC95 [{lo:+.3f}, {hi:+.3f}]')
# [v2] Avec 7 points, le bootstrap degenere (rho = -1 donne IC [-1, -1]).
# Test exact par permutation : les 7! = 5040 ordres possibles.
from itertools import permutations
def p_permutation_exact(x, y):
    rho0 = abs(stats.spearmanr(x, y)[0])
    ry = stats.rankdata(y)
    rx = stats.rankdata(x)
    n_ext = sum(abs(np.corrcoef(rx, np.array(pp))[0, 1]) >= rho0 - 1e-12
                for pp in permutations(ry))
    return n_ext / 5040.0
print('\n[v2] p exact par permutation (bilateral), a publier a la place de l IC :')
for nom, valeurs in [('lambda', STATS['lambda'].to_numpy()),
                     ('R2 univarie', STATS['R2_uni'].to_numpy()),
                     ('R2 multivarie', STATS['R2_multi'].to_numpy())]:
    print(f'  {nom:<16} p exact = {p_permutation_exact(valeurs, seuils):.4f}')
# [v2] echelle : pente log-log entre n* (k=9) et seuil GBM observe
nst = np.array([9 * (1 - r) / r / TRANS_PAR_PATIENT for r in STATS['R2_multi']])
pente, ordo = np.polyfit(np.log(nst), np.log(seuils), 1)
print(f'[v2] log(seuil GBM) = {ordo:.2f} + {pente:.2f} log(n* k=9) ; '
      'pente 1 = n* predit l echelle a un facteur pres, pente > 1 = non.')
print('\nAvec sept points l IC sera large. Il faut le dire, et augmenter le')
print('nombre de points (RR, duree P, autres paires de services, strates).')

# ==========================================================================
# A5 : controle de degenerescence a n = 20
# ==========================================================================
print('\n' + '=' * 78)
print('A5  DEGENERESCENCE DU MODELE LOCAL AUX PETITS EFFECTIFS')
print('=' * 78)
print('Si les arbres ne coupent pas, "local" est la moyenne du changement')
print('dans la cohorte, et son inferiorite vient de l intercept, pas de')
print('l adaptation. Ce point touche le titre.')
print()


def gbm():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_depth=4, early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=15, random_state=0)


# modele source, necessaire au bras augmente
print('entrainement du modele source (une fois)...')
Xs = Xsrc(SRC)
MODELE_SRC = {}
for m in MESURES_7:
    y = SRC[f'd_{m}'].to_numpy() / ECH_SRC[m]
    MODELE_SRC[m] = gbm().fit(Xs, y)
pred_src_te = np.column_stack([MODELE_SRC[m].predict(Xte) * ECH_SRC[m]
                               for m in MESURES_7])
gele = evaluer(pred_src_te)
print(f'  gele composite {gele["composite"]:.4f}'
      + ('' if SMOKE_TEST else '   (doit valoir 0,9321 a 0,002 pres)'))

rng_deg = np.random.default_rng(GRAINE)
print()
print(f'{"n":>6} {"transitions":>11} {"local constant":>16} '
      f'{"augmente constant":>18} {"aug == local":>13} {"rError local":>13}')
lignes_deg = []
for n in ([20, 60] if SMOKE_TEST else [20, 30, 44, 60, 100, 220, 500]):
    const_loc = const_aug = identiques = 0
    n_trans, rerrs, rerr_moy, rerr_aug = [], [], [], []
    nb = 3 if SMOKE_TEST else 10
    for r in range(nb):
        sous = set(rng_deg.choice(POOL_IDS, size=n, replace=False))
        tr = POOL[POOL.sujet.isin(sous)]
        if len(tr) < 5:
            continue
        n_trans.append(len(tr))
        ctr_l, ech_l = stats_niveau(tr, MESURES_7, AXES_3)
        ech_dl = stats_delta(tr, MESURES_7)
        Xtr_l = X(tr, ctr_l, ech_l, MESURES_7, AXES_3)
        Xte_l = X(TE, ctr_l, ech_l, MESURES_7, AXES_3)
        # prediction source sur la cohorte d entrainement, pour le bras augmente
        Xtr_s = Xsrc(tr)
        p_loc = np.zeros((len(TE), 7))
        p_aug = np.zeros((len(TE), 7))
        for i, m in enumerate(MESURES_7):
            s = max(ech_dl[m], 1e-9)
            y = tr[f'd_{m}'].to_numpy() / s
            p_loc[:, i] = gbm().fit(Xtr_l, y).predict(Xte_l) * s
            aug_tr = np.column_stack(
                [Xtr_l, MODELE_SRC[m].predict(Xtr_s) * ECH_SRC[m] / s])
            aug_te = np.column_stack(
                [Xte_l, MODELE_SRC[m].predict(Xte) * ECH_SRC[m] / s])
            p_aug[:, i] = gbm().fit(aug_tr, y).predict(aug_te) * s
        const_loc += int(np.allclose(p_loc.std(axis=0), 0, atol=1e-10))
        const_aug += int(np.allclose(p_aug.std(axis=0), 0, atol=1e-10))
        identiques += int(np.allclose(p_loc, p_aug, atol=1e-10))
        rerrs.append(evaluer(p_loc)['composite'])
        # [v2] predicteur "moyenne du changement de la cohorte" explicite :
        # s il egale le local, l inferiorite a n petit est celle de l intercept.
        p_moy = np.tile([tr[f'd_{m}'].mean() for m in MESURES_7], (len(TE), 1))
        rerr_moy.append(evaluer(p_moy)['composite'])
        rerr_aug.append(evaluer(p_aug)['composite'])
    lignes_deg.append({'n': n, 'transitions_moy': float(np.mean(n_trans)),
                       'local_constant': const_loc, 'augmente_constant': const_aug,
                       'identiques': identiques, 'rError_local': float(np.mean(rerrs)),
                       'rError_moyenne_cohorte': float(np.mean(rerr_moy)),
                       'rError_augmente': float(np.mean(rerr_aug))})
    print(f'{n:>6} {np.mean(n_trans):>11.0f} {const_loc:>12}/{nb} '
          f'{const_aug:>14}/{nb} {identiques:>9}/{nb} {np.mean(rerrs):>13.4f}'
          f'   moyenne-cohorte {np.mean(rerr_moy):.4f}  augmente {np.mean(rerr_aug):.4f}')

pd.DataFrame(lignes_deg).to_csv(f'{SORTIE}/A_degenerescence.csv', index=False)
print()
print('Si "local constant" vaut 10/10 a n=20, alors a cet effectif le modele')
print('local EST la moyenne du changement de la cohorte, et le titre ne peut')
print('pas reposer sur lui. Le resultat honnete a petit n est la')
print('recalibration, 0,9578 contre 0,9321.')
print()

# ==========================================================================
# [v2] A6 : TETE-A-TETE lineaire gele contre GBM gele, IC bootstrap apparies
# ==========================================================================
print('=' * 78)
print('A6  LINEAIRE SOURCE GELE CONTRE GBM SOURCE GELE (IC apparies par patient)')
print('=' * 78)
print('Si le lineaire gagne sur le composite, le modele recommande change.')


def ic_apparie(pred_a, pred_b, n_boot=2000, graine=GRAINE):
    """rError(a) - rError(b), IC95 par bootstrap des patients du test.
    Negatif = a meilleur que b."""
    ea, pa = err_sujet(TE, pred_a, ECH_MET, MESURES_7, AXES_3)
    eb, pb = err_sujet(TE, pred_b, ECH_MET, MESURES_7, AXES_3)
    series = {'composite': (ea, eb, e_pers)}
    for m in MESURES_7:
        series[m] = (pa[m], pb[m], parts_pers[m])
    r = np.random.default_rng(graine)
    n = len(e_pers)
    idx = r.integers(0, n, (n_boot, n))
    out = {}
    for nom, (a, b, p) in series.items():
        a, b, p = (np.asarray(v.sort_index(), float) for v in (a, b, p))
        diff = (a.mean() - b.mean()) / p.mean()
        boots = (a[idx].mean(1) - b[idx].mean(1)) / p[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[nom] = (diff, lo, hi)
    return out


lignes_h2h = []
for nom_a, pred_a in [('lineaire uni', pred_uni), ('lineaire multi', pred_multi)]:
    res = ic_apparie(pred_a, pred_src_te, n_boot=300 if SMOKE_TEST else 2000)
    print(f'\n  {nom_a} - GBM gele (negatif = lineaire meilleur)')
    for cle, (dlt, lo, hi) in res.items():
        verdict = ('LINEAIRE meilleur' if hi < 0 else
                   'GBM meilleur' if lo > 0 else 'indistinguables')
        print(f'    {cle:<14} {dlt:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  {verdict}')
        lignes_h2h.append({'comparaison': f'{nom_a} - GBM gele', 'sortie': cle,
                           'diff': dlt, 'ic_bas': lo, 'ic_haut': hi, 'verdict': verdict})
pd.DataFrame(lignes_h2h).to_csv(f'{SORTIE}/A_tete_a_tete.csv', index=False)

# decomposition de l ecart forme close / reference empirique
print('\n  Forme close (source, EQM) contre reference empirique (cible, MAE) :')
print('  on separe le TRANSPORT source->cible et le passage EQM->MAE.')
rmse_ratio = {}
for i, m in enumerate(MESURES_7):
    r = TE[f'd_{m}'].to_numpy() - pred_uni[:, i]
    d0 = TE[f'd_{m}'].to_numpy()
    if m in AXES_3:
        r, d0 = enrouler(r), enrouler(d0)
    rmse_ratio[m] = float(np.sqrt(np.mean(r ** 2) / np.mean(d0 ** 2)))
print(f'  {"mesure":<14} {"forme close":>12} {"EQM cible":>10} {"MAE cible":>10}')
for m in MESURES_7:
    print(f'  {m:<14} {STATS.loc[m, "R_min_forme_close"]:>12.4f} '
          f'{rmse_ratio[m]:>10.4f} {STATS.loc[m, "ref_lineaire_uni"]:>10.4f}')
print('  colonne 1 -> 2 : effet du transport ; colonne 2 -> 3 : effet EQM -> MAE')
print('  (+ agregation par patient).')

# ==========================================================================
# [v2] A7 : courbe d apprentissage LINEAIRE locale -> test direct de n*
# ==========================================================================
print('\n' + '=' * 78)
print('A7  COURBE LINEAIRE LOCALE : SEUILS OBSERVES CONTRE n*')
print('=' * 78)
print('La formule n* decrit un lineaire local ; c est ici qu elle se teste.')
print('Le rapport seuil GBM / seuil lineaire est le "cout de la flexibilite".')


def ajuster_lineaire_local(tr, variante):
    """Renvoie une matrice de prediction (len(TE), 7) en unites physiques."""
    P = np.zeros((len(TE), 7))
    if variante == 'multi':
        ctr_l, ech_l = stats_niveau(tr, MESURES_7, AXES_3)
        Xtr_l = np.column_stack([np.ones(len(tr)), X(tr, ctr_l, ech_l, MESURES_7, AXES_3)])
        Xte_l = np.column_stack([np.ones(len(TE)), X(TE, ctr_l, ech_l, MESURES_7, AXES_3)])
    for i, m in enumerate(MESURES_7):
        d_tr = tr[f'd_{m}'].to_numpy(float)
        if variante == 'multi':
            coef, *_ = np.linalg.lstsq(Xtr_l, d_tr, rcond=None)
            P[:, i] = Xte_l @ coef
            continue
        x_tr = tr[f'src_{m}'].to_numpy(float)
        x_te = TE[f'src_{m}'].to_numpy(float)
        if m in AXES_3:
            c = moyenne_circulaire(x_tr)
            x_tr, x_te = enrouler(x_tr - c), enrouler(x_te - c)
        else:
            c = x_tr.mean()
            x_tr, x_te = x_tr - c, x_te - c
        if variante == 'uni':            # intercept + pente
            coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x_tr), x_tr]), d_tr, rcond=None)
            P[:, i] = coef[0] + coef[1] * x_te
        else:                            # 'uni_sans_intercept' : pente seule
            pente = float(x_tr @ d_tr / max(x_tr @ x_tr, 1e-12))
            P[:, i] = pente * x_te
    return P


GRILLE_LIN = ([10, 20, 44, 100] if SMOKE_TEST else
              [10, 15, 20, 30, 44, 60, 100, 150, 220, 300, 500, 1000, 3000])
REPS_LIN = 3 if SMOKE_TEST else 20
rng_lin = np.random.default_rng(GRAINE + 7)
courbes = []
for n in GRILLE_LIN:
    for r in range(REPS_LIN):
        tr = POOL[POOL.sujet.isin(set(rng_lin.choice(POOL_IDS, size=n, replace=False)))]
        for variante in ['uni', 'uni_sans_intercept', 'multi']:
            res = evaluer(ajuster_lineaire_local(tr, variante))
            courbes.append({'n': n, 'rep': r, 'variante': variante, **res})
COURBES = pd.DataFrame(courbes)
COURBES.to_csv(f'{SORTIE}/A_courbes_lineaires.csv', index=False)
moy = COURBES.groupby(['variante', 'n']).mean(numeric_only=True).drop(columns='rep')


def seuil_durable(serie):
    """Plus petit n de la grille a partir duquel la courbe reste sous 1."""
    s = serie.sort_index()
    sous = s < 1.0
    for n in s.index:
        if sous.loc[n:].all():
            return float(n)
    return np.nan


lignes_seuil = []
for m in MESURES_7:
    ligne = {'mesure': m, 'obs_GBM': SEUILS_OBSERVES[m]}
    for variante in ['uni', 'uni_sans_intercept', 'multi']:
        ligne[f'seuil_{variante}'] = seuil_durable(moy.loc[variante, m])
    ligne.update(N_ETOILE.loc[m, ['exact_p1', 'exact_p8']].to_dict())
    lignes_seuil.append(ligne)
SEUILS_LIN = pd.DataFrame(lignes_seuil).set_index('mesure')
SEUILS_LIN['cout_flexibilite'] = SEUILS_LIN.obs_GBM / SEUILS_LIN.seuil_uni
print('\nComposite moyen par variante (lignes) et effectif (colonnes) :')
print(moy['composite'].unstack('n').round(4).to_string())
print('\nSeuils durables (patients), grille discrete :')
print(SEUILS_LIN.round(1).to_string())
print('\nLecture : exact_p1 doit suivre seuil_uni, exact_p8 seuil_multi.')
print('Si uni_sans_intercept croise bien plus tot que uni, l intercept est le')
print('cout principal aux petits effectifs -> meme diagnostic que A5.')
SEUILS_LIN.to_csv(f'{SORTIE}/A_seuils_lineaires.csv')
N_ETOILE.to_csv(f'{SORTIE}/A_n_etoile.csv')

STATS.to_csv(f'{SORTIE}/A_statistiques_source.csv')
with open(f'{SORTIE}/A_resume.json', 'w') as f:
    json.dump({'ref_lineaire_uni': ref_uni, 'ref_lineaire_multi': ref_multi,
               'transitions_par_patient': TRANS_PAR_PATIENT}, f, indent=2)
print(f'\necrit : {SORTIE}/A_statistiques_source.csv et A_resume.json')
print(f'duree : {(time.time() - t0) / 60:.1f} min')