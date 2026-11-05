"""ANALYSE B (v2, relue) — intervalles de confiance et bras manquants.

Repond aux points 3, 4 et 5 de la relecture.

  B1  IC BOOTSTRAP apparies sur les patients du test, pour chaque strategie
      a chaque effectif, plus l ecart-type entre repetitions. La difference
      gele - local a 12 000 (0,9321 contre 0,9336, soit 0,0015 sur trois
      tirages) est testee directement : c est la revendication centrale de
      l article et elle n a jamais ete accompagnee d une incertitude.
  B2  IC SUR LES SEUILS, par bootstrap des patients du test.
  B3  BRAS MANQUANTS :
      - boosting residuel depuis la source (l affirmation "un ensemble
        d arbres ne peut pas repartir a chaud" est fausse ; scikit-learn a
        warm_start, XGBoost init_model, LightGBM init_score, et le boosting
        residuel en est l equivalent exact et portable) ;
      - local a prediction retrecie vers zero, facteur choisi par validation
        croisee DANS la cohorte d entrainement, jamais sur le test ;
      - recalibration retrecie vers 1, meme procedure ;
      - bras poole source + cohorte cible.

Le retrecissement est le point cle : si le local retreci egale la
persistance a n=20 au lieu de faire pire, alors l inferiorite du local a
petit effectif vient d un intercept non retreci, pas de l adaptation.

Duree : environ 1 h avec la grille fine (v2). Mettre SMOKE_TEST a True d abord.
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


# ==========================================================================
# Erreur par patient : c est l objet que l on bootstrappe
# ==========================================================================
def erreurs_par_patient(pred):
    """Renvoie (n_patients, 7) : erreur absolue moyenne par patient et mesure."""
    sortie = {}
    for i, m in enumerate(MESURES_7):
        r = TE[f'd_{m}'].to_numpy() - pred[:, i]
        if m in AXES_3:
            r = enrouler(r)
        sortie[m] = pd.Series(np.abs(r) / ECH_MET[m],
                              index=SUJETS_TE).groupby(level=0).mean()
    return pd.DataFrame(sortie).reindex(PATIENTS_TE).to_numpy()


E_PERS = erreurs_par_patient(np.zeros((len(TE), 7)))


def rerror_depuis_erreurs(E, idx=None):
    """rError composite et par mesure, sur un sous-echantillon de patients."""
    A = E if idx is None else E[idx]
    B = E_PERS if idx is None else E_PERS[idx]
    composite = A.mean(axis=1).mean() / B.mean(axis=1).mean()
    par_mesure = A.mean(axis=0) / B.mean(axis=0)
    return float(composite), par_mesure


RNG_BOOT = np.random.default_rng(GRAINE)
IDX_BOOT = RNG_BOOT.integers(0, len(PATIENTS_TE), (N_BOOT, len(PATIENTS_TE)))


def ic_bootstrap(E):
    """IC95 du composite, bootstrap apparie sur les patients du test."""
    tirages = np.array([rerror_depuis_erreurs(E, IDX_BOOT[b])[0]
                        for b in range(N_BOOT)])
    return np.percentile(tirages, [2.5, 97.5]), tirages


def ic_difference(E1, E2):
    """IC95 de la difference E1 - E2, apparie sur les memes patients."""
    diffs = np.array([rerror_depuis_erreurs(E1, IDX_BOOT[b])[0]
                      - rerror_depuis_erreurs(E2, IDX_BOOT[b])[0]
                      for b in range(N_BOOT)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi), float((diffs > 0).mean())


# ==========================================================================
# Modele source et strategies
# ==========================================================================
print('entrainement du modele source...')
Xs = Xsrc(SRC)
MODELE_SRC = {m: gbm().fit(Xs, SRC[f'd_{m}'].to_numpy() / ECH_SRC[m])
              for m in MESURES_7}
PRED_SRC_TE = np.column_stack([MODELE_SRC[m].predict(Xte) * ECH_SRC[m]
                               for m in MESURES_7])
E_GELE = erreurs_par_patient(PRED_SRC_TE)
print(f'  gele composite {rerror_depuis_erreurs(E_GELE)[0]:.4f}')


def facteur_retrecissement_vc(Xtr, ytr, pred_base_tr, groupes, n_splits=5):
    """Choisit w dans [0,1] minimisant l erreur absolue en validation croisee
    DANS la cohorte d entrainement. w=0 revient a la persistance (ou a la
    prediction de base), w=1 a la prediction locale non retrecie."""
    # [v2] GroupKFold par patient : avec KFold, les transitions d un meme patient
    # tombaient des deux cotes du pli, ce qui gonfle w (fuite).
    n = len(ytr)
    n_grp = len(np.unique(groupes))
    if n < 10 or n_grp < 3:
        return 0.0
    kf = GroupKFold(n_splits=min(n_splits, n_grp))
    scores = np.zeros(len(GRILLE_RETRECISSEMENT))
    for tr_i, va_i in kf.split(Xtr, groups=groupes):
        if len(tr_i) < 3:
            continue
        mod = gbm().fit(Xtr[tr_i], ytr[tr_i])
        p = mod.predict(Xtr[va_i])
        base = pred_base_tr[va_i]
        for j, w in enumerate(GRILLE_RETRECISSEMENT):
            scores[j] += np.abs(ytr[va_i] - (base + w * (p - base))).sum()
    return GRILLE_RETRECISSEMENT[int(np.argmin(scores))]


def strategies(tr):
    """Renvoie {nom: prediction (n_test, 7)} pour une cohorte cible tiree."""
    ctr_l, ech_l = stats_niveau(tr, MESURES_7, AXES_3)
    ech_dl = stats_delta(tr, MESURES_7)
    Xtr_l = X(tr, ctr_l, ech_l, MESURES_7, AXES_3)
    Xte_l = X(TE, ctr_l, ech_l, MESURES_7, AXES_3)
    Xtr_s = Xsrc(tr)

    groupes = tr.sujet.to_numpy()
    out = {nom: np.zeros((len(TE), 7)) for nom in
           ['local', 'local retreci', 'recalibration', 'recalibration retrecie',
            'recalibration MAE', 'augmente', 'boosting residuel',
            'boosting residuel retreci', 'poole']}
    w_retreci, c_brut, c_retreci, c_mae_all, w_resid = {}, {}, {}, {}, {}

    for i, m in enumerate(MESURES_7):
        s_loc = max(ech_dl[m], 1e-9)
        d_tr = tr[f'd_{m}'].to_numpy()
        y_loc = d_tr / s_loc

        # --- local
        out['local'][:, i] = gbm().fit(Xtr_l, y_loc).predict(Xte_l) * s_loc

        # --- local retreci vers zero (w choisi en validation croisee interne)
        w = facteur_retrecissement_vc(Xtr_l, y_loc, np.zeros(len(y_loc)), groupes)
        w_retreci[m] = w
        out['local retreci'][:, i] = out['local'][:, i] * w

        # --- recalibration : un coefficient scalaire sur la prediction source
        p_src_tr = MODELE_SRC[m].predict(Xtr_s) * ECH_SRC[m]
        p_src_te = PRED_SRC_TE[:, i]
        denom = float((p_src_tr ** 2).sum())
        c = float((p_src_tr * d_tr).sum() / denom) if denom > 0 else 1.0
        c_brut[m] = c
        out['recalibration'][:, i] = c * p_src_te

        # --- recalibration retrecie vers 1 (w en validation croisee interne)
        n = len(d_tr)
        scores = np.zeros(len(GRILLE_RETRECISSEMENT))
        n_grp = len(np.unique(groupes))
        if n >= 10 and n_grp >= 3:
            kf = GroupKFold(n_splits=min(5, n_grp))          # [v2] groupe par patient
            for tr_i, va_i in kf.split(p_src_tr, groups=groupes):
                dd = float((p_src_tr[tr_i] ** 2).sum())
                c_i = float((p_src_tr[tr_i] * d_tr[tr_i]).sum() / dd) if dd > 0 else 1.0
                for j, w2 in enumerate(GRILLE_RETRECISSEMENT):
                    c_w = 1.0 + w2 * (c_i - 1.0)
                    scores[j] += np.abs(d_tr[va_i] - c_w * p_src_tr[va_i]).sum()
            w2 = GRILLE_RETRECISSEMENT[int(np.argmin(scores))]
        else:
            w2 = 0.0
        c_r = 1.0 + w2 * (c - 1.0)
        c_retreci[m] = c_r
        out['recalibration retrecie'][:, i] = c_r * p_src_te

        # --- [v2] recalibration ajustee en erreur ABSOLUE (la metrique d evaluation).
        # min_c sum |d - c p| = mediane ponderee de d/p, poids |p|. A 12 000 la
        # recalibration MCO (0,9331) reste pire que le gele (0,9321) : signe que
        # le c optimal en EQM n est pas celui de la MAE (queues lourdes).
        ok = np.abs(p_src_tr) > 1e-12
        if ok.sum() >= 1:
            q = d_tr[ok] / p_src_tr[ok]
            wq = np.abs(p_src_tr[ok])
            o = np.argsort(q)
            cum = np.cumsum(wq[o])
            c_mae = float(q[o][np.searchsorted(cum, 0.5 * cum[-1])])
        else:
            c_mae = 1.0
        c_mae_all[m] = c_mae
        out['recalibration MAE'][:, i] = c_mae * p_src_te

        # --- augmente : prediction source en entree supplementaire
        aug_tr = np.column_stack([Xtr_l, p_src_tr / s_loc])
        aug_te = np.column_stack([Xte_l, p_src_te / s_loc])
        out['augmente'][:, i] = gbm().fit(aug_tr, y_loc).predict(aug_te) * s_loc

        # --- boosting residuel depuis la source : l analogue du warm start
        resid = (d_tr - p_src_tr) / s_loc
        mod_r = gbm().fit(Xtr_s, resid)
        corr_te = mod_r.predict(Xte) * s_loc
        out['boosting residuel'][:, i] = p_src_te + corr_te
        # [v2] a petit n l arbre ne coupe pas : la correction est la MOYENNE du
        # residu, soit le meme probleme d intercept que le local. On la retrecit
        # vers zero par validation croisee groupee.
        w_r = facteur_retrecissement_vc(Xtr_s, resid, np.zeros(len(resid)), groupes)
        w_resid[m] = w_r
        out['boosting residuel retreci'][:, i] = p_src_te + w_r * corr_te

        # --- poole : source et cohorte cible ensemble
        Xp = np.vstack([Xs, Xtr_s])
        yp = np.concatenate([SRC[f'd_{m}'].to_numpy() / ECH_SRC[m],
                             d_tr / ECH_SRC[m]])
        out['poole'][:, i] = gbm().fit(Xp, yp).predict(Xte) * ECH_SRC[m]

    return out, {'w_local': w_retreci, 'c_mco': c_brut, 'c_retreci': c_retreci,
                 'c_mae': c_mae_all, 'w_residuel': w_resid}


# ==========================================================================
# B1 + B3 : boucle principale
# ==========================================================================
print('\n' + '=' * 78)
print('B1 + B3  STRATEGIES, IC BOOTSTRAP ET BRAS MANQUANTS')
print('=' * 78)

rng = np.random.default_rng(GRAINE)
lignes, erreurs_par_taille = [], {}
ERREURS_REPS, journal_params = {}, []   # [v2]
for n in TAILLES:
    t1 = time.time()
    accum, w_all = {}, []
    for r in range(REPS[n]):
        sous = set(rng.choice(POOL_IDS, size=n, replace=False))
        tr = POOL[POOL.sujet.isin(sous)]
        if len(tr) < 5:
            continue
        preds, params = strategies(tr)
        w_all.append(np.mean(list(params['w_local'].values())))
        for cle, dico in params.items():
            for m, v in dico.items():
                journal_params.append({'n': n, 'rep': r, 'parametre': cle,
                                       'mesure': m, 'valeur': v})
        for nom, p in preds.items():
            E = erreurs_par_patient(p)
            accum.setdefault(nom, []).append(E)
            comp, _ = rerror_depuis_erreurs(E)
            lignes.append({'strategie': nom, 'n': n, 'repetition': r,
                           'rError': comp})
    # erreurs moyennes sur les repetitions : base des IC
    erreurs_par_taille[n] = {nom: np.mean(v, axis=0) for nom, v in accum.items()}
    ERREURS_REPS[n] = {nom: np.stack(v).astype(np.float32) for nom, v in accum.items()}
    erreurs_par_taille[n]['gele'] = E_GELE
    print(f'  n={n:>6} termine en {(time.time() - t1) / 60:.1f} min '
          f'(retrecissement local moyen w={np.mean(w_all):.2f})')

RES = pd.DataFrame(lignes)
RES.to_csv(f'{SORTIE}/B_repetitions.csv', index=False)
pd.DataFrame(journal_params).to_csv(f'{SORTIE}/B_parametres.csv', index=False)

# ==========================================================================
# [v2] BOOTSTRAP A DEUX NIVEAUX : tirages d entrainement ET patients du test
# ==========================================================================
# La v1 moyennait les erreurs sur les repetitions puis bootstrappait les seuls
# patients du test : l IC ignorait la variabilite du tirage d entrainement, qui
# domine a petit n (local, n=20 : ET entre repetitions 0,032, soit ES 0,007,
# contre une demi-largeur d IC de 0,001). Ici on reechantillonne les deux.
# Estimande : la performance ATTENDUE d une strategie quand un centre tire n
# patients. NB : a 6 000 et 12 000 le reservoir (12 104) est presque epuise,
# les tirages se recouvrent et la variabilite d entrainement est sous-estimee.
P = len(PATIENTS_TE)
rng2 = np.random.default_rng(GRAINE + 1)
W_P = np.stack([np.bincount(IDX_BOOT[b], minlength=P) for b in range(N_BOOT)]).astype(np.float64)
DEN = {'composite': (W_P @ E_PERS.mean(axis=1)) / P}
for j, m in enumerate(MESURES_7):
    DEN[m] = (W_P @ E_PERS[:, j]) / P
W_R = {n: None for n in TAILLES}


def poids_reps(n, R):
    if W_R[n] is None or W_R[n].shape[1] != R:
        W_R[n] = np.stack([np.bincount(rng2.integers(0, R, R), minlength=R)
                           for _ in range(N_BOOT)]).astype(np.float64)
    return W_R[n]


def tirages_rerror(E_reps, cible, n=None):
    """E_reps : (R, P, 7). Renvoie N_BOOT valeurs de rError attendue.
    Les memes poids patients (W_P) et, a n donne, les memes poids de repetitions
    servent a toutes les strategies : les differences sont appariees."""
    C = E_reps.mean(axis=2) if cible == 'composite' else E_reps[:, :, MESURES_7.index(cible)]
    C = C.astype(np.float64)
    R = C.shape[0]
    num_rb = C @ W_P.T / P                       # (R, N_BOOT)
    if R == 1:
        num = num_rb[0]
    else:
        Wr = poids_reps(n, R)                    # (N_BOOT, R)
        num = (Wr * num_rb.T).sum(axis=1) / R
    return num / DEN[cible]


E_GELE_R = E_GELE[None].astype(np.float32)
T_GELE = tirages_rerror(E_GELE_R, 'composite')

print('\nrError attendue, IC95 bootstrap a deux niveaux (tirages + patients) :')
print(f'{"strategie":<26} {"n":>6} {"rError":>8} {"ET rep":>8} {"IC95":>20}'
      f' {"strategie - gele":>18} {"IC95 difference":>22}  verdict')
resume = []
for n in TAILLES:
    for nom in ERREURS_REPS[n]:
        Er = ERREURS_REPS[n][nom]
        comp = rerror_depuis_erreurs(Er.mean(axis=0).astype(np.float64))[0]
        t = tirages_rerror(Er, 'composite', n)
        lo, hi = np.percentile(t, [2.5, 97.5])
        dd = t - T_GELE
        dlo, dhi = np.percentile(dd, [2.5, 97.5])
        verdict = ('gele meilleur' if dlo > 0 else 'BRAS meilleur' if dhi < 0
                   else 'indistinguable')
        sd = RES[(RES.n == n) & (RES.strategie == nom)].rError.std()
        resume.append({'strategie': nom, 'n': n, 'rError': comp, 'et_repetitions': sd,
                       'ic_bas': lo, 'ic_haut': hi, 'diff_gele': comp - T_GELE.mean(),
                       'diff_ic_bas': dlo, 'diff_ic_haut': dhi, 'verdict': verdict})
        print(f'{nom:<26} {n:>6} {comp:>8.4f} {sd:>8.4f}  [{lo:.4f}, {hi:.4f}]'
              f' {comp - rerror_depuis_erreurs(E_GELE)[0]:>+18.4f}'
              f'  [{dlo:+.4f}, {dhi:+.4f}]  {verdict}')
pd.DataFrame(resume).to_csv(f'{SORTIE}/B_resume_ic.csv', index=False)
print(f'gele : {rerror_depuis_erreurs(E_GELE)[0]:.4f}  '
      f'IC95 [{np.percentile(T_GELE, 2.5):.4f}, {np.percentile(T_GELE, 97.5):.4f}]')

# ==========================================================================
# [v2] B2 : seuils durables, IC a deux niveaux, grille fine
# ==========================================================================
print('\n' + '=' * 78)
print('B2  SEUILS DURABLES (IC a deux niveaux), local et local retreci')
print('=' * 78)


def seuil_durable(courbe, tailles, seuil=1.0):
    sous = courbe < seuil
    for i in range(len(courbe)):
        if sous[i:].all():
            return tailles[i]
    return np.nan


lignes_seuils = []
for bras in ['local', 'local retreci']:
    for cible in ['composite'] + MESURES_7:
        courbes = np.column_stack([tirages_rerror(ERREURS_REPS[n][bras], cible, n)
                                   for n in TAILLES])[:N_BOOT_SEUILS]
        point = seuil_durable(np.array([rerror_depuis_erreurs(
            ERREURS_REPS[n][bras].mean(axis=0).astype(np.float64))[0]
            if cible == 'composite' else
            rerror_depuis_erreurs(ERREURS_REPS[n][bras].mean(axis=0).astype(np.float64))[1][MESURES_7.index(cible)]
            for n in TAILLES]), TAILLES)
        sb = np.array([seuil_durable(c, TAILLES) for c in courbes], float)
        jamais = float(np.isnan(sb).mean())
        lo, hi = (np.nanpercentile(sb, [2.5, 97.5], method='nearest') if np.isfinite(sb).sum() > 10
                  else (np.nan, np.nan))
        lignes_seuils.append({'bras': bras, 'sortie': cible, 'seuil': point,
                              'ic_bas': lo, 'ic_haut': hi, 'jamais': jamais})
        print(f'  {bras:<14} {cible:<12} seuil {point:>7.0f}  IC95 [{lo:.0f}, {hi:.0f}]'
              f'  sans croisement {jamais:.0%}')
pd.DataFrame(lignes_seuils).to_csv(f'{SORTIE}/B_seuils.csv', index=False)
print('\nGrille :', TAILLES)
print('Ces seuils remplacent le tableau 3 ; refaire la correlation avec R2 (script A)')
print('sur la colonne "local", et rapporter aussi celle de "local retreci".')
print(f'\nduree totale : {(time.time() - t0) / 60:.1f} min')
