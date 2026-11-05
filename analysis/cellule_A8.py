# ==========================================================================
# A8 (a lancer apres analyse_A_v2, meme session) : n* avec un R2 "MAE" SOURCE SEULE
# ==========================================================================
# Constat de A7 : avec le R2 des moindres carres, n* sous-estime meme le seuil
# LINEAIRE (1,7x a 23x). Cause : la metrique est la MAE, et le gain du lineaire
# en MAE est bien plus faible qu en EQM (A6, colonnes 2 -> 3). On remplace donc
# R2 par R2_eff = 1 - (rapport MAE hors-pli)^2, calcule par validation croisee
# groupee sur la SOURCE uniquement, sans rien regarder de la cible.
from sklearn.model_selection import GroupKFold

oof_u = np.zeros((len(SRC), 7))
oof_m = np.zeros((len(SRC), 7))
for tr_i, va_i in GroupKFold(5).split(SRC, groups=SRC.sujet.to_numpy()):
    a, b = SRC.iloc[tr_i], SRC.iloc[va_i]
    Aa = np.column_stack([np.ones(len(a)), Xsrc(a)])
    Ab = np.column_stack([np.ones(len(b)), Xsrc(b)])
    for i, m in enumerate(MESURES_7):
        d = a[f'd_{m}'].to_numpy(float)
        xa, xb = a[f'src_{m}'].to_numpy(float), b[f'src_{m}'].to_numpy(float)
        if m in AXES_3:
            c = moyenne_circulaire(xa)
            xa, xb = enrouler(xa - c), enrouler(xb - c)
        else:
            c = xa.mean()
            xa, xb = xa - c, xb - c
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(xa), xa]), d, rcond=None)
        oof_u[va_i, i] = coef[0] + coef[1] * xb
        coef, *_ = np.linalg.lstsq(Aa, d, rcond=None)
        oof_m[va_i, i] = Ab @ coef

_, parts0 = err_sujet(SRC, np.zeros((len(SRC), 7)), ECH_SRC, MESURES_7, AXES_3)
_, parts_u = err_sujet(SRC, oof_u, ECH_SRC, MESURES_7, AXES_3)
_, parts_m = err_sujet(SRC, oof_m, ECH_SRC, MESURES_7, AXES_3)

lignes_a8 = []
for m in MESURES_7:
    ru = parts_u[m].mean() / parts0[m].mean()
    rm = parts_m[m].mean() / parts0[m].mean()
    r2u, r2m = 1 - ru ** 2, 1 - rm ** 2
    # derive deja incluse dans la MAE de la persistance -> derive2 = 0
    nu = n_etoile_exact(r2u, 1, 0.0) / TRANS_PAR_PATIENT if r2u > 0 else np.nan
    nm = n_etoile_exact(r2m, 8, 0.0) / TRANS_PAR_PATIENT if r2m > 0 else np.nan
    lignes_a8.append({'mesure': m, 'R2_eff_uni': r2u, 'R2_eff_multi': r2m,
                      'n_etoile_uni': nu, 'seuil_uni': SEUILS_LIN.loc[m, 'seuil_uni'],
                      'n_etoile_multi': nm, 'seuil_multi': SEUILS_LIN.loc[m, 'seuil_multi'],
                      'seuil_GBM': SEUILS_OBSERVES[m]})
A8 = pd.DataFrame(lignes_a8).set_index('mesure')
A8['rapport_uni'] = A8.seuil_uni / A8.n_etoile_uni
A8['rapport_multi'] = A8.seuil_multi / A8.n_etoile_multi
print(A8.round(3).to_string())
ok = A8[['R2_eff_multi', 'seuil_GBM']].dropna()
rho, _ = stats.spearmanr(ok.R2_eff_multi, ok.seuil_GBM)
print(f'\nSpearman R2_eff_multi (source) contre seuil GBM : {rho:+.3f}, '
      f'p exact = {p_permutation_exact(ok.R2_eff_multi.to_numpy(), ok.seuil_GBM.to_numpy()):.4f}'
      if len(ok) == 7 else f'\nSpearman sur {len(ok)} points : {rho:+.3f}')
print('Rapports stables (~1-2,5) -> n* est publiable comme prediction a un facteur pres.')
A8.to_csv(f'{SORTIE}/A8_n_etoile_mae.csv')
