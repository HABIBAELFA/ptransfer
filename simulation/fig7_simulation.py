import glob
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, MUTED, GRID = '#000000', '#333333', '#666666', '#e3e3e3'
COLS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'font.size': 8, 'axes.titlesize': 9,
                     'axes.edgecolor': '#000000', 'axes.linewidth': 0.6, 'xtick.color': '#000000',
                     'ytick.color': '#000000', 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
                     'legend.fontsize': 7.3, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})

def ptitle(ax, title):
    letter, rest = title.split('   ', 1)
    rest = ' '.join(rest.split())
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom')
    ax.set_title(rest, loc='left', fontweight='normal', fontsize=8.5)

NEVER = 16000

rows = []
for f in sorted(glob.glob('res_*.json')):
    r = json.load(open(f))
    rows.append(dict(R2=r['R2'], delta=r['delta'], noise=r['noise'], visits=r['visits'], learner=r['learner'],
                     rep=r['rep'], frozen=r['frozen'], source_r2=r['source_r2'],
                     thr_local=r['thr_local']['threshold'], thr_local_shrunk=r['thr_local, shrunk']['threshold'],
                     benefit=r['benefit_residual, shrunk'], benefit_recal=r['benefit_recalibration, shrunk'],
                     benefit_local=r['benefit_local'], maxcost=r['maxcost_residual, shrunk'],
                     worse_upto=r['worse_upto_residual, shrunk']))
D = pd.DataFrame(rows)
for c in ['benefit', 'benefit_recal', 'benefit_local', 'thr_local', 'thr_local_shrunk']:
    D[c] = pd.to_numeric(D[c], errors='coerce')
D.to_csv('simulation_results.csv', index=False)


base = (D.noise == 'gaussian') & (D.visits == 2) & (D.learner == 'boosting')
G = D[base]
ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000]


def ylog(ax):
    ax.set_yscale('log')
    ax.set_yticks(ticks + [NEVER])
    ax.set_yticklabels([f'{t:,}' for t in ticks] + ['none'])
    ax.minorticks_off()
    ax.set_ylim(15, 24000)
    ax.axhline(8000, color=MUTED, lw=0.6, ls=':')
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)


fig, axs = plt.subplots(1, 3, figsize=(7.5, 3.3), gridspec_kw={'width_ratios': [1.15, 0.9, 1.25]})
ax = axs[0]
for k, r2 in enumerate(sorted(G.R2.unique())):
    g = G[G.R2 == r2].copy()
    g['b'] = g.benefit.fillna(NEVER)
    s = g.groupby('delta').b.agg(['median', 'min', 'max']).reset_index()
    x = s.delta + (k - 1.5) * 0.008
    ax.vlines(x, s['min'], s['max'], color=COLS[k], lw=1.0)
    ax.plot(x, s['median'], marker='o', ms=4, color=COLS[k], lw=1.3, label=f'R² = {r2:.2f}')
ylog(ax)
ax.set_xticks([0, 0.1, 0.25, 0.5])
ax.set_xticklabels(['0', '0.1', '0.25', '0.5'])
ax.set_xlabel('Domain shift δ (share of signal that moves)')
ax.set_ylabel('Target patients')
ptitle(ax, 'A   Shrunk residual boosting beats frozen')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)

ax = axs[1]
for k, (col, lab, mk) in enumerate([('thr_local', 'Local training', 's'), ('thr_local_shrunk', 'Local training, shrunk', 'o')]):
    g = G.copy()
    g['t'] = g[col].fillna(NEVER)
    st = g.groupby('R2').t.agg(['median', 'min', 'max']).reset_index()
    x = st.R2 + (k - 0.5) * 0.006
    ax.vlines(x, st['min'], st['max'], color=COLS[k], lw=1.0)
    ax.plot(x, st['median'], marker=mk, ms=4, color=COLS[k], lw=1.3, label=lab)
ylog(ax)
ax.set_xticks([0.05, 0.1, 0.2, 0.3])
ax.set_xticklabels(['.05', '.10', '.20', '.30'])
ax.set_xlim(0.0, 0.33)
ax.set_xlabel('Signal R²')
ptitle(ax, 'B   Local training beats persistence')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)

ax = axs[2]
cases = [('Base case', (D.R2 == 0.2) & (D.delta == 0.25) & base),
         ('Heavy-tailed noise, t(3)', D.noise == 't3'),
         ('1 transition / patient', D.visits == 1),
         ('3 transitions / patient', D.visits == 3),
         ('Linear learners', D.learner == 'linear')]
yy = np.arange(len(cases))[::-1]
for (lab, m), y in zip(cases, yy):
    g = D[m]
    for col, c, mk, dy in [('thr_local', COLS[0], 's', 0.12), ('benefit', COLS[2], 'o', -0.12)]:
        v = g[col].fillna(NEVER)
        ax.hlines(y + dy, v.min(), v.max(), color=c, lw=1.0)
        ax.plot(v.median(), y + dy, mk, color=c, ms=4.5)
ax.plot([], [], 's', color=COLS[0], label='local beats persistence')
ax.plot([], [], 'o', color=COLS[2], label='shrunk residual beats frozen')
ax.set_yticks(yy)
ax.set_yticklabels([c[0] for c in cases])
ax.set_xscale('log')
ax.set_xticks([20, 100, 1000])
ax.set_xticklabels(['20', '100', '1,000'])
ax.minorticks_off()
ax.set_xlim(12, 2500)
ax.grid(axis='x', color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.set_xlabel('Target patients')
ptitle(ax, 'C   One factor at a time (R² = 0.20, δ = 0.25)')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.3, -0.2), ncol=1)
fig.tight_layout(w_pad=1.2)
fig.savefig('../figs/Fig7_parametric_simulation.png')
fig.savefig('../figs/Fig7_parametric_simulation.pdf')


# summary for the text
print('runs', len(D))
print('benefit (median by R2 x delta):')
print(G.assign(b=G.benefit.fillna(np.inf)).pivot_table(index='R2', columns='delta', values='b', aggfunc='median'))
print('local threshold (median):')
print(G.pivot_table(index='R2', columns='delta', values='thr_local', aggfunc='median'))
print('local shrunk threshold (median):')
print(G.pivot_table(index='R2', columns='delta', values='thr_local_shrunk', aggfunc='median'))
print('max cost of shrunk residual: overall max', D.maxcost.max(), 'median', D.maxcost.median())
print(D.groupby('delta').maxcost.max())
print('frozen by delta', G.groupby('delta').frozen.median())
print('never-benefit share by delta', G.groupby('delta').benefit.apply(lambda s: s.isna().mean()))
print('benefit recal', G.pivot_table(index='R2', columns='delta', values='benefit_recal', aggfunc='median'))
for lab, m in cases:
    g = D[m]
    print(lab, 'thr_local', g.thr_local.tolist(), 'benefit', g.benefit.tolist(), 'maxcost', g.maxcost.round(4).tolist())
