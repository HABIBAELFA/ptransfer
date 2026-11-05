import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats

OUT = 'figs'
import os
os.makedirs(OUT, exist_ok=True)

INK = '#0b0b0b'
INK2 = '#333333'
MUTED = '#666666'
GRID = '#e3e3e3'
BLUE, ORANGE, AQUA, VIOLET = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
BOXFILL = '#f3f2ee'
SRCFILL = '#e8f0fb'
TGTFILL = '#fdeee7'

plt.rcParams.update({
    'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'font.size': 8, 'axes.titlesize': 9,
    'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.edgecolor': '#000000', 'axes.linewidth': 0.6,
    'xtick.color': '#000000', 'ytick.color': '#000000', 'axes.labelcolor': INK,
    'text.color': INK, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

B = pd.read_csv('data/B_main.csv')
BI = pd.read_csv('data/B_intra.csv')
T = pd.read_csv('data/B_main_thresholds.csv')
TI = pd.read_csv('data/B_intra_thresholds.csv')
FZ = json.load(open('data/B_main_frozen.json'))
FZI = json.load(open('data/B_intra_frozen.json'))
O = json.load(open('data/other.json'))
A = O['A']

NAMES = {'local': 'Local training', 'local retreci': 'Local training, shrunk',
         'recalibration': 'Recalibration (least squares)',
         'recalibration retrecie': 'Recalibration, shrunk',
         'recalibration MAE': 'Recalibration (absolute error)',
         'augmente': 'Augmented', 'boosting residuel': 'Residual boosting',
         'boosting residuel retreci': 'Residual boosting, shrunk',
         'poole': 'Pooled source + target'}
MEAS = {'PR': 'PR interval', 'QRS': 'QRS duration', 'QT': 'QT interval',
        'QTc': 'QTc interval', 'P_wave_axis': 'P axis', 'QRS_axis': 'QRS axis',
        'T_wave_axis': 'T axis', 'RR': 'RR interval', 'composite': 'Composite'}



def ptitle(ax, title):
    letter, rest = title.split('   ', 1)
    rest = ' '.join(rest.split())
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='right', va='bottom')
    ax.set_title(rest, loc='left', fontweight='normal', fontsize=8.5)


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png')
    fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig)


def logx(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:,}' for t in ticks])
    ax.minorticks_off()
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# =========================================================================
# Figure 3 — relative error against target cohort size
# =========================================================================
ARMS3 = [('local', BLUE, 'o', '-', True), ('local retreci', BLUE, 'o', '--', False),
         ('recalibration retrecie', ORANGE, 's', '-', True),
         ('boosting residuel retreci', AQUA, '^', '-', True)]


def panel_abs(ax, D, fz, title, ticks):
    ax.axhline(1.0, color=MUTED, lw=1.0)
    ax.text(ticks[-1] * 1.05, 1.0015, 'Persistence (no data)', fontsize=6.8, color=INK2, va='bottom', ha='right')
    ax.axhspan(fz[1], fz[2], color='#d9d8d3', alpha=0.6, lw=0)
    ax.axhline(fz[0], color=INK, lw=1.0, ls=(0, (4, 2)))
    for s, c, mk, ls, filled in ARMS3:
        d = D[D.strategy == s].sort_values('n')
        ax.fill_between(d.n, d.lo, d.hi, color=c, alpha=0.13, lw=0)
        ax.plot(d.n, d.rerror, color=c, lw=1.4, ls=ls, marker=mk, ms=3.6,
                mfc=c if filled else 'white', mec=c, mew=0.9, label=NAMES[s])
    ax.plot([], [], color=INK, lw=1.0, ls=(0, (4, 2)), label='Frozen imported model (grey band: 95% CI)')
    logx(ax, ticks)
    ax.set_ylim(0.92, 1.055)
    ax.set_xlabel('Target patients used for updating')
    ptitle(ax, title)


fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.3), sharey=True)
panel_abs(axs[0], B, FZ, 'A   All transitions (primary)', [20, 100, 1000, 12000])
panel_abs(axs[1], BI, FZI, 'B   Within-department transitions', [20, 100, 1000, 6000])
axs[0].set_ylabel('Relative error against persistence')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.07, 1, 1))
save(fig, 'Fig3_relative_error')

# =========================================================================
# Figure 4 — difference from the frozen model
# =========================================================================
ARMS4 = [('recalibration retrecie', ORANGE, 's', '-'),
         ('boosting residuel retreci', AQUA, '^', '-'),
         ('poole', VIOLET, 'D', '-')]


def panel_diff(ax, D, title, ticks):
    ax.axhline(0, color=INK, lw=1.0, ls=(0, (4, 2)))
    ax.text(ticks[-1] * 1.05, 0.0165, 'frozen model better', fontsize=6.8, color=INK2,
            ha='right', va='center')
    ax.text(ticks[0] * 0.9, -0.0112, 'updating better', fontsize=6.8, color=INK2,
            ha='left', va='center')
    for k, (s, c, mk, ls) in enumerate(ARMS4):
        d = D[D.strategy == s].sort_values('n')
        off = np.exp((k - 1) * 0.045)
        ax.errorbar(d.n * off, d['diff'], yerr=[d['diff'] - d.dlo, d.dhi - d['diff']], color=c,
                    lw=1.2, marker=mk, ms=3.6, mfc=c, mec='white', mew=0.5,
                    elinewidth=0.8, capsize=0, label=NAMES[s])
    logx(ax, ticks)
    ax.set_ylim(-0.0125, 0.0185)
    ax.set_xlabel('Target patients used for updating')
    ptitle(ax, title)


fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
panel_diff(axs[0], B, 'A   All transitions (primary)', [20, 100, 1000, 12000])
panel_diff(axs[1], BI, 'B   Within-department transitions', [20, 100, 1000, 6000])
axs[0].set_ylabel('Strategy − frozen (relative error)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, 'Fig4_difference_from_frozen')

# =========================================================================
# Figure 5 — per-measurement thresholds
# =========================================================================
order = T[T.arm == 'local'].sort_values('thr').output.tolist()
order = [o for o in order if o != 'composite'] + ['composite']
fig, ax = plt.subplots(figsize=(7.5, 3.3))
yy = np.arange(len(order))[::-1]
for (tab, arm, c, mk, fill, dy, lab) in [
        (T, 'local', BLUE, 'o', True, 0.18, 'Local training, all transitions'),
        (T, 'local retreci', BLUE, 'o', False, 0.0, 'Local training, shrunk, all transitions'),
        (TI, 'local', ORANGE, 'D', True, -0.18, 'Local training, within-department')]:
    d = tab[tab.arm == arm].set_index('output').loc[order]
    ax.hlines(yy + dy, d.lo, d.hi, color=c, lw=1.1)
    ax.plot(d.thr, yy + dy, mk, color=c, mfc=c if fill else 'white', mec=c, ms=5, mew=1.0,
            ls='none', label=lab)
    if arm == 'local' and tab is TI:
        for o, yv in zip(order, yy + dy):
            nv = d.loc[o, 'never']
            if nv > 0:
                ax.text(d.loc[o, 'hi'] * 1.15, yv, f'no crossing in {nv}% of resamples',
                        fontsize=6.6, color=INK2, va='center')
ax.set_yticks(yy)
ax.set_yticklabels([MEAS[o] for o in order])
ax.set_xscale('log')
tk = [20, 50, 100, 200, 500, 1000, 2000, 5000, 12000]
ax.set_xticks(tk)
ax.set_xticklabels([f'{t:,}' for t in tk])
ax.minorticks_off()
ax.set_xlim(15, 40000)
ax.grid(axis='x', color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.set_xlabel('Target patients needed to beat persistence durably (log scale; bars: 95% CI)')
ax.axhline(0.5, color=GRID, lw=0.8)
ax.legend(loc='lower center', frameon=False, ncol=3, bbox_to_anchor=(0.45, 1.0))
fig.tight_layout()
save(fig, 'Fig5_thresholds')

# =========================================================================
# Figure 6 — source R2 against thresholds, two databases
# =========================================================================
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2))
M7 = A['measures']
thr_main = T[T.arm == 'local'].set_index('output').loc[M7]
r2 = np.array(A['R2_multi'])
rho1, _ = stats.spearmanr(r2, thr_main.thr)
MI = O['MIMIC']
rho2 = MI['rho']
for ax, x, yv, lo, hi, labs, c, title, rho, pstr in [
        (axs[0], r2, thr_main.thr.values, thr_main.lo.values, thr_main.hi.values, M7, BLUE,
         'A   ECG-ViEW II (department split)', rho1, 'p < 0.001'),
        (axs[1], np.array(MI['R2_oos']), np.array(MI['thr']), None, None, MI['measures'], ORANGE,
         'B   MIMIC-IV-ECG (device split)', rho2, f"p = {MI['p']}")]:
    if lo is not None:
        ax.vlines(x, lo, hi, color=c, lw=1.0, alpha=0.7)
    ax.plot(x, yv, 'o', color=c, ms=5.5, mec='white', mew=0.6)
    if ax is axs[0]:
        OFF = {'QT': (5, 4, 'left'), 'QTc': (-5, -10, 'right'), 'P_wave_axis': (-6, 3, 'right'),
               'PR': (-6, 5, 'right')}
    else:
        OFF = {'QT': (-6, 5, 'right'), 'QTc': (-6, -3, 'right'), 'PR': (-6, 5, 'right'),
               'P_wave_axis': (5, -11, 'left'), 'RR': (6, -3, 'left')}
    for xi, yi, lb in zip(x, yv, labs):
        dx, dy, ha = OFF.get(lb, (5, 3, 'left'))
        ax.annotate(MEAS[lb], (xi, yi), xytext=(dx, dy), textcoords='offset points', fontsize=6.8,
                    color=INK2, ha=ha)
    ax.set_yscale('log')
    ax.set_yticks([30, 100, 300, 1000, 3000, 10000])
    ax.set_yticklabels(['30', '100', '300', '1,000', '3,000', '10,000'])
    ax.minorticks_off()
    ax.set_ylim(20, 15000)
    ax.set_xlim(0.0, 0.3)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('R² of a linear model of change, source cohort only')
    ptitle(ax, title)
    ax.text(0.97, 0.95, f'Spearman ρ = {rho:.2f}\n{pstr}', transform=ax.transAxes, ha='right',
            va='top', fontsize=7.3)
axs[0].set_ylabel('Target patients needed to beat persistence')
fig.tight_layout()
save(fig, 'Fig6_R2_ranking')

# =========================================================================
# Figure 7 — cost of flexibility
# =========================================================================
fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.1), gridspec_kw={'width_ratios': [1.35, 1]})
ax = axs[0]
ax.axhline(1.0, color=MUTED, lw=1.0)
ax.axhline(FZ[0], color=INK, lw=1.0, ls=(0, (4, 2)))
ax.text(12, FZ[0] - 0.004, 'Frozen imported model', fontsize=6.8, color=INK2, va='top')
ax.text(12, 1.002, 'Persistence', fontsize=6.8, color=INK2, va='bottom')
d = B[B.strategy == 'local'].sort_values('n')
ax.plot(d.n, d.rerror, color=BLUE, marker='o', ms=3.4, lw=1.4, label='Local gradient boosting')
ax.plot(A['A7_n'], A['A7_uni'], color=ORANGE, marker='s', ms=3.4, lw=1.4,
        label='Local linear model, 2 parameters per measurement')
logx(ax, [10, 100, 1000, 12000])
ax.set_ylim(0.92, 1.07)
ax.set_xlabel('Target patients')
ax.set_ylabel('Relative error against persistence')
ptitle(ax, 'A   Local learners, all transitions')
ax.legend(frameon=False, loc='upper right')

ax = axs[1]
S = O['SIM']
xs = np.arange(3)
ax.bar(xs - 0.18, S['ols'], 0.34, color=ORANGE, label='Least squares')
ax.bar(xs + 0.18, S['gbm'], 0.34, color=BLUE, label='Gradient boosting')
for i, r in enumerate(S['ratio']):
    ax.text(xs[i] + 0.18, S['gbm'][i] * 1.25, f'×{r:.1f}', ha='center', fontsize=7.3, color=INK)
ax.set_yscale('log')
ax.set_yticks([30, 100, 300, 1000, 3000, 10000])
ax.set_yticklabels(['30', '100', '300', '1,000', '3,000', '10,000'])
ax.minorticks_off()
ax.set_ylim(20, 20000)
ax.set_xticks(xs)
ax.set_xticklabels([f'{v:.2f}' for v in S['rsb']])
ax.set_xlabel('Signal-to-noise ratio of the simulated process')
ax.set_ylabel('Subjects needed to beat persistence')
ptitle(ax, 'B   Simulation, linear truth')
ax.grid(axis='y', color=GRID, lw=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper right')
fig.tight_layout()
save(fig, 'FigS2_cost_of_flexibility')
print('done')
