import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK2, MUTED, GRID = '#333333', '#666666', '#e3e3e3'
BLUE, ORANGE = '#0072B2', '#D55E00'
plt.rcParams.update({'font.family': 'Liberation Sans', 'pdf.fonttype': 42, 'font.size': 7.5, 'axes.edgecolor': '#000000', 'axes.linewidth': 0.6,
                     'xtick.color': '#000000', 'ytick.color': '#000000', 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})
ev = pd.read_csv('data/ev_curves.csv'); mi = pd.read_csv('data/mimic_curves.csv')
M = ['QT', 'QTc', 'P_wave_axis', 'PR', 'QRS', 'T_wave_axis', 'QRS_axis', 'RR']
NAME = {'PR': 'PR interval', 'QRS': 'QRS duration', 'QT': 'QT interval', 'QTc': 'QTc interval', 'P_wave_axis': 'P axis',
        'QRS_axis': 'QRS axis', 'T_wave_axis': 'T axis', 'RR': 'RR interval'}
fig, axs = plt.subplots(2, 4, figsize=(7.5, 3.9), sharex=True, sharey=True)
for ax, m in zip(axs.flat, M):
    ax.axhline(1.0, color=MUTED, lw=0.9)
    if m in ev:
        ax.plot(ev.n, ev[m], color=BLUE, marker='o', ms=2.5, lw=1.2, label='ECG-ViEW II')
    ax.plot(mi.n, mi[m], color=ORANGE, marker='s', ms=2.5, lw=1.2, label='MIMIC-IV-ECG')
    ax.set_xscale('log'); ax.set_xticks([20, 100, 1000, 10000]); ax.set_xticklabels(['20', '100', '1k', '10k']); ax.minorticks_off()
    ax.set_ylim(0.86, 1.08); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(NAME[m], loc='left', fontsize=8, fontweight='normal')
for ax in axs[1]: ax.set_xlabel('Target patients')
for ax in axs[:, 0]: ax.set_ylabel('Relative error')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig('figs/FigS1_per_measurement_curves.png'); fig.savefig('figs/FigS1_per_measurement_curves.pdf')
print('ok')
