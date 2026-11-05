"""Figure 2 redrawn in plain journal style: (A) evaluation protocol, (B) strategies table."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                     "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold"})
W, H = 7.2, 5.6
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
LW = 0.7

# ---------------- Panel A
ax.text(0.15, 5.4, "A", fontsize=10, weight="bold", va="center")
ax.text(0.4, 5.4, "Persistence-anchored evaluation", fontsize=8.5, va="center")
steps = [
    ("1  Outcome as change", [r"$d = y_{j+1} - y_j$", "axes wrapped to ±180°", r"$z = d\,/\,s$, with $s$ fixed", "on the source"]),
    ("2  Fit any model", [r"$\hat{z} = f(\mathbf{x})$", r"$\mathbf{x}$: seven current levels", "and log(interval)"]),
    ("3  Back to units", [r"$\hat{y}_{j+1} = y_j + s\,f(\mathbf{x})$", r"$f \equiv 0$ gives $\hat{y}_{j+1} = y_j$:", "persistence lies inside", "the model class"]),
    ("4  Error per patient", [r"$e_i$ = mean over the", "patient's transitions of", r"$|d - s\,f(\mathbf{x})|$"]),
    ("5  Relative error", [r"$R = \bar{e}(f)\,/\,\bar{e}(0)$", r"$R < 1$: better than", "predicting no change"]),
]
bw, gap, x0, ytop, bh = 1.27, 0.12, 0.22, 5.12, 1.18
for k, (title, lines) in enumerate(steps):
    x = x0 + k * (bw + gap)
    ax.add_patch(Rectangle((x, ytop - bh), bw, bh, fill=False, lw=LW, ec="black"))
    ax.text(x + 0.07, ytop - 0.14, title, fontsize=7.5, weight="bold", va="center")
    ax.plot([x, x + bw], [ytop - 0.28, ytop - 0.28], color="black", lw=0.4)
    for i, t in enumerate(lines):
        ax.text(x + bw / 2, ytop - 0.47 - i * 0.17, t, ha="center", va="center", fontsize=7.5)
    if k < 4:
        ax.annotate("", xy=(x + bw + gap, ytop - bh / 2), xytext=(x + bw, ytop - bh / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=LW, color="black", shrinkA=0, shrinkB=0, mutation_scale=6))
ax.text(0.22, 3.75, "Steps 1 and 3 place the comparator, which needs no data, inside the model class; steps 4 and 5 make the result "
        "a statement about patients rather than records.", fontsize=7, va="center")
ax.text(0.22, 3.6, "No step depends on the learner.", fontsize=7, va="center")

# ---------------- Panel B
ax.text(0.15, 3.3, "B", fontsize=10, weight="bold", va="center")
ax.text(0.4, 3.3, "Strategies compared at each local cohort size", fontsize=8.5, va="center")
cols = [0.22, 2.02, 2.72, 3.42, 5.62]  # x of column starts
heads = ["Strategy", "Uses source\nmodel", "Uses target\npatients", "What is fitted in the target centre",
         "Shrunk toward*"]
rows = [
    ("Frozen (reference)", "Yes", "No", "Nothing", "–"),
    ("Recalibration (least squares)", "Yes", "Yes", "One slope per measurement", "–"),
    ("Recalibration (absolute error)", "Yes", "Yes", "One slope, weighted-median fit", "–"),
    ("Recalibration, shrunk", "Yes", "Yes", "One slope", "Slope = 1"),
    ("Local training", "No", "Yes", "Full boosting model", "–"),
    ("Local training, shrunk", "No", "Yes", "Full boosting model", "Zero change"),
    ("Augmented", "Yes", "Yes", "Boosting model, source output as input", "–"),
    ("Residual boosting", "Yes", "Yes", "Boosting model of the residual", "–"),
    ("Residual boosting, shrunk", "Yes", "Yes", "Boosting model of the residual", "The frozen model"),
    ("Pooled source + target", "Source data", "Yes", "Boosting model on both cohorts", "–"),
]
xr = 7.0
ytab = 3.08
ax.plot([0.22, xr], [ytab, ytab], color="black", lw=0.9)
for c, h in zip(cols, heads):
    ax.text(c if c == cols[0] else c, ytab - 0.17, h, fontsize=7.5, weight="bold", va="center",
            ha="left" if c in (cols[0], cols[3], cols[4]) else "left", linespacing=1.1)
yh = ytab - 0.38
ax.plot([0.22, xr], [yh, yh], color="black", lw=0.5)
rh = 0.235
for i, r in enumerate(rows):
    y = yh - 0.14 - i * rh
    bold = r[0] in ("Frozen (reference)", "Residual boosting, shrunk")
    for c, v in zip(cols, r):
        ax.text(c, y, v, fontsize=7.5, va="center", weight="bold" if (bold and c == cols[0]) else "normal")
yb = yh - 0.14 - (len(rows) - 1) * rh - 0.14
ax.plot([0.22, xr], [yb, yb], color="black", lw=0.9)
ax.text(0.22, yb - 0.15, "*Shrinkage weight chosen by five-fold cross-validation within the local cohort, with folds grouped by patient.", fontsize=7, va="center")
ax.set_ylim(yb - 0.28, H)
fig.set_size_inches(W, H - (yb - 0.28))
fig.savefig("figs/Fig2_protocol_strategies.pdf")
fig.savefig("figs/Fig2_protocol_strategies.png", dpi=300)
print("ok")
