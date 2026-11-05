"""Figure 1 redrawn as a plain CONSORT-style flow diagram (journal style)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "pdf.fonttype": 42})
W, H = 7.2, 7.3  # inches; JBI full width
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0.3, 7.6); ax.axis("off")
LW = 0.7

def box(x, y, w, h, lines, bold_first=False, size=7.5):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=LW, ec="black"))
    n = len(lines); step = 0.135
    y0 = y + h / 2 + (n - 1) * step / 2
    for k, t in enumerate(lines):
        ax.text(x + w / 2, y0 - k * step, t, ha="center", va="center", fontsize=size,
                weight="bold" if (bold_first and k == 0) else "normal")
    return (x, y, w, h)

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=LW, color="black", shrinkA=0, shrinkB=0, mutation_scale=7))

def line(x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color="black", lw=LW)

# ---------------- Panel A
ax.text(0.15, 7.4, "A", fontsize=10, weight="bold", va="center")
ax.text(0.4, 7.4, "Development database and transfer design (ECG-ViEW II)", fontsize=8.5, va="center")
cx, bw = 2.35, 3.9
x = cx - bw / 2
b1 = box(x, 6.75, bw, 0.42, ["ECG-ViEW II, one Korean tertiary hospital, 1994–2013",
                              "979,273 ECGs from 461,034 patients"])
arrow(cx, 6.75, cx, 6.35)
box(x, 5.93, bw, 0.42, ["Single acquisition source, seven measurements complete,",
                        "duplicates removed: 726,438 ECGs from 364,417 patients"])
arrow(cx, 5.93, cx, 5.53)
box(x, 5.11, bw, 0.42, ["Consecutive transitions 7 days to 1 year apart",
                        "138,644 transitions from 70,179 patients"])
# split into source / target
line(cx, 5.11, cx, 4.95); line(1.2, 4.95, 3.5, 4.95)
arrow(1.2, 4.95, 1.2, 4.78); arrow(3.5, 4.95, 3.5, 4.78)
box(0.4, 4.3, 1.6, 0.48, ["Source department", "41,144 patients", "62,942 transitions"], bold_first=True)
box(2.6, 4.3, 1.8, 0.48, ["Target department", "17,291 patients (disjoint)", "29,789 transitions"], bold_first=True)
arrow(1.2, 4.3, 1.2, 3.98)
box(0.4, 3.5, 1.6, 0.48, ["Source model:", "gradient boosting, one per", "measurement, trained once"])
line(3.5, 4.3, 3.5, 4.14); line(3.0, 4.14, 4.0, 4.14)
arrow(3.0, 4.14, 3.0, 3.98); arrow(4.0, 4.14, 4.0, 3.98)
box(2.6, 3.5, 0.82, 0.48, ["Reservoir", "12,104", "patients"])
box(3.55, 3.5, 0.85, 0.48, ["Test set", "5,187", "patients"])
arrow(3.0, 3.5, 3.0, 3.18)
box(2.15, 2.72, 2.25, 0.46, ["Draw n patients from the reservoir", "n = 20 to 12,000 (16 sizes)", "20, 10 or 5 draws per size"])
# frozen model + draws feed strategies
line(1.2, 3.5, 1.2, 2.45); arrow(1.2, 2.45, 1.32, 2.45)
arrow(3.27, 2.72, 3.27, 2.56)
box(1.32, 1.98, 2.9, 0.58, ["Frozen reference and nine updating strategies",
                            "recalibration (least squares, absolute error, shrunk);",
                            "local training (raw, shrunk); augmented;",
                            "residual boosting (raw, shrunk); pooled source + target"], bold_first=True)
line(4.4, 3.74, 4.62, 3.74); line(4.62, 3.74, 4.62, 1.55); arrow(4.62, 1.55, 4.22, 1.55)
arrow(2.77, 1.98, 2.77, 1.78)
box(1.32, 1.32, 2.9, 0.46, ["Evaluation on the fixed test set against persistence",
                            "relative error averaged within patient (Fig. 2)",
                            "two-level bootstrap, 2,000 resamples"], bold_first=False)
# outputs
line(2.77, 1.32, 2.77, 1.18); line(1.55, 1.18, 4.0, 1.18)
for xo in (1.55, 2.77, 4.0):
    arrow(xo, 1.18, xo, 1.02)
box(0.98, 0.5, 1.14, 0.52, ["Each strategy", "versus the", "frozen model"])
box(2.2, 0.5, 1.14, 0.52, ["Patients needed", "to beat persistence,", "per measurement"])
box(3.43, 0.5, 1.14, 0.52, ["Source-only R²", "versus these", "thresholds"])

# ---------------- Panel B
bx, bwid = 5.05, 2.0
ax.text(bx - 0.15, 7.4, "B", fontsize=10, weight="bold", va="center")
ax.text(bx + 0.1, 7.4, "Sensitivity, replication and simulation", fontsize=8.5, va="center")
box(bx, 6.35, bwid, 0.82, ["Sensitivity analysis", "both ECGs of a transition in", "the same department",
                           "80,714 transitions; source 26,915;", "test 3,085; reservoir 7,200"], bold_first=True)
box(bx, 5.05, bwid, 0.82, ["Replication: MIMIC-IV-ECG (USA)", "8 measurements including RR", "split by recording device",
                           "source 31,512 subjects;", "target 24,697 subjects"], bold_first=True)
box(bx, 3.95, bwid, 0.62, ["External cohort: PF12RED (Spain)", "44 professional footballers", "106 transitions"], bold_first=True)
box(bx, 2.6, bwid, 0.88, ["Parametric simulation", "20 scenarios × 3 seeds:", "signal R², domain shift, noise,",
                          "visits per patient, learner;", "plus local linear models"], bold_first=True)
fig.savefig("figs/Fig1_study_design.pdf")
fig.savefig("figs/Fig1_study_design.png", dpi=300)
print("ok")
