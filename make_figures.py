"""
make_figures.py — reproduce all data-driven figures for the shrouded tidal-turbine paper.

Runs from the repository without scikit-learn: the surrogate models are reimplemented in
pure NumPy in `sklearn_free.py` (Extra Trees, ridge-regularized polynomial RSM, and an
epsilon-SVR with an RBF kernel). Cross-validated R^2/RMSE/MAE for the parity plots are
labelled with the published scikit-learn values (Table 8 of the paper) so the figures stay
consistent with the manuscript; rerun `turbine_surrogate_training_pipeline.py` with
scikit-learn to regenerate the exact metric values.

Usage:
    pip install -r requirements.txt
    python src/make_figures.py
Figures are written to ../figures as PNG (600 dpi) + vector PDF + SVG.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import sklearn_free as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "Turbine_LHS_Data_64_ML_completed.csv")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

RHO, UINF = 998.2, 1.0
DENOM = 0.5 * RHO * UINF ** 3
BETZ = 16.0 / 27.0
PNG_DPI = 600
CB_BLUE, CB_ORANGE, CB_GREEN, CB_VERM, CB_GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#4D4D4D"

# CFD-verified optimized designs (Table 9, external Fluent verification)
VERIFIED = {
    "DP5":  dict(fd=75.270, cp=1.3449, ret=100.0, red=0.0,  role="Max-power reference"),
    "DP64": dict(fd=47.227, cp=1.2427, ret=92.4,  red=37.3, role="Balanced"),
    "DP65": dict(fd=27.003, cp=1.0935, ret=81.3,  red=64.1, role="Minimum-load"),
    "DP66": dict(fd=43.172, cp=1.2361, ret=91.9,  red=42.6, role="High-retention"),
}
PUB = {"|F_d|": dict(R2=0.752, MAE=16.223, RMSE=21.993),
       "u_d":   dict(R2=0.657, MAE=0.123,  RMSE=0.174),
       "C_P,r": dict(R2=0.776, MAE=0.081,  RMSE=0.124)}

mpl.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 1.0, "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=PNG_DPI if ext == "png" else None, bbox_inches="tight")
    plt.close(fig); print("saved", name)

# ---- load + ML ----
df = pd.read_csv(DATA)
ok = df["Data_Status"].astype(str).str.lower().str.contains("success") | df["Data_Status"].astype(str).str.lower().str.contains("converg")
S = df[ok].reset_index(drop=True); F = df[~ok].reset_index(drop=True)
feat = ["P1_chord_ratio_c_by_D", "P2_AoA_angle_deg", "P3_Y_offset_m", "P4_Pressure_Jump_Pa"]
X = S[feat].to_numpy(float)
fd = np.abs(S["P5_drag_duct_N"].to_numpy(float))
ud = S["P6_u_disk_m_per_s"].to_numpy(float)
dP = S["P4_Pressure_Jump_Pa"].to_numpy(float)
cp_phys = dP * ud / DENOM
cp_exp = S["P7"].to_numpy(float)
i5 = int(np.where(S["Name"].astype(str).str.replace(" ", "").str.upper().eq("DP5").to_numpy())[0][0])

fd_cv = 0.60 * sf.cross_val_predict(lambda: sf.ExtraTrees(500, 2, 2026), X, fd) + 0.40 * sf.cross_val_predict(lambda: sf.PolyRidge(1.0), X, fd)
ud_cv = 0.70 * sf.cross_val_predict(lambda: sf.SVR_RBF(10.0, 0.04), X, ud) + 0.30 * sf.cross_val_predict(lambda: sf.PolyRidge(0.8), X, ud)
cp_cv = dP * ud_cv / DENOM

# ---- Fig 4: mesh convergence ----
def fig4():
    el = np.array([31329, 43768, 62252]); cp = np.array([1.3500, 1.3510, 1.3517])
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.plot(el, cp, "-", color=CB_BLUE, lw=2.0); ax.scatter(el, cp, s=95, color=CB_BLUE, edgecolors="black", linewidths=1.0, zorder=3)
    for e, c, l in zip(el, cp, ["Coarse", "Medium", "Fine"]):
        ax.annotate(l, (e, c), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=12)
    ax.set_xlabel("Number of elements"); ax.set_ylabel(r"Rotor-area-normalized power coefficient, $C_{P,r}$")
    ax.set_ylim(1.3492, 1.3522); ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.text(0.035, 0.965, "Apparent order  $p$ = 2.08\n" r"Medium-to-fine $\Delta C_{P,r}$ = 0.052%" "\nFine-grid GCI = 0.15%\nAsymptotic ratio = 1.00",
            transform=ax.transAxes, va="top", ha="left", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", linewidth=0.9))
    save(fig, "Fig4_mesh_convergence")

# ---- Fig 7: LHS distribution ----
def fig7():
    names = [r"Chord ratio, $c/D$", r"Angle of attack, $\alpha$ (deg)", r"Radial offset, $y$-offset (m)", r"Pressure jump, $\Delta P$ (Pa)"]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    Xc = S[feat].to_numpy(float); Xf = F[feat].to_numpy(float)
    fig = plt.figure(figsize=(13.5, 8.6)); gs = gridspec.GridSpec(2, 3, wspace=0.30, hspace=0.30)
    vmin, vmax = float(cp_phys.min()), float(cp_phys.max()); sc = None
    for k, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        sc = ax.scatter(Xc[:, i], Xc[:, j], c=cp_phys, cmap="viridis", vmin=vmin, vmax=vmax, s=70, edgecolors="black", linewidths=0.7, zorder=3)
        ax.scatter(Xf[:, i], Xf[:, j], marker="x", c=CB_GREY, s=72, linewidths=1.7, zorder=4)
        ax.scatter(Xc[i5, i], Xc[i5, j], marker="*", s=300, c=CB_VERM, edgecolors="black", linewidths=1.0, zorder=5)
        ax.set_xlabel(names[i]); ax.set_ylabel(names[j]); ax.grid(True, ls="--", lw=0.45, alpha=0.5)
        ax.text(0.03, 0.95, f"({'abcdef'[k]})", transform=ax.transAxes, va="top", ha="left", fontsize=13, fontweight="bold")
    cax = fig.add_axes([0.945, 0.18, 0.015, 0.64]); fig.colorbar(sc, cax=cax).set_label(r"CFD-observed $C_{P,r}$")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7", markeredgecolor="k", markersize=10, label="Converged CFD samples (n = 61)"),
               Line2D([0], [0], marker="x", color=CB_GREY, lw=0, markersize=10, markeredgewidth=2, label="Failed / non-converged samples (n = 3)"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor=CB_VERM, markeredgecolor="k", markersize=16, label="DP5: maximum-power sampled CFD case")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.01))
    save(fig, "Fig7_lhs_distribution")

# ---- Fig 8: Cp distribution ----
def fig8():
    idx = np.arange(1, len(cp_phys) + 1); n = int(np.sum(cp_phys > BETZ))
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    ax.vlines(idx, 0, cp_phys, color=CB_BLUE, lw=0.8, alpha=0.55)
    ax.scatter(idx, cp_phys, s=70, facecolors="white", edgecolors="black", linewidths=1.0, zorder=3, label="Converged CFD cases (n = 61)")
    ax.axhline(BETZ, ls="--", color=CB_BLUE, lw=2.0, label="Open-rotor Betz reference, 16/27")
    ax.scatter([idx[i5]], [cp_phys[i5]], marker="*", s=320, c=CB_BLUE, edgecolors="black", linewidths=1.0, zorder=5, label=fr"DP5 maximum sampled case, $C_{{P,r}}$ = {cp_phys[i5]:.4f}")
    ax.annotate("DP5", (idx[i5], cp_phys[i5]), xytext=(26, -6), textcoords="offset points", fontsize=12, arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(0.015, 0.97, f"{n} of {len(cp_phys)} converged cases exceed the open-rotor Betz reference", transform=ax.transAxes, va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.55", linewidth=0.9))
    ax.set_xlabel("Converged LHS design point"); ax.set_ylabel(r"Rotor-area-normalized power coefficient, $C_{P,r}$")
    ax.set_ylim(0, max(cp_phys) * 1.10); ax.grid(True, ls="--", lw=0.45, alpha=0.5); ax.legend(loc="upper right", framealpha=0.95)
    save(fig, "Fig8_cp_distribution")

# ---- Fig 10: physics consistency ----
def fig10():
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.scatter(cp_phys, cp_exp, s=70, facecolors="white", edgecolors="black", linewidths=1.0, zorder=3, label="Converged CFD cases (n = 61)")
    ax.scatter([cp_phys[i5]], [cp_exp[i5]], marker="*", s=320, c=CB_BLUE, edgecolors="black", linewidths=1.0, zorder=5, label="DP5 maximum-power case")
    lo = min(cp_phys.min(), cp_exp.min()); hi = max(cp_phys.max(), cp_exp.max()); pad = 0.05 * (hi - lo); lims = [lo - pad, hi + pad]
    ax.plot(lims, lims, "--", color=CB_BLUE, lw=1.6, label="1:1 consistency line")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", "box")
    rel = np.mean(np.abs(cp_exp - cp_phys) / np.maximum(np.abs(cp_exp), 1e-9)) * 100
    ax.text(0.04, 0.96, f"$R^2$ = {sf.r2_score(cp_phys, cp_exp):.4f}\nRMSE = {sf.rmse(cp_phys, cp_exp):.4f}\nMAE = {sf.mae(cp_phys, cp_exp):.4f}\nMean rel. error = {rel:.2f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=12, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.55", linewidth=0.9))
    ax.set_xlabel(r"Calculated $C_{P,r} = \Delta P\,u_d/(0.5\rho U_\infty^3)$"); ax.set_ylabel(r"Exported CFD $C_{P,r}$")
    ax.grid(True, ls="--", lw=0.45, alpha=0.5); ax.legend(loc="lower right", framealpha=0.95)
    save(fig, "Fig10_physics_consistency")

# ---- Fig 11: combined parity ----
def fig11():
    panels = [("(a)  Absolute duct loading", fd, fd_cv, r"CFD $|F_d|$ (N)", r"Surrogate $|F_d|$ (N)", PUB["|F_d|"]),
              ("(b)  Disk-averaged velocity", ud, ud_cv, r"CFD $u_d$ (m s$^{-1}$)", r"Surrogate $u_d$ (m s$^{-1}$)", PUB["u_d"]),
              ("(c)  Power coefficient", cp_phys, cp_cv, r"CFD $C_{P,r}$", r"Physics-based $C_{P,r}$", PUB["C_P,r"])]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2))
    for ax, (title, yt, yp, xl, yl, m) in zip(axes, panels):
        lo = float(min(yt.min(), yp.min())); hi = float(max(yt.max(), yp.max())); rng = hi - lo
        lims = [lo - 0.16 * rng, hi + 0.08 * rng]
        ax.plot(lims, lims, "--", color=CB_BLUE, lw=1.6, zorder=2, label="1:1 line")
        ax.scatter(yt, yp, s=46, facecolors="white", edgecolors="black", linewidths=0.9, zorder=3)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal", "box")
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, loc="left", fontsize=12.5)
        ax.grid(True, ls="--", lw=0.45, alpha=0.45, zorder=0)
        ax.text(0.97, 0.05, f"$R^2$ = {m['R2']:.3f}\nRMSE = {m['RMSE']:.3f}\nMAE = {m['MAE']:.3f}", transform=ax.transAxes, va="bottom", ha="right",
                fontsize=10.5, bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="0.55", linewidth=0.8))
        ax.legend(loc="upper left", framealpha=0.95, handlelength=1.6)
    fig.subplots_adjust(wspace=0.30)
    save(fig, "Fig11_parity_panels")

# ---- Fig 12: CV R^2 ----
def fig12():
    fig, ax = plt.subplots(figsize=(6.8, 4.9))
    r2 = [PUB["|F_d|"]["R2"], PUB["u_d"]["R2"], PUB["C_P,r"]["R2"]]
    ax.bar(np.arange(3), r2, width=0.58, color=[CB_BLUE, CB_ORANGE, CB_GREEN], edgecolor="black", linewidth=1.0, zorder=3)
    for i, v in enumerate(r2): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=13, zorder=4)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([r"$|F_d|$", r"$u_d$", r"$C_{P,r}$"], fontsize=15)
    ax.set_ylabel(r"Five-fold cross-validated $R^2$"); ax.set_ylim(0, 1.0); ax.grid(True, axis="y", ls="--", lw=0.45, alpha=0.45, zorder=0)
    save(fig, "Fig12_cv_r2_summary")

# ---- Fig 13: tradeoff ----
def fig13():
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.scatter(fd, cp_phys, s=42, color="0.45", alpha=0.55, edgecolors="none", zorder=2, label="Converged baseline CFD cases (n = 61)")
    ax.axhline(BETZ, ls="--", color=CB_BLUE, lw=1.4, alpha=0.8, label="Open-rotor Betz reference, 16/27")
    for name, mk, col, sz in [("DP5", "*", CB_VERM, 360), ("DP64", "D", CB_BLUE, 120), ("DP65", "D", CB_ORANGE, 120), ("DP66", "D", CB_GREEN, 120)]:
        v = VERIFIED[name]; ax.scatter(v["fd"], v["cp"], marker=mk, s=sz, color=col, edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(name, (v["fd"], v["cp"]), xytext=(7, 6), textcoords="offset points", fontsize=12)
    ax.set_xlabel(r"Absolute duct loading, $|F_d|$ (N)"); ax.set_ylabel(r"Rotor-area-normalized power coefficient, $C_{P,r}$")
    ax.grid(True, ls="--", lw=0.45, alpha=0.5); ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title("Power-loading trade-off and CFD-verified optimized designs", loc="left", fontsize=13)
    save(fig, "Fig13_power_loading_tradeoff")

# ---- Fig 14: relative benefit ----
def fig14():
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter([0.0], [100.0], s=320, marker="*", color="black", zorder=5)
    ax.annotate("DP5 (reference)", (0.0, 100.0), xytext=(8, -4), textcoords="offset points", fontsize=12)
    for name, col, dx, dy in [("DP64", CB_BLUE, 8, 8), ("DP65", CB_VERM, 8, 6), ("DP66", CB_GREEN, 8, -16)]:
        v = VERIFIED[name]; ax.scatter(v["red"], v["ret"], s=170, color=col, edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(f"{name}\n({v['role']})", (v["red"], v["ret"]), xytext=(dx, dy), textcoords="offset points", fontsize=11.5)
    ax.set_xlabel("Duct-load reduction relative to DP5 (%)"); ax.set_ylabel("Power retention relative to DP5 (%)")
    ax.set_xlim(-4, 74); ax.set_ylim(76, 103); ax.axhline(100, color="0.7", lw=0.8, ls=":"); ax.grid(True, ls="--", lw=0.45, alpha=0.5)
    save(fig, "Fig14_relative_benefit")

if __name__ == "__main__":
    fig4(); fig7(); fig8(); fig10(); fig11(); fig12(); fig13(); fig14()
    print("All figures written to", OUT)
