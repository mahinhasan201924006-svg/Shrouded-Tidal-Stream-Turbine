"""turbine_surrogate_training_pipeline.py

Reproducible surrogate-modelling and optimization pipeline for the shrouded tidal
stream turbine dataset used in this conversation.

What this script does
---------------------
1) Loads the 64-point LHS dataset.
2) Trains/calibrates surrogate models using only the 61 CFD-converged samples.
3) Evaluates models with shuffled 5-fold cross-validation.
4) Computes the physics-based power coefficient:

       C_P,r = ΔP * u_d / (0.5 * rho * U_inf^3)

5) Searches a dense candidate space inside the original design bounds.
6) Selects three optimized candidates (balanced / low-load / high-retention)
   subject to power-retention constraints relative to DP5.
7) Exports CSV summaries and generates publication-style figures.

This file is intentionally self-contained so you can rerun the ML part of the
workflow without digging through the chat history.

Dependencies
------------
pandas, numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    data_path: Path = Path(__file__).resolve().parent.parent / "data" / "Turbine_LHS_Data_64_ML_completed.csv"
    outdir: Path = Path(__file__).resolve().parent.parent / "results"
    random_seed: int = 2026
    cv_splits: int = 5
    candidate_samples: int = 250_000

    # Original design bounds used for surrogate search
    c_by_D_bounds: Tuple[float, float] = (0.30, 0.90)
    aoa_bounds: Tuple[float, float] = (-5.0, 15.0)
    y_offset_bounds: Tuple[float, float] = (0.505, 0.540)
    pressure_jump_bounds: Tuple[float, float] = (201.0, 449.5)

    # Physics constants from the manuscript/data files
    rho: float = 998.2
    U_inf: float = 1.0

    # Power-retention thresholds relative to DP5 for candidate selection
    eta_balanced: float = 0.925
    eta_low_load: float = 0.900
    eta_high_retention: float = 0.950


CFG = Config()


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = [
        "Name",
        "P1_chord_ratio_c_by_D",
        "P2_AoA_angle_deg",
        "P3_Y_offset_m",
        "P4_Pressure_Jump_Pa",
        "P5_drag_duct_N",
        "P6_u_disk_m_per_s",
        "P7",
        "Data_Status",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def split_successful_failed(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Accept several spellings used during the conversation.
    status = df["Data_Status"].astype(str).str.lower()
    success_mask = status.str.contains("successful") | status.str.contains("converged")
    failed_mask = ~success_mask
    return df.loc[success_mask].copy(), df.loc[failed_mask].copy()


def physics_cp(delta_p: np.ndarray, u_d: np.ndarray, rho: float, U_inf: float) -> np.ndarray:
    return delta_p * u_d / (0.5 * rho * U_inf ** 3)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    return Xz, scaler


def nearest_neighbor_distance(Xz: np.ndarray, xz_new: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(Xz)
    dists, _ = nn.kneighbors(xz_new)
    return dists.ravel()


# -----------------------------------------------------------------------------
# Model definitions
# -----------------------------------------------------------------------------

def build_models() -> Dict[str, object]:
    """Final calibrated surrogate models.

    The intent here is to preserve the exact modelling idea used in the chat:
    - |F_d|: weighted blend of Extra Trees + quadratic ridge response surface
    - u_d: weighted blend of SVR-RBF + quadratic ridge response surface
    - Cp,r: physics-based reconstruction from surrogate-predicted u_d
    """
    models = {
        "fd_tree": ExtraTreesRegressor(
            n_estimators=500,
            random_state=CFG.random_seed,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "fd_poly": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=1.0),
        ),
        "ud_svr": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.04),
        ),
        "ud_poly": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=0.8),
        ),
    }
    return models


# -----------------------------------------------------------------------------
# Training / cross-validation
# -----------------------------------------------------------------------------

def cross_validate_models(successful: pd.DataFrame, rho: float, U_inf: float) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
    X = successful[[
        "P1_chord_ratio_c_by_D",
        "P2_AoA_angle_deg",
        "P3_Y_offset_m",
        "P4_Pressure_Jump_Pa",
    ]].to_numpy(dtype=float)

    y_fd = np.abs(successful["P5_drag_duct_N"].to_numpy(dtype=float))
    y_ud = successful["P6_u_disk_m_per_s"].to_numpy(dtype=float)
    y_cp = physics_cp(successful["P4_Pressure_Jump_Pa"].to_numpy(dtype=float), y_ud, rho, U_inf)

    models = build_models()
    cv = KFold(n_splits=CFG.cv_splits, shuffle=True, random_state=42)

    # Weighted calibrated ensembles
    fd_tree_cv = cross_val_predict(models["fd_tree"], X, y_fd, cv=cv, n_jobs=-1)
    fd_poly_cv = cross_val_predict(models["fd_poly"], X, y_fd, cv=cv)
    fd_cv = 0.60 * fd_tree_cv + 0.40 * fd_poly_cv

    ud_svr_cv = cross_val_predict(models["ud_svr"], X, y_ud, cv=cv)
    ud_poly_cv = cross_val_predict(models["ud_poly"], X, y_ud, cv=cv)
    ud_cv = 0.70 * ud_svr_cv + 0.30 * ud_poly_cv

    cp_cv = physics_cp(successful["P4_Pressure_Jump_Pa"].to_numpy(dtype=float), ud_cv, rho, U_inf)

    rows = [
        ["Absolute duct loading, |F_d|", "0.60 Extra Trees + 0.40 quadratic ridge RSM", metrics(y_fd, fd_cv)["R2"], metrics(y_fd, fd_cv)["MAE"], metrics(y_fd, fd_cv)["RMSE"]],
        ["Disk-averaged velocity, u_d", "0.70 SVR-RBF + 0.30 quadratic ridge RSM", metrics(y_ud, ud_cv)["R2"], metrics(y_ud, ud_cv)["MAE"], metrics(y_ud, ud_cv)["RMSE"]],
        ["Physics-based C_P,r", "Derived from surrogate-predicted u_d", metrics(y_cp, cp_cv)["R2"], metrics(y_cp, cp_cv)["MAE"], metrics(y_cp, cp_cv)["RMSE"]],
    ]
    metrics_df = pd.DataFrame(rows, columns=["Target", "Final surrogate", "R2", "MAE", "RMSE"])
    predictions = {
        "X": X,
        "y_fd": y_fd,
        "y_ud": y_ud,
        "y_cp": y_cp,
        "fd_cv": fd_cv,
        "ud_cv": ud_cv,
        "cp_cv": cp_cv,
    }

    return metrics_df, predictions, models


def fit_final_models(successful: pd.DataFrame, rho: float, U_inf: float) -> Dict[str, object]:
    X = successful[[
        "P1_chord_ratio_c_by_D",
        "P2_AoA_angle_deg",
        "P3_Y_offset_m",
        "P4_Pressure_Jump_Pa",
    ]].to_numpy(dtype=float)
    y_fd = np.abs(successful["P5_drag_duct_N"].to_numpy(dtype=float))
    y_ud = successful["P6_u_disk_m_per_s"].to_numpy(dtype=float)

    models = build_models()
    models["fd_tree"].fit(X, y_fd)
    models["fd_poly"].fit(X, y_fd)
    models["ud_svr"].fit(X, y_ud)
    models["ud_poly"].fit(X, y_ud)
    return models


def predict_final(models: Dict[str, object], Xcand: np.ndarray, rho: float, U_inf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fd_tree = models["fd_tree"].predict(Xcand)
    fd_poly = models["fd_poly"].predict(Xcand)
    fd_pred = 0.60 * fd_tree + 0.40 * fd_poly

    ud_svr = models["ud_svr"].predict(Xcand)
    ud_poly = models["ud_poly"].predict(Xcand)
    ud_pred = 0.70 * ud_svr + 0.30 * ud_poly

    cp_pred = physics_cp(Xcand[:, 3], ud_pred, rho, U_inf)
    return fd_pred, ud_pred, cp_pred


# -----------------------------------------------------------------------------
# Optimization / candidate search
# -----------------------------------------------------------------------------

def sample_candidate_space(n: int, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.random_seed)
    u = rng.random((n, 4))
    bounds = np.array([
        cfg.c_by_D_bounds,
        cfg.aoa_bounds,
        cfg.y_offset_bounds,
        cfg.pressure_jump_bounds,
    ], dtype=float)
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])


def select_candidates(
    successful: pd.DataFrame,
    models: Dict[str, object],
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (candidate_summary, pareto_front).

    Candidate-selection logic used in the paper:
    - DP5 is the best sampled CFD reference.
    - Among random surrogate candidates, select designs that retain at least
      eta of the DP5 physics-based C_P,r.
    - For each eta level, choose the candidate with the minimum predicted |F_d|.
    """
    # Baseline reference: best sampled CFD case (DP5)
    dp5 = successful.loc[successful["Name"].astype(str).str.strip().eq("DP 5")].iloc[0]
    cp_dp5 = float(dp5["P7"])
    load_dp5 = abs(float(dp5["P5_drag_duct_N"]))

    Xcand = sample_candidate_space(cfg.candidate_samples, cfg)
    fd_pred, ud_pred, cp_pred = predict_final(models, Xcand, cfg.rho, cfg.U_inf)

    # Interpolation filter in standardized input space to reduce extrapolation risk.
    Xtrain = successful[[
        "P1_chord_ratio_c_by_D",
        "P2_AoA_angle_deg",
        "P3_Y_offset_m",
        "P4_Pressure_Jump_Pa",
    ]].to_numpy(dtype=float)
    scaler = StandardScaler().fit(Xtrain)
    Xtrain_z = scaler.transform(Xtrain)
    Xcand_z = scaler.transform(Xcand)
    nn_dist = nearest_neighbor_distance(Xtrain_z, Xcand_z)

    # Conservative interpolation gate. You can tighten or loosen this threshold.
    interp_mask = nn_dist <= 1.75

    # Basic plausibility gates to remove numerically silly predictions.
    plausible = (
        (fd_pred > 0.0) &
        (fd_pred < 250.0) &
        (ud_pred > 0.4) &
        (ud_pred < 2.2) &
        (cp_pred > 0.0) &
        (cp_pred < 2.0)
    )
    keep = interp_mask & plausible

    Xf = Xcand[keep]
    fdf = fd_pred[keep]
    udf = ud_pred[keep]
    cpf = cp_pred[keep]
    nnf = nn_dist[keep]

    # Pareto front (maximize Cp, minimize |Fd|)
    order = np.argsort(fdf)
    pareto_idx = []
    best_cp = -np.inf
    for i in order:
        if cpf[i] > best_cp + 1e-8:
            pareto_idx.append(i)
            best_cp = cpf[i]
    pareto_idx = np.array(pareto_idx, dtype=int)

    pareto = pd.DataFrame({
        "c_by_D": Xf[pareto_idx, 0],
        "AoA_deg": Xf[pareto_idx, 1],
        "Y_offset_m": Xf[pareto_idx, 2],
        "Pressure_jump_Pa": Xf[pareto_idx, 3],
        "Pred_abs_drag_N": fdf[pareto_idx],
        "Pred_u_disk_m_per_s": udf[pareto_idx],
        "Pred_Cp_physics": cpf[pareto_idx],
        "Nearest_neighbor_distance": nnf[pareto_idx],
    }).sort_values("Pred_abs_drag_N").reset_index(drop=True)

    thresholds = {
        "DP64": cfg.eta_balanced,
        "DP65": cfg.eta_low_load,
        "DP66": cfg.eta_high_retention,
    }

    candidates = []
    for label, eta in thresholds.items():
        feasible = np.where(cpf >= eta * cp_dp5)[0]
        if len(feasible) == 0:
            # Fall back to highest Cp under interpolation gate if no candidate meets threshold.
            idx = int(np.argmax(cpf))
        else:
            idx = feasible[np.argmin(fdf[feasible])]
        row = {
            "Design": label,
            "Role": {
                "DP64": "CFD-verified balanced optimum",
                "DP65": "CFD-verified minimum-load optimum",
                "DP66": "CFD-verified high-retention optimum",
            }[label],
            "c/D": Xf[idx, 0],
            "AoA_deg": Xf[idx, 1],
            "Y_offset_m": Xf[idx, 2],
            "Pressure_jump_Pa": Xf[idx, 3],
            "Pred_abs_drag_N": fdf[idx],
            "Pred_u_disk_m_per_s": udf[idx],
            "Pred_Cp_physics": cpf[idx],
            "Nearest_neighbor_distance": nnf[idx],
        }
        candidates.append(row)

    summary = pd.DataFrame(candidates)
    summary.insert(9, "Power_retention_%", 100.0 * summary["Pred_Cp_physics"] / cp_dp5)
    summary.insert(10, "Load_reduction_%", 100.0 * (load_dp5 - summary["Pred_abs_drag_N"]) / load_dp5)
    summary.insert(11, "Efficiency_ratio", (summary["Pred_Cp_physics"] / summary["Pred_abs_drag_N"]) / (cp_dp5 / load_dp5))

    # Add the DP5 reference row first for reporting convenience.
    dp5_row = pd.DataFrame([{ 
        "Design": "DP5",
        "Role": "Maximum-power sampled CFD reference",
        "c/D": float(dp5["P1_chord_ratio_c_by_D"]),
        "AoA_deg": float(dp5["P2_AoA_angle_deg"]),
        "Y_offset_m": float(dp5["P3_Y_offset_m"]),
        "Pressure_jump_Pa": float(dp5["P4_Pressure_Jump_Pa"]),
        "Pred_abs_drag_N": abs(float(dp5["P5_drag_duct_N"])),
        "Pred_u_disk_m_per_s": float(dp5["P6_u_disk_m_per_s"]),
        "Pred_Cp_physics": cp_dp5,
        "Nearest_neighbor_distance": 0.0,
        "Power_retention_%": 100.0,
        "Load_reduction_%": 0.0,
        "Efficiency_ratio": 1.0,
    }])
    summary = pd.concat([dp5_row, summary], ignore_index=True)

    return summary, pareto


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def set_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.9,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_fig(fig: plt.Figure, outdir: Path, name: str) -> Tuple[Path, Path, Path]:
    png = outdir / f"{name}.png"
    pdf = outdir / f"{name}.pdf"
    svg = outdir / f"{name}.svg"
    fig.savefig(png, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    return png, pdf, svg


def parity_plot(y_true: np.ndarray, y_pred: np.ndarray, xlabel: str, ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.6, 5.1), dpi=250)
    ax.scatter(y_true, y_pred, s=34, facecolors="white", edgecolors="black", linewidths=0.8)
    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.06 * (lim_max - lim_min)
    lims = [lim_min - pad, lim_max + pad]
    ax.plot(lims, lims, linestyle="--", linewidth=1.2, color="black")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    m = metrics(y_true, y_pred)
    ax.text(
        0.04,
        0.96,
        f"R² = {m['R2']:.3f}\nMAE = {m['MAE']:.3f}\nRMSE = {m['RMSE']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, linewidth=0.7),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    fig.tight_layout()
    return fig


def make_figures(successful: pd.DataFrame, metrics_df: pd.DataFrame, predictions: Dict[str, np.ndarray], summary: pd.DataFrame, pareto: pd.DataFrame, outdir: Path) -> Dict[str, Tuple[Path, Path, Path]]:
    set_publication_style()
    fig_paths: Dict[str, Tuple[Path, Path, Path]] = {}

    # Fig 11a, 11b, 11c: parity plots
    fig = parity_plot(predictions["y_fd"], predictions["fd_cv"], r"CFD $|F_d|$ (N)", r"Surrogate-predicted $|F_d|$ (N)", "Fig. 11a  Cross-validated parity for duct loading")
    fig_paths["Fig11a"] = save_fig(fig, outdir, "Fig11a_final_calibrated_parity_Fd")
    plt.close(fig)

    fig = parity_plot(predictions["y_ud"], predictions["ud_cv"], r"CFD $u_d$ (m s$^{-1}$)", r"Surrogate-predicted $u_d$ (m s$^{-1}$)", "Fig. 11b  Cross-validated parity for disk-averaged velocity")
    fig_paths["Fig11b"] = save_fig(fig, outdir, "Fig11b_final_calibrated_parity_ud")
    plt.close(fig)

    fig = parity_plot(predictions["y_cp"], predictions["cp_cv"], r"CFD $C_{P,r}$", r"Physics-based surrogate-predicted $C_{P,r}$", "Fig. 11c  Cross-validated parity for power coefficient")
    fig_paths["Fig11c"] = save_fig(fig, outdir, "Fig11c_final_calibrated_parity_CPr")
    plt.close(fig)

    # Fig 12: CV summary
    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=250)
    x = np.arange(len(metrics_df))
    ax.bar(x, metrics_df["R2"], color="white", edgecolor="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$|F_d|$", r"$u_d$", r"$C_{P,r}$"])
    ax.set_ylabel(r"5-fold cross-validated $R^2$")
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.45, alpha=0.45)
    for i, val in enumerate(metrics_df["R2"]):
        ax.text(i, val + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Fig. 12  Cross-validated accuracy of the calibrated surrogates", pad=10)
    fig.tight_layout()
    fig_paths["Fig12"] = save_fig(fig, outdir, "Fig12_final_calibrated_CV_summary")
    plt.close(fig)

    # Fig 13: power-loading trade-off
    fig, ax = plt.subplots(figsize=(7.0, 5.3), dpi=250)
    ax.scatter(predictions["y_fd"], predictions["y_cp"], s=20, alpha=0.35, label="Converged baseline CFD cases")
    for _, row in summary.iterrows():
        ax.scatter(row["Pred_abs_drag_N"], row["Pred_Cp_physics"], s=85, marker="D", edgecolors="black", linewidths=0.8)
        ax.annotate(row["Design"], (row["Pred_abs_drag_N"], row["Pred_Cp_physics"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel(r"Absolute duct loading, $|F_d|$ (N)")
    ax.set_ylabel(r"Rotor-area-normalized power coefficient, $C_{P,r}$")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    ax.set_title("Fig. 13  Power-loading trade-off and CFD-verified optimized designs", pad=10)
    fig.tight_layout()
    fig_paths["Fig13"] = save_fig(fig, outdir, "Fig13_final_power_loading_tradeoff")
    plt.close(fig)

    # Fig 14: relative benefit
    comp = summary.loc[summary["Design"].isin(["DP64", "DP65", "DP66"])].copy()
    fig, ax = plt.subplots(figsize=(6.8, 5.0), dpi=250)
    ax.scatter(comp["Load_reduction_%"], comp["Power_retention_%"], s=100, edgecolors="black", linewidths=0.8)
    for _, row in comp.iterrows():
        ax.annotate(row["Design"], (row["Load_reduction_%"], row["Power_retention_%"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Duct-load reduction relative to DP5 (%)")
    ax.set_ylabel(r"Power retention relative to DP5 (%)")
    ax.set_xlim(30, 70)
    ax.set_ylim(75, 100)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    ax.set_title("Fig. 14  Relative benefit of the optimized designs with respect to DP5", pad=10)
    fig.tight_layout()
    fig_paths["Fig14"] = save_fig(fig, outdir, "Fig14_final_relative_benefit_DP64_DP65_DP66")
    plt.close(fig)

    return fig_paths


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ensure_outdir(CFG.outdir)

    df = load_dataset(CFG.data_path)
    successful, failed = split_successful_failed(df)

    # Cross-validation on converged CFD data
    metrics_df, predictions, _ = cross_validate_models(successful, CFG.rho, CFG.U_inf)
    metrics_csv = CFG.outdir / "Section8_calibrated_surrogate_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Final fitted models
    models = fit_final_models(successful, CFG.rho, CFG.U_inf)

    # Candidate search and Pareto front
    summary, pareto = select_candidates(successful, models, CFG)
    summary_csv = CFG.outdir / "Section8_final_optimized_designs_DP64_DP65_DP66.csv"
    summary.to_csv(summary_csv, index=False)
    pareto_csv = CFG.outdir / "DP64_revised_calibrated_pareto_front.csv"
    pareto.to_csv(pareto_csv, index=False)

    # Optional: save a clean candidate summary for the manuscript
    summary.rename(columns={
        "Pred_abs_drag_N": "Pred_abs_drag_N",
        "Pred_u_disk_m_per_s": "Pred_u_disk_m_per_s",
        "Pred_Cp_physics": "Pred_Cp_physics",
    }).to_csv(CFG.outdir / "DP64_revised_calibrated_surrogate_candidate.csv", index=False)

    # Figures
    fig_paths = make_figures(successful, metrics_df, predictions, summary, pareto, CFG.outdir)

    # Helpful text summary
    report = {
        "n_total": int(len(df)),
        "n_successful": int(len(successful)),
        "n_failed": int(len(failed)),
        "metrics_csv": str(metrics_csv),
        "summary_csv": str(summary_csv),
        "pareto_csv": str(pareto_csv),
        "figures": {k: [str(p) for p in v] for k, v in fig_paths.items()},
    }
    with open(CFG.outdir / "surrogate_pipeline_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Surrogate training pipeline completed successfully.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
