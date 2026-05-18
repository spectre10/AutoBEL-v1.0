"""Headless runner for AutoBEL.

Runs the same pipeline as Control_Pannel.ipynb but with no GUI: matplotlib
is forced to the Agg backend, plt.show() is a no-op, and after each stage
the script prints summary statistics computed from the artifacts the
pipeline writes to output/. Intended for `python run_autobel.py` over SSH.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *a, **k: None

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from source_code.Auto_BEL import Auto_BEL
from source_code.global_param_posterior import (
    run_global_param_analysis,
)


MODEL_NAMES = ["thickness"]
MODEL_TYPES = [1]
X_DIM, Y_DIM, Z_DIM = 200, 100, 1
GRID_H_RESOLUTION = 250 * 250
PRI_M_SAMPLES_DIR = "input/prior_samples/"
SAMPLES_SIZE = 250
MGL = "input/thickness_mgl.txt"
DOBS_FILE = "input/thickness_obs"
OUTPUT_DIR = "output/"


def ensure_output_dirs():
    for sub in ("model", "data", "prediction", "figures"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def describe_array(name, arr):
    arr = np.asarray(arr)
    flat = arr.ravel()
    print(
        f"{name:<32} shape={str(arr.shape):<18} "
        f"mean={flat.mean():>10.4g}  std={flat.std():>10.4g}  "
        f"min={flat.min():>10.4g}  max={flat.max():>10.4g}"
    )


def percentiles(name, arr, ps=(5, 25, 50, 75, 95)):
    arr = np.asarray(arr).ravel()
    vals = np.percentile(arr, ps)
    parts = "  ".join(f"p{p}={v:.4g}" for p, v in zip(ps, vals))
    print(f"{name:<32} {parts}")


def print_prior_model_stats():
    banner("Prior model samples")
    for name in MODEL_NAMES:
        arr = np.load(os.path.join(PRI_M_SAMPLES_DIR, name + ".npy"))
        describe_array(f"prior[{name}]", arr)
        percentiles(f"prior[{name}] percentiles", arr)


def print_prior_prediction_stats():
    banner("Prior prediction (GIIP)")
    giip = np.load(os.path.join(OUTPUT_DIR, "prediction/GIIP_pri.npy"))
    describe_array("GIIP_pri", giip)
    percentiles("GIIP_pri percentiles", giip)


def print_dimension_reduction_stats():
    banner("Dimension reduction")
    for name in MODEL_NAMES:
        pcscr = np.load(os.path.join(OUTPUT_DIR, f"model/{name}_pcscr_pri.npy"))
        eigvec = np.load(os.path.join(OUTPUT_DIR, f"model/{name}_eigvec_pri.npy"))
        var = pcscr.var(axis=0)
        cum = np.cumsum(var) / var.sum()
        n90 = int(np.searchsorted(cum, 0.90) + 1)
        n95 = int(np.searchsorted(cum, 0.95) + 1)
        print(f"model[{name}] PC scores  shape={pcscr.shape}  eigvecs={eigvec.shape}")
        print(f"  PCs to reach 90% var: {n90}    95% var: {n95}")
        print(
            f"  variance explained by first 5 PCs: "
            f"{(cum[:5] * 100).round(2).tolist()}"
        )

        d_pcscr = np.load(os.path.join(OUTPUT_DIR, f"data/dpcscr_pri_{name}.npy"))
        d_obs = np.load(os.path.join(OUTPUT_DIR, f"data/dpcscr_obs_{name}.npy"))
        print(f"data[{name}] prior PC scores shape={d_pcscr.shape}  obs shape={d_obs.shape}")


def print_sensitivity_stats():
    banner("Global Sensitivity Analysis (DGSA)")
    for name in MODEL_NAMES:
        sa = np.load(os.path.join(OUTPUT_DIR, f"data/SA_measure_{name}.npy"))
        sa_col = sa[:, 0] if sa.ndim > 1 else sa
        sensitive = np.argwhere(sa_col > 1.0).ravel() + 1
        print(f"model[{name}] SA measures (PC -> sensitivity):")
        for i, v in enumerate(sa_col, start=1):
            marker = " *sensitive*" if v > 1.0 else ""
            print(f"  PC{i:>2}: {v:.4f}{marker}")
        print(f"  Sensitive PCs (>1.0): {sensitive.tolist()}")


def print_posterior_stats():
    banner("Posterior model & prediction")
    for name in MODEL_NAMES:
        m_pri = np.load(os.path.join(PRI_M_SAMPLES_DIR, name + ".npy"))
        m_post = np.load(os.path.join(OUTPUT_DIR, f"model/{name}_model_post.npy"))
        describe_array(f"prior[{name}]", m_pri)
        describe_array(f"posterior[{name}]", m_post)

        pri_std = m_pri.std(axis=0).mean()
        post_std = m_post.std(axis=0).mean()
        reduction = (1 - post_std / pri_std) * 100 if pri_std > 0 else 0.0
        print(
            f"  cell-wise mean std  prior={pri_std:.4g}  posterior={post_std:.4g}  "
            f"reduction={reduction:.2f}%"
        )

    giip_pri = np.load(os.path.join(OUTPUT_DIR, "prediction/GIIP_pri.npy"))
    giip_post = np.load(os.path.join(OUTPUT_DIR, "prediction/GIIP_post.npy"))
    describe_array("GIIP_pri", giip_pri)
    describe_array("GIIP_post", giip_post)
    percentiles("GIIP_pri percentiles", giip_pri)
    percentiles("GIIP_post percentiles", giip_post)
    pri_spread = giip_pri.std()
    post_spread = giip_post.std()
    reduction = (1 - post_spread / pri_spread) * 100 if pri_spread > 0 else 0.0
    print(f"  GIIP std reduction: {reduction:.2f}%")


def main():
    ensure_output_dirs()

    banner("Running Auto_BEL pipeline (headless)")
    Auto_BEL(
        PRI_M_SAMPLES_DIR,
        MODEL_NAMES,
        MODEL_TYPES,
        MGL,
        SAMPLES_SIZE,
        X_DIM,
        Y_DIM,
        Z_DIM,
        GRID_H_RESOLUTION,
        DOBS_FILE,
    )
    plt.close("all")

    print_prior_model_stats()
    print_prior_prediction_stats()
    print_dimension_reduction_stats()
    print_sensitivity_stats()
    print_posterior_stats()

    banner("Global parameter posterior analysis")
    run_global_param_analysis(
        mgl_file=MGL,
        model_name=MODEL_NAMES[0],
        output_dir=OUTPUT_DIR,
        sensitivity_threshold=1.0,
        plot_results=False,
    )
    plt.close("all")

    banner("Done")


if __name__ == "__main__":
    main()
