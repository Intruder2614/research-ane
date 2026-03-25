"""
analysis/anova_decomposition.py
================================
Multi-factor ANOVA to decompose variance in inference latency across
the experimental factors: precision, model_size_bin, compute_unit, thermal_bin.

Partial eta² tells you what fraction of latency variance is explained
by each factor, controlling for the others. This answers:
  "Is the precision effect (FP32 vs INT8) the dominant driver of speedup,
   or does the model-size effect (small vs large) swamp it?"

A large eta² for model_size_bin × precision interaction would support
the hypothesis that precision effects are NOT uniform across model sizes
— i.e., the cache effect is real.

Uses pingouin for readable ANOVA output, with statsmodels as fallback.

Usage:
    python analysis/anova_decomposition.py
    python analysis/anova_decomposition.py --input data/processed/master.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ── binning helpers ───────────────────────────────────────────────────────────

def add_size_bins(df: pd.DataFrame, bins_mb: list[float]) -> pd.DataFrame:
    """Add model_size_bin column based on MB thresholds."""
    labels = []
    for i in range(len(bins_mb) - 1):
        labels.append(f"{bins_mb[i]}-{bins_mb[i+1]}MB")
    df = df.copy()
    df["model_size_bin"] = pd.cut(
        df["model_size_mb"],
        bins=bins_mb,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    return df


def add_thermal_bins(df: pd.DataFrame, temp_bins: list[float]) -> pd.DataFrame:
    """Add thermal_bin column from max thermal reading."""
    df = df.copy()
    # Use the run's max thermal sample, or assign 'cold' if unavailable
    if "thermal_max_celsius" not in df.columns:
        df["thermal_bin"] = "unknown"
        return df
    labels = ["cold", "warm", "hot"]
    df["thermal_bin"] = pd.cut(
        df["thermal_max_celsius"],
        bins=temp_bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    return df


# ── ANOVA via pingouin ────────────────────────────────────────────────────────

def run_anova_pingouin(df: pd.DataFrame, dv: str, factors: list[str]) -> pd.DataFrame:
    """
    Run a between-subjects ANOVA using pingouin.
    Returns the ANOVA table with partial eta² column.
    """
    # Drop rows with missing values in any factor or DV
    cols = [dv] + factors
    df_clean = df[cols].dropna()
    print(f"  ANOVA sample: {len(df_clean)} observations")

    aov = pg.anova(data=df_clean, dv=dv, between=factors, detailed=True)
    return aov


# ── ANOVA via statsmodels (fallback) ─────────────────────────────────────────

def run_anova_statsmodels(df: pd.DataFrame, dv: str, factors: list[str]) -> pd.DataFrame:
    """
    Run Type III SS ANOVA via statsmodels.
    Returns a results DataFrame with partial eta².
    """
    formula = f"Q('{dv}') ~ " + " * ".join([f"C(Q('{f}'))" for f in factors])
    cols = [dv] + factors
    df_clean = df[cols].dropna()

    model = smf.ols(formula, data=df_clean).fit()
    aov_table = sm_anova_lm(model, typ=3)  # Type III SS

    # Compute partial eta²: SS_effect / (SS_effect + SS_residual)
    ss_residual = aov_table.loc["Residual", "sum_sq"]
    aov_table["partial_eta_sq"] = aov_table["sum_sq"] / (aov_table["sum_sq"] + ss_residual)
    return aov_table


def sm_anova_lm(model, typ: int):
    """Thin wrapper to avoid import repetition."""
    from statsmodels.stats.anova import anova_lm
    return anova_lm(model, typ=typ)


# ── effect size interpretation ────────────────────────────────────────────────

def interpret_eta_sq(eta: float) -> str:
    """Cohen (1988) conventions for partial eta²."""
    if eta < 0.01:  return "negligible"
    if eta < 0.06:  return "small"
    if eta < 0.14:  return "medium"
    return "large"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-factor ANOVA on latency data")
    parser.add_argument("--input",  default="data/processed/master.csv")
    parser.add_argument("--output", default="data/results/anova_table.csv")
    parser.add_argument("--dv",     default="median_us", help="Dependent variable column")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Add binned columns
    df = add_size_bins(df, [0, 4, 12, 100])
    df = add_thermal_bins(df, [0, 38, 42, 55])

    # Filter to the primary ANE condition for main ANOVA
    df_ane = df[df["compute_unit"] == "cpuAndNeuralEngine"].copy()
    print(f"ANE condition rows: {len(df_ane)}")

    factors = ["precision", "model_size_bin"]
    dv = args.dv

    print(f"\nRunning ANOVA: {dv} ~ {' × '.join(factors)}")
    print("="*60)

    aov_table = None
    if PINGOUIN_AVAILABLE:
        print("Using pingouin...")
        try:
            aov_table = run_anova_pingouin(df_ane, dv, factors)
        except Exception as e:
            print(f"  pingouin failed: {e}")

    if aov_table is None and STATSMODELS_AVAILABLE:
        print("Falling back to statsmodels...")
        try:
            aov_table = run_anova_statsmodels(df_ane, dv, factors)
        except Exception as e:
            print(f"  statsmodels failed: {e}")

    if aov_table is None:
        print("ERROR: Neither pingouin nor statsmodels available.")
        print("Install: pip install pingouin statsmodels")
        return

    print("\nANOVA results:")
    print(aov_table.to_string())

    # Summarise partial eta² values
    print("\nPartial η² (effect size):")
    eta_col = "np2" if "np2" in aov_table.columns else "partial_eta_sq"
    if eta_col in aov_table.columns:
        for source, row in aov_table.iterrows():
            eta = row.get(eta_col)
            if pd.notna(eta) and str(source) not in ("Residual", "Intercept"):
                print(f"  {str(source):40s}  η²={eta:.4f}  [{interpret_eta_sq(eta)}]")

    # Research interpretation
    print("\nResearch interpretation:")
    print("  If 'precision' η² >> 'model_size_bin' η²:")
    print("    → Arithmetic effect dominates (conventional narrative)")
    print("  If 'precision × model_size_bin' interaction η² is large:")
    print("    → Cache effect is real — precision benefit depends on model size")
    print("  If 'model_size_bin' η² alone is large:")
    print("    → Memory-bound behaviour — supports cache hypothesis")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    aov_table.to_csv(args.output)
    print(f"\nANOVA table saved to {args.output}")


if __name__ == "__main__":
    main()
