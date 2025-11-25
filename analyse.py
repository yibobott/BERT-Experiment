import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

RESULT_DIR = "./result-nsp-1"
BASELINE = "nsp_baseline"
IMPROVE = "nsp_improve"
BASELINE_CSV = RESULT_DIR + "/" + BASELINE + "/" + BASELINE + "_loss_all_seeds.csv"
IMPROVE_CSV = RESULT_DIR + "/" + IMPROVE + "/" + IMPROVE + "_loss_all_seeds.csv"

def load_losses(path):
    df = pd.read_csv(path)
    # safety checks
    required_cols = {"step", "loss", "type", "seed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df

def summarize_final_eval(df):
    """
    Return per-seed final eval loss, mean, std
    """
    eval_df = df[df["type"] == "eval"].copy()
    # take last eval point per seed
    last_eval = eval_df.sort_values("step").groupby("seed").tail(1)
    per_seed = last_eval[["seed", "step", "loss"]].sort_values("seed")
    mean = per_seed["loss"].mean()
    std = per_seed["loss"].std(ddof=1)
    return per_seed, mean, std

def summarize_final_train(df):
    """
    Return per-seed final train loss, mean, std
    """
    train_df = df[df["type"] == "train"].copy()
    last_train = train_df.sort_values("step").groupby("seed").tail(1)
    per_seed = last_train[["seed", "step", "loss"]].sort_values("seed")
    mean = per_seed["loss"].mean()
    std = per_seed["loss"].std(ddof=1)
    return per_seed, mean, std

def curve_mean_std(df, loss_type="eval"):
    """
    Return mean and std of loss curve
    """
    sub = df[df["type"] == loss_type].copy()
    # 把数据 reshape 成 (step × seed) 的矩阵
    pivot = sub.pivot_table(index="step", columns="seed", values="loss")
    # 在同一行上聚合: 对同一个 step、跨不同 seed 的 loss 求平均和标准差
    mean = pivot.mean(axis=1)
    std = pivot.std(axis=1, ddof=1)
    return mean, std

def plot_mean_std(mean, std, label, title, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(mean.index, mean.values, label=label)
    plt.fill_between(mean.index, mean-std, mean+std, alpha=0.2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_compare(baseline_mean, baseline_std, improve_mean, improve_std, title, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(baseline_mean.index, baseline_mean.values, label= BASELINE + " mean")
    plt.fill_between(baseline_mean.index, baseline_mean-baseline_std, baseline_mean+baseline_std, alpha=0.2)

    plt.plot(improve_mean.index, improve_mean.values, label= IMPROVE + " mean")
    plt.fill_between(improve_mean.index, improve_mean-improve_std, improve_mean+improve_std, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    baseline_df = load_losses(BASELINE_CSV)
    improve_df = load_losses(IMPROVE_CSV)

    print("\n==============================")
    print("1) Final Eval Loss (per seed)")
    print("==============================")
    baseline_eval_per_seed, baseline_eval_mean, baseline_eval_std = summarize_final_eval(baseline_df)
    improve_eval_per_seed, improve_eval_mean, improve_eval_std = summarize_final_eval(improve_df)

    print("\n" + BASELINE + " final eval per seed:")
    print(baseline_eval_per_seed.to_string(index=False))
    print(f"{BASELINE} final eval mean±std: {baseline_eval_mean:.4f} ± {baseline_eval_std:.4f}")

    print("\n" + IMPROVE + " final eval per seed:")
    print(improve_eval_per_seed.to_string(index=False))
    print(f"{IMPROVE} final eval mean±std: {improve_eval_mean:.4f} ± {improve_eval_std:.4f}")

    diff = improve_eval_mean - baseline_eval_mean
    print(f"\nMean difference ({IMPROVE} - {BASELINE}): {diff:.4f}")
    if diff > 0:
        print("→" + BASELINE + " is slightly better (lower eval loss) in this setup.")
    else:
        print("→" + IMPROVE + " is slightly better (lower eval loss) in this setup.")

    # simple t-test on final eval losses (small n, just for reference)
    t_stat, p_val = ttest_ind(
        improve_eval_per_seed["loss"].values,
        baseline_eval_per_seed["loss"].values,
        equal_var=False
    )
    print(f"\nT-test on final eval losses: t={t_stat:.3f}, p={p_val:.3f}")
    print("(Note: n=3 per group, so treat p-value as indicative only.)")

    print("\n==============================")
    print("2) Final Train Loss (per seed)")
    print("==============================")
    baseline_train_per_seed, baseline_train_mean, baseline_train_std = summarize_final_train(baseline_df)
    improve_train_per_seed, improve_train_mean, improve_train_std = summarize_final_train(improve_df)

    print("\n" + BASELINE + " final train per seed:")
    print(baseline_train_per_seed.to_string(index=False))
    print(f"{BASELINE} final train mean±std: {baseline_train_mean:.4f} ± {baseline_train_std:.4f}")

    print("\n" + IMPROVE + " final train per seed:")
    print(improve_train_per_seed.to_string(index=False))
    print(f"{IMPROVE} final train mean±std: {improve_train_mean:.4f} ± {improve_train_std:.4f}")

    print("\n==============================")
    print("3) Curve Mean±Std + Plots")
    print("==============================")

    # eval curves
    baseline_eval_curve_mean, baseline_eval_curve_std = curve_mean_std(baseline_df, "eval")
    improve_eval_curve_mean, improve_eval_curve_std = curve_mean_std(improve_df, "eval")

    plot_mean_std(baseline_eval_curve_mean, baseline_eval_curve_std,
                  BASELINE + " eval", BASELINE + " Eval Loss (mean±std)",
                  RESULT_DIR + "/" + BASELINE + "_eval_mean_std.png")
    plot_mean_std(improve_eval_curve_mean, improve_eval_curve_std,
                  IMPROVE + " eval", IMPROVE + " Eval Loss (mean±std)",
                  RESULT_DIR + "/" + IMPROVE + "_eval_mean_std.png")
    plot_compare(baseline_eval_curve_mean, baseline_eval_curve_std,
                 improve_eval_curve_mean, improve_eval_curve_std,
                 BASELINE + " vs " + IMPROVE + " Eval Loss (mean±std)",
                 RESULT_DIR + "/" + BASELINE + "_vs_" + IMPROVE + "_compare_eval_mean_std.png")

    # train curves
    baseline_train_curve_mean, baseline_train_curve_std = curve_mean_std(baseline_df, "train")
    improve_train_curve_mean, improve_train_curve_std = curve_mean_std(improve_df, "train")

    plot_mean_std(baseline_train_curve_mean, baseline_train_curve_std,
                  BASELINE + " train", BASELINE + " Train Loss (mean±std)",
                  RESULT_DIR + "/" + BASELINE + "_train_mean_std.png")
    plot_mean_std(improve_train_curve_mean, improve_train_curve_std,
                  IMPROVE + " train", IMPROVE + " Train Loss (mean±std)",
                  RESULT_DIR + "/" + IMPROVE + "_train_mean_std.png")
    plot_compare(baseline_train_curve_mean, baseline_train_curve_std,
                 improve_train_curve_mean, improve_train_curve_std,
                 BASELINE + " vs " + IMPROVE + " Train Loss (mean±std)",
                 RESULT_DIR + "/" + BASELINE + "_vs_" + IMPROVE + "_compare_train_mean_std.png")

    print("Saved plots to " + RESULT_DIR + "/")
    print("- " + BASELINE + "_eval_mean_std.png")
    print("- " + IMPROVE + "_eval_mean_std.png")
    print("- " + BASELINE + "_vs_" + IMPROVE + "_compare_eval_mean_std.png")
    print("- " + BASELINE + "_train_mean_std.png")
    print("- " + IMPROVE + "_train_mean_std.png")
    print("- " + BASELINE + "_vs_" + IMPROVE + "_compare_train_mean_std.png")

    print("\n==============================")
    print("4) Stability / Variance summary")
    print("==============================")
    print(f"Eval std ({BASELINE}):  {baseline_eval_std:.4f}")
    print(f"Eval std ({IMPROVE}): {improve_eval_std:.4f}")
    if improve_eval_std > baseline_eval_std:
        print("→" + IMPROVE + " masking is less stable (higher variance), as expected.")
    else:
        print("→" + IMPROVE + " masking is equally/more stable in this setup.")

if __name__ == "__main__":
    main()
