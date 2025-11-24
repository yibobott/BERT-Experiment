import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

RESULT_DIR = "./result"
STATIC_CSV = RESULT_DIR + "/static/" + "static_loss_all_seeds.csv"
DYNAMIC_CSV = RESULT_DIR + "/dynamic/" + "dynamic_loss_all_seeds.csv"

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

def plot_compare(static_mean, static_std, dynamic_mean, dynamic_std, title, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(static_mean.index, static_mean.values, label="Static mean")
    plt.fill_between(static_mean.index, static_mean-static_std, static_mean+static_std, alpha=0.2)

    plt.plot(dynamic_mean.index, dynamic_mean.values, label="Dynamic mean")
    plt.fill_between(dynamic_mean.index, dynamic_mean-dynamic_std, dynamic_mean+dynamic_std, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    static_df = load_losses(STATIC_CSV)
    dynamic_df = load_losses(DYNAMIC_CSV)

    print("\n==============================")
    print("1) Final Eval Loss (per seed)")
    print("==============================")
    s_eval_per_seed, s_eval_mean, s_eval_std = summarize_final_eval(static_df)
    d_eval_per_seed, d_eval_mean, d_eval_std = summarize_final_eval(dynamic_df)

    print("\nStatic final eval per seed:")
    print(s_eval_per_seed.to_string(index=False))
    print(f"Static final eval mean±std: {s_eval_mean:.4f} ± {s_eval_std:.4f}")

    print("\nDynamic final eval per seed:")
    print(d_eval_per_seed.to_string(index=False))
    print(f"Dynamic final eval mean±std: {d_eval_mean:.4f} ± {d_eval_std:.4f}")

    diff = d_eval_mean - s_eval_mean
    print(f"\nMean difference (Dynamic - Static): {diff:.4f}")
    if diff > 0:
        print("→ Static is slightly better (lower eval loss) in this setup.")
    else:
        print("→ Dynamic is slightly better (lower eval loss) in this setup.")

    # simple t-test on final eval losses (small n, just for reference)
    t_stat, p_val = ttest_ind(
        d_eval_per_seed["loss"].values,
        s_eval_per_seed["loss"].values,
        equal_var=False
    )
    print(f"\nT-test on final eval losses: t={t_stat:.3f}, p={p_val:.3f}")
    print("(Note: n=3 per group, so treat p-value as indicative only.)")

    print("\n==============================")
    print("2) Final Train Loss (per seed)")
    print("==============================")
    s_train_per_seed, s_train_mean, s_train_std = summarize_final_train(static_df)
    d_train_per_seed, d_train_mean, d_train_std = summarize_final_train(dynamic_df)

    print("\nStatic final train per seed:")
    print(s_train_per_seed.to_string(index=False))
    print(f"Static final train mean±std: {s_train_mean:.4f} ± {s_train_std:.4f}")

    print("\nDynamic final train per seed:")
    print(d_train_per_seed.to_string(index=False))
    print(f"Dynamic final train mean±std: {d_train_mean:.4f} ± {d_train_std:.4f}")

    print("\n==============================")
    print("3) Curve Mean±Std + Plots")
    print("==============================")

    # eval curves
    s_eval_curve_mean, s_eval_curve_std = curve_mean_std(static_df, "eval")
    d_eval_curve_mean, d_eval_curve_std = curve_mean_std(dynamic_df, "eval")

    plot_mean_std(s_eval_curve_mean, s_eval_curve_std,
                  "Static eval", "Static Eval Loss (mean±std)",
                  RESULT_DIR + "/static_eval_mean_std.png")
    plot_mean_std(d_eval_curve_mean, d_eval_curve_std,
                  "Dynamic eval", "Dynamic Eval Loss (mean±std)",
                  RESULT_DIR + "/dynamic_eval_mean_std.png")
    plot_compare(s_eval_curve_mean, s_eval_curve_std,
                 d_eval_curve_mean, d_eval_curve_std,
                 "Static vs Dynamic Eval Loss (mean±std)",
                 RESULT_DIR + "/compare_eval_mean_std.png")

    # train curves
    s_train_curve_mean, s_train_curve_std = curve_mean_std(static_df, "train")
    d_train_curve_mean, d_train_curve_std = curve_mean_std(dynamic_df, "train")

    plot_mean_std(s_train_curve_mean, s_train_curve_std,
                  "Static train", "Static Train Loss (mean±std)",
                  RESULT_DIR + "/static_train_mean_std.png")
    plot_mean_std(d_train_curve_mean, d_train_curve_std,
                  "Dynamic train", "Dynamic Train Loss (mean±std)",
                  RESULT_DIR + "/dynamic_train_mean_std.png")
    plot_compare(s_train_curve_mean, s_train_curve_std,
                 d_train_curve_mean, d_train_curve_std,
                 "Static vs Dynamic Train Loss (mean±std)",
                 RESULT_DIR + "/compare_train_mean_std.png")

    print("Saved plots to " + RESULT_DIR + "/")
    print("- static_eval_mean_std.png")
    print("- dynamic_eval_mean_std.png")
    print("- compare_eval_mean_std.png")
    print("- static_train_mean_std.png")
    print("- dynamic_train_mean_std.png")
    print("- compare_train_mean_std.png")

    print("\n==============================")
    print("4) Stability / Variance summary")
    print("==============================")
    print(f"Eval std (static):  {s_eval_std:.4f}")
    print(f"Eval std (dynamic): {d_eval_std:.4f}")
    if d_eval_std > s_eval_std:
        print("→ Dynamic masking is less stable (higher variance), as expected.")
    else:
        print("→ Dynamic masking is equally/more stable in this setup.")

if __name__ == "__main__":
    main()
