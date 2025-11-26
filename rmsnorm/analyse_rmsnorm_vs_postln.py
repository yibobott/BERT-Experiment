import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


plt.switch_backend("Agg")  # 避免某些环境下无显示错误

# 手动填入你保存结果的目录（对应不同时间跑的实验）
OUTPUT_DIR = "./result-rmsnorm-3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSTLN_DIR = "result-rmsnorm-20251126-152638-postln-dynamic"
RMSNORM_DIR = "result-rmsnorm-20251126-145941-rmsnorm-dynamic"
MASKING_TYPE = "dynamic"

postln_csv = f"{OUTPUT_DIR}/{POSTLN_DIR}/postln_{MASKING_TYPE}_loss_all_seeds.csv"
rmsnorm_csv = f"{OUTPUT_DIR}/{RMSNORM_DIR}/rmsnorm_{MASKING_TYPE}_loss_all_seeds.csv"


def load_losses(path):
    df = pd.read_csv(path)
    # safety checks
    required_cols = {"step", "loss", "type", "seed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def _summarize_final_by_type(df: pd.DataFrame, loss_type: str):
    """
    通用：返回某种 loss_type ('eval' / 'train') 的
    per-seed 最后一个 loss 的表 + mean + std
    """
    sub = df[df["type"] == loss_type].copy()
    last = sub.sort_values("step").groupby("seed").tail(1)
    per_seed = last[["seed", "step", "loss"]].sort_values("seed")
    mean = per_seed["loss"].mean()
    std = per_seed["loss"].std(ddof=1)
    return per_seed, mean, std


def summarize_final_eval(df: pd.DataFrame):
    """Return per-seed final eval loss, mean, std."""
    return _summarize_final_by_type(df, loss_type="eval")


def summarize_final_train(df: pd.DataFrame):
    """Return per-seed final train loss, mean, std."""
    return _summarize_final_by_type(df, loss_type="train")


def _plot_loss_curves_by_type(df, loss_type: str, title: str, out_png: str, ylabel: str):
    plt.figure()
    mask = df["type"] == loss_type
    for seed, sub in df[mask].groupby("seed"):
        sub_sorted = sub.sort_values("step")
        plt.plot(sub_sorted["step"], sub_sorted["loss"], label=f"seed={seed}")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png))
    plt.close()
    print(f"[Saved plot] {out_png}")


def plot_eval_curves(df, title, out_png):
    _plot_loss_curves_by_type(df, loss_type="eval", title=title, out_png=out_png, ylabel="Eval loss")


def plot_train_curves(df, title, out_png):
    """画每个 seed 的 train loss 曲线。"""
    _plot_loss_curves_by_type(df, loss_type="train", title=title, out_png=out_png, ylabel="Train loss")


def plot_mean_std(df, title, out_png):
    eval_df = df[df["type"] == "eval"].copy()
    grouped = eval_df.groupby("step")
    steps = sorted(grouped.groups.keys())
    means = []
    stds = []
    for s in steps:
        vals = grouped.get_group(s)["loss"].values
        means.append(vals.mean())
        stds.append(vals.std(ddof=1))

    steps = np.array(steps)
    means = np.array(means)
    stds = np.array(stds)

    plt.figure()
    plt.plot(steps, means, label="mean")
    plt.fill_between(steps, means - stds, means + stds, alpha=0.3, label="±1 std")
    plt.xlabel("Step")
    plt.ylabel("Eval loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png))
    plt.close()
    print(f"[Saved plot] {out_png}")


def plot_compare_mean_std(df1, df2, title, out_png):
    # 计算 df1 的 mean / std
    eval_df1 = df1[df1["type"] == "eval"].copy()
    grouped1 = eval_df1.groupby("step")
    steps1 = sorted(grouped1.groups.keys())
    means1, stds1 = [], []
    for s in steps1:
        vals = grouped1.get_group(s)["loss"].values
        means1.append(vals.mean())
        stds1.append(vals.std(ddof=1))

    steps1 = np.array(steps1)
    means1 = np.array(means1)
    stds1 = np.array(stds1)

    # 计算 df2 的 mean / std
    eval_df2 = df2[df2["type"] == "eval"].copy()
    grouped2 = eval_df2.groupby("step")
    steps2 = sorted(grouped2.groups.keys())
    means2, stds2 = [], []
    for s in steps2:
        vals = grouped2.get_group(s)["loss"].values
        means2.append(vals.mean())
        stds2.append(vals.std(ddof=1))

    steps2 = np.array(steps2)
    means2 = np.array(means2)
    stds2 = np.array(stds2)

    plt.figure()
    # Post-LN
    plt.plot(steps1, means1, label="Post-LN mean")
    plt.fill_between(steps1, means1 - stds1, means1 + stds1, alpha=0.2)

    # RMSNorm
    plt.plot(steps2, means2, label="RMSNorm mean")
    plt.fill_between(steps2, means2 - stds2, means2 + stds2, alpha=0.2)

    plt.xlabel("Step")
    plt.ylabel("Eval loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png))
    plt.close()
    print(f"[Saved plot] {out_png}")

def t_test_final_eval(postln_df, rms_df):
    _, postln_mean, _ = summarize_final_eval(postln_df)
    _, rms_mean, _ = summarize_final_eval(rms_df)

    postln_vals = summarize_final_eval(postln_df)[0]["loss"].values
    rms_vals = summarize_final_eval(rms_df)[0]["loss"].values

    t, p = stats.ttest_ind(rms_vals, postln_vals, equal_var=False)
    print(f"T-test on final eval losses: t={t:.3f}, p={p:.3f}")
    print("(Note: n=3 per group, so treat p-value as indicative only.)")

def plot_eval_diff(postln_df, rms_df, out_png):
    post_eval = postln_df[postln_df["type"] == "eval"]
    rms_eval = rms_df[rms_df["type"] == "eval"]

    merged = post_eval.merge(
        rms_eval,
        on=["step", "seed"],
        suffixes=("_postln", "_rms"),
    )
    merged["diff"] = merged["loss_rms"] - merged["loss_postln"]

    plt.figure()
    for seed, sub in merged.groupby("seed"):
        sub_sorted = sub.sort_values("step")
        plt.plot(sub_sorted["step"], sub_sorted["diff"], label=f"seed={seed}")

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("RMSNorm - Post-LN (eval loss)")
    plt.title("Eval loss difference (RMSNorm - Post-LN)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png))
    plt.close()
    print(f"[Saved plot] {out_png}")



def main():
    # 1) 加载两种配置的日志
    postln_df = load_losses(postln_csv)
    rms_df = load_losses(rmsnorm_csv)

    # 2) Final Eval Loss (per seed)
    postln_eval_per_seed, postln_eval_mean, postln_eval_std = summarize_final_eval(postln_df)
    rms_eval_per_seed, rms_eval_mean, rms_eval_std = summarize_final_eval(rms_df)

    print("\n==============================")
    print("1) Final Eval Loss (per seed)")
    print("==============================")

    print("\nPost-LN final eval per seed:")
    print(postln_eval_per_seed.to_string(index=False))
    print(f"Post-LN final eval mean±std: {postln_eval_mean:.4f} ± {postln_eval_std:.4f}")

    print("\nRMSNorm final eval per seed:")
    print(rms_eval_per_seed.to_string(index=False))
    print(f"RMSNorm final eval mean±std: {rms_eval_mean:.4f} ± {rms_eval_std:.4f}")

    # 2b) Final Train Loss (per seed)
    postln_train_per_seed, postln_train_mean, postln_train_std = summarize_final_train(postln_df)
    rms_train_per_seed, rms_train_mean, rms_train_std = summarize_final_train(rms_df)

    # simple t-test on final eval losses (small n, just for reference)
    t_test_final_eval(postln_df, rms_df)

    print("\n==============================")
    print("2) Final Train Loss (per seed)")
    print("==============================")

    print("\nPost-LN final train per seed:")
    print(postln_train_per_seed.to_string(index=False))
    print(f"Post-LN final train mean±std: {postln_train_mean:.4f} ± {postln_train_std:.4f}")

    print("\nRMSNorm final train per seed:")
    print(rms_train_per_seed.to_string(index=False))
    print(f"RMSNorm final train mean±std: {rms_train_mean:.4f} ± {rms_train_std:.4f}")

    print("\n==============================")
    print("3) Curve Mean±Std + Plots")
    print("==============================")

    # 3) 画曲线，对比不同 seed 的 eval loss / train loss 曲线是否更“靠近”

    # eval curves
    plot_eval_curves(
        postln_df, "Post-LN: Eval Loss (per seed)", "postln_eval_curves_per_seed.png"
    )
    plot_eval_curves(rms_df, "RMSNorm: Eval Loss (per seed)", "rmsnorm_eval_curves_per_seed.png")

    # train curves
    plot_train_curves(
        postln_df, "Post-LN: Train Loss (per seed)", "postln_train_curves_per_seed.png"
    )
    plot_train_curves(rms_df, "RMSNorm: Train Loss (per seed)", "rmsnorm_train_curves_per_seed.png")

    # 4) 画 mean±std 曲线
    plot_mean_std(
        postln_df, "Post-LN: Eval Loss (mean±std)", "postln_eval_mean_std.png"
    )
    plot_mean_std(rms_df, "RMSNorm: Eval Loss (mean±std)", "rmsnorm_eval_mean_std.png")

    # 5) 对比 mean±std 曲线
    plot_compare_mean_std(
        postln_df,
        rms_df,
        "Post-LN vs RMSNorm: Eval Loss (mean±std)",
        "postln_vs_rmsnorm_eval_mean_std.png",
    )

    # 6) 画 eval loss 差值曲线
    plot_eval_diff(postln_df, rms_df, "eval_loss_diff_rmsnorm_minus_postln.png")

    print("\n==============================")
    print("4) Check if two dataframes are the same")
    print("==============================")

    print(postln_df.head())
    print(rms_df.head())
    print("Same shape?", postln_df.shape, rms_df.shape)
    print("Exactly equal?", postln_df.equals(rms_df))


if __name__ == "__main__":
    main()
