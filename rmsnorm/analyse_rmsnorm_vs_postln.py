import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")  # 避免某些环境下无显示错误

# 手动填入你保存结果的目录（对应不同时间跑的实验）
OUTPUT_DIR = "./result-1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

POSTLN_DIR = "./result-postln-static"  # 例如: result-20251126-postln-static
RMSNORM_DIR = "./result-rmsnorm-static"  # 例如: result-20251126-rmsnorm-static
MASKING_TYPE = "dynamic"

postln_csv = f"{POSTLN_DIR}/postln_{MASKING_TYPE}_loss_all_seeds.csv"
rmsnorm_csv = f"{RMSNORM_DIR}/rmsnorm_{MASKING_TYPE}_loss_all_seeds.csv"


def load_losses(path):
    df = pd.read_csv(path)
    # safety checks
    required_cols = {"step", "loss", "type", "seed"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def get_final_eval_losses(df: pd.DataFrame):
    """
    从记录中提取每个 seed 的最后一个 eval loss
    记录中有 type=train/eval，我们只取 type=='eval'。
    """
    final_losses = {}
    for seed, sub in df[df["type"] == "eval"].groupby("seed"):
        # 按 step 排序，取最后一条
        sub_sorted = sub.sort_values("step")
        last = sub_sorted.iloc[-1]["loss"]
        final_losses[seed] = last
    return final_losses


def get_final_train_losses(df: pd.DataFrame):
    """从记录中提取每个 seed 的最后一个 train loss。

    记录中有 type=train/eval，这里只取 type=='train'。
    """
    final_losses = {}
    for seed, sub in df[df["type"] == "train"].groupby("seed"):
        sub_sorted = sub.sort_values("step")
        last = sub_sorted.iloc[-1]["loss"]
        final_losses[seed] = last
    return final_losses


def summarize(name, d):
    vals = np.array(list(d.values()), dtype=float)
    print(f"{name}: mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}")


def plot_eval_curves(df, title, out_png):
    plt.figure()
    for seed, sub in df[df["type"] == "eval"].groupby("seed"):
        sub_sorted = sub.sort_values("step")
        plt.plot(sub_sorted["step"], sub_sorted["loss"], label=f"seed={seed}")
    plt.xlabel("Step")
    plt.ylabel("Eval loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png))
    plt.close()
    print(f"[Saved plot] {out_png}")


def plot_train_curves(df, title, out_png):
    """画每个 seed 的 train loss 曲线。"""
    plt.figure()
    for seed, sub in df[df["type"] == "train"].groupby("seed"):
        sub_sorted = sub.sort_values("step")
        plt.plot(sub_sorted["step"], sub_sorted["loss"], label=f"seed={seed}")
    plt.xlabel("Step")
    plt.ylabel("Train loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, out_png))
    plt.close()
    print(f"[Saved plot] {out_png}")


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


def main():
    # 1) 加载两种配置的日志
    postln_df = load_losses(postln_csv)
    rms_df = load_losses(rmsnorm_csv)

    # 2) 计算每种配置的最终 eval loss per seed
    postln_final = get_final_eval_losses(postln_df)
    rms_final = get_final_eval_losses(rms_df)

    print("=== Final Eval Loss per seed ===")
    print("Post-LayerNorm:", postln_final)
    print("RMSNorm:       ", rms_final)

    print("\n=== Summary ===")
    summarize("Post-LN", postln_final)
    summarize("RMSNorm", rms_final)

    # 2b) 计算每种配置的最终 train loss per seed
    postln_train_final = get_final_train_losses(postln_df)
    rms_train_final = get_final_train_losses(rms_df)

    print("\n=== Final Train Loss per seed ===")
    print("Post-LayerNorm (train):", postln_train_final)
    print("RMSNorm (train):       ", rms_train_final)

    print("\n=== Train Summary ===")
    summarize("Post-LN train", postln_train_final)
    summarize("RMSNorm train", rms_train_final)

    # 3) 画曲线，对比不同 seed 的 eval loss / train loss 曲线是否更“靠近”

    # eval curves
    plot_eval_curves(
        postln_df, "Post-LN: Eval Loss (per seed)", "postln_eval_curves.png"
    )
    plot_eval_curves(rms_df, "RMSNorm: Eval Loss (per seed)", "rmsnorm_eval_curves.png")

    # train curves
    plot_train_curves(
        postln_df, "Post-LN: Train Loss (per seed)", "postln_train_curves.png"
    )
    plot_train_curves(rms_df, "RMSNorm: Train Loss (per seed)", "rmsnorm_train_curves.png")

    # 4) 可以再画 mean±std 曲线（可选）
    plot_mean_std(
        postln_df, "Post-LN: Eval Loss (mean±std)", "postln_eval_mean_std.png"
    )
    plot_mean_std(rms_df, "RMSNorm: Eval Loss (mean±std)", "rmsnorm_eval_mean_std.png")

    plot_compare_mean_std(
        postln_df,
        rms_df,
        "Post-LN vs RMSNorm: Eval Loss (mean±std)",
        "postln_vs_rmsnorm_eval_mean_std.png",
    )


if __name__ == "__main__":
    main()
