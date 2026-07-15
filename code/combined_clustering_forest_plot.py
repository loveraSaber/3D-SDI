import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_label(name: str) -> str:
    mapping = {
        "HTN": "Hypertension",
        "OSA": "Sleep Apnea",
        "Poor Sleep": "Poor Sleep Quality",
        "Good Sleep": "Good Sleep",
    }
    return mapping.get(name, name)


def build_y_order(frames):
    group_titles = {"Features", "Outcomes"}
    excluded_labels = {"Good Sleep"}
    preferred = [
        "Age", "BMI", "SE", "SL",
        "Features",
        "CWSD", "STDE", "TDTL", "SSTR", "BBAI",
        "Outcomes",
        "Sex", "Sleep Apnea", "Poor Sleep Quality", "Insomnia", "Diabetes", "CVD", "Hypertension",
    ]
    existing = set()
    for df in frames:
        existing.update(df["Variable"].tolist())
    existing.update(group_titles)
    existing -= excluded_labels

    # 固定保留预定义纵轴顺序（即使某些变量在某次统计结果中缺失，也保留标签）
    ordered = preferred.copy()
    remain = sorted(v for v in existing if v not in ordered and v not in group_titles)
    return ordered + remain


def plot_panel(ax, df, y_labels, title):
    ax_top = ax.twiny()
    y_map = {label: i for i, label in enumerate(y_labels)}
    max_y = len(y_labels) - 1
    top_y = -0.5
    bottom_y = max_y + 0.5

    df = df.copy()
    df["Variable"] = df["Variable"].apply(normalize_label)
    df = df[df["Variable"] != "Good Sleep"]
    df = df[df["Variable"].isin(y_map)]

    d_df = df[df["Type"] == "d"]
    d_df = d_df[d_df["Val"].notna()]
    if not d_df.empty:
        y = d_df["Variable"].map(y_map).to_numpy()
        ax_top.errorbar(
            d_df["Val"], y,
            xerr=[d_df["Val"] - d_df["Low"], d_df["High"] - d_df["Val"]],
            fmt="o", color="#6baed6", capsize=2.5, markersize=5
        )

    or_df = df[df["Type"] == "OR"]
    or_df = or_df[or_df["Val"].notna()]
    if not or_df.empty:
        y = or_df["Variable"].map(y_map).to_numpy()
        ax.errorbar(
            or_df["Val"], y,
            xerr=[or_df["Val"] - or_df["Low"], or_df["High"] - or_df["Val"]],
            fmt="*", color="#e6550d", capsize=2.5, markersize=7
        )

    # 分组分隔线与分段参考线（以 Outcomes 为分界）
    if "Outcomes" in y_map:
        outcomes_y = y_map["Outcomes"]
        ax.axhline(outcomes_y, color="black", linestyle="-", linewidth=1.6, alpha=0.9)
        ax_top.axhline(outcomes_y, color="black", linestyle="-", linewidth=1.6, alpha=0.9)

        # Cohen's d: 从顶部延伸到 Outcomes
        ax_top.plot([0, 0], [top_y, outcomes_y], linestyle="--", linewidth=1, color="gray", alpha=0.7)
        # Odds Ratio: 从底部延伸到 Outcomes
        ax.plot([1, 1], [outcomes_y, bottom_y], linestyle="--", linewidth=1, color="gray", alpha=0.7)
    else:
        ax.axvline(1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax_top.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(-1, 5)
    ax_top.set_xlim(-3, 3)
    ax.set_xlabel("Odds Ratio", fontweight="bold")
    ax_top.set_xlabel("Cohen's d", fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(False)
    ax_top.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax_top.tick_params(axis="x", labelsize=9)
    ax.set_ylim(bottom_y, top_y)
    ax_top.set_ylim(bottom_y, top_y)

    # 底部方向提示（与单队列图风格一致）
    ax.annotate("", xy=(0.08, -0.09), xycoords="axes fraction", xytext=(0.42, -0.09),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
    ax.text(0.25, -0.13, "Normal Sleep", transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(0.92, -0.09), xycoords="axes fraction", xytext=(0.58, -0.09),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
    ax.text(0.75, -0.13, "Disturbed Sleep", transform=ax.transAxes, ha="center", fontsize=9, fontweight="bold")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "..", "output"))
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "SHHS": os.path.join(output_dir, "shhs_forest_stats.csv"),
        "MROS": os.path.join(output_dir, "mros_forest_stats.csv"),
        "CFS": os.path.join(output_dir, "cfs_forest_stats.csv"),
    }

    data = {}
    for cohort, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少文件: {path}，请先运行 {cohort} 分析脚本。")
        df = pd.read_csv(path)
        df["Variable"] = df["Variable"].apply(normalize_label)
        df = df[df["Variable"] != "Good Sleep"]
        data[cohort] = df

    # 额外输出：三队列结果合并保存为一份 CSV（便于后续统计/制表）
    combined_df = pd.concat(
        [data["SHHS"].assign(Cohort="SHHS"),
         data["MROS"].assign(Cohort="MROS"),
         data["CFS"].assign(Cohort="CFS")],
        ignore_index=True
    )
    combined_csv_path = os.path.join(output_dir, "combined_forest_stats_all_cohorts.csv")
    combined_df.to_csv(combined_csv_path, index=False, encoding="utf-8-sig")
    print(f"三队列结果已保存至: {combined_csv_path}")

    y_labels = build_y_order([data["SHHS"], data["MROS"], data["CFS"]])
    y_pos = np.arange(len(y_labels))

    fig, axes = plt.subplots(1, 3, figsize=(21, 12), sharey=True)
    plot_panel(axes[0], data["SHHS"], y_labels, "SHHS")
    plot_panel(axes[1], data["MROS"], y_labels, "MROS")
    plot_panel(axes[2], data["CFS"], y_labels, "CFS")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(y_labels, fontsize=10)
    for label in axes[0].get_yticklabels():
        if label.get_text() in {"Features", "Outcomes"}:
            label.set_fontweight("bold")
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("", fontsize=16, fontweight="bold")
    plt.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.15, wspace=0.12)

    out_path = os.path.join(output_dir, "combined_forest_plot_shared_y.png")
    plt.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close()
    print(f"合并图已保存至: {out_path}")


if __name__ == "__main__":
    main()
