#!/usr/bin/env python3
"""
绘制 results/0day 目录下 6 个 *_0day_res.csv 的 TP 交集图。
composite key = (vul_id, repo, fixed_commit_sha, target_file, target_func)

生成三种图表：
1. UpSet Plot (upsetplot)
2. supervenn
3. venn (5集合，去掉 VUDDY)
"""

import csv
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "results/0day"

# 6 个工具及其对应文件
TOOLS = OrderedDict([
    ("FIRE",    "fire_0day_res.csv"),
    ("MOVERY",  "movery_0day_res.csv"),
    ("V1SCAN",  "v1scan_0day_res.csv"),
    ("VUDDY",   "vuddy_0day_res.csv"),
    ("VULTURE", "vulture_0day_res.csv"),
    ("VERDICT", "verdict_0day_res.csv"),
])


def normalize_repo(repo: str) -> str:
    """统一 repo 名称：取最后一个部分"""
    return repo.strip().split("/")[-1]


def normalize_filepath(path: str, repo_short: str) -> str:
    """统一文件路径：去掉可能的 repo 名前缀"""
    path = path.strip()
    for prefix in [repo_short + "/"]:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    return path


def load_tp_keys(filepath: str) -> set:
    """从 CSV 中读取 TP 行，返回 composite key 的集合。"""
    keys = set()
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            judgement = row.get("judgement", "").strip()
            if judgement != "TP":
                continue

            vul_id = row.get("vul_id", "").strip()
            repo = normalize_repo(row.get("repo", ""))
            sha = row.get("fixed_commit_sha", "").strip()
            target_file = row.get("target_file", "").strip()
            target_func = row.get("target_func", "").strip()

            if not target_file:
                target_file = row.get("patch_file", "").strip()
            if not target_func:
                target_func = row.get("patch_func", "").strip()

            target_file = normalize_filepath(target_file, repo)

            key = (vul_id, repo, sha, target_file, target_func)
            keys.add(key)
    return keys


def plot_upset(tool_sets: dict):
    """使用 upsetplot 绘制 UpSet Plot"""
    from upsetplot import from_contents, UpSet

    # 准备数据：使用字符串化的 key
    contents = {}
    for tool_name, tp_keys in tool_sets.items():
        contents[tool_name] = [str(k) for k in tp_keys]

    data = from_contents(contents)

    upset = UpSet(
        data,
        show_counts=True,
        show_percentages=False,
        sort_by="cardinality",
        sort_categories_by="cardinality",
        min_subset_size=0,
        element_size=40,
    )

    fig = plt.figure(figsize=(14, 8))
    upset.plot(fig=fig)
    fig.suptitle("0-day TP UpSet Plot", fontsize=16, y=1.02)

    out_pdf = "results/0day/tp_upset.pdf"
    out_png = "results/0day/tp_upset.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"UpSet Plot 已保存至: {out_pdf} / {out_png}")


def plot_supervenn(tool_sets: dict):
    """使用 supervenn 绘制"""
    from supervenn import supervenn

    all_keys = set()
    for v in tool_sets.values():
        all_keys |= v

    all_keys_list = sorted(all_keys)
    key_to_idx = {k: i for i, k in enumerate(all_keys_list)}

    sets = []
    labels = []
    for tool_name, tp_keys in tool_sets.items():
        idx_set = set(key_to_idx[k] for k in tp_keys)
        sets.append(idx_set)
        labels.append(f"{tool_name} ({len(tp_keys)})")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    supervenn(sets, labels, side_plots="right", chunks_ordering="minimize gaps",
              sets_ordering=None, widths_minmax_ratio=0.05, ax=ax)
    ax.set_title("0-day TP Venn Diagram (supervenn)", fontsize=16, pad=20)

    plt.tight_layout()
    fig.savefig("results/0day/tp_venn_supervenn.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("results/0day/tp_venn_supervenn.png", bbox_inches="tight", dpi=150)
    print("supervenn 图表已保存")


def plot_venn5(tool_sets: dict):
    """使用 venn 库绘制 5 集合韦恩图（去掉 VUDDY）"""
    from venn import venn

    active_sets = OrderedDict()
    for tool_name, tp_keys in tool_sets.items():
        if len(tp_keys) > 0:
            label = f"{tool_name} ({len(tp_keys)})"
            active_sets[label] = set(str(k) for k in tp_keys)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    venn(active_sets, ax=ax, fontsize=11, legend_loc="upper right")
    ax.set_title("0-day TP Venn Diagram\n(VUDDY: 0 TPs, not shown)", fontsize=16, pad=15)

    plt.tight_layout()
    fig.savefig("results/0day/tp_venn.pdf", bbox_inches="tight", dpi=150)
    fig.savefig("results/0day/tp_venn.png", bbox_inches="tight", dpi=150)
    print("Venn (5集合) 已保存")


def main():
    tool_sets = OrderedDict()
    all_keys = set()

    for tool_name, filename in TOOLS.items():
        fpath = os.path.join(DATA_DIR, filename)
        tp_keys = load_tp_keys(fpath)
        tool_sets[tool_name] = tp_keys
        all_keys |= tp_keys
        print(f"{tool_name:>10}: {len(tp_keys):>4} TPs")

    print(f"\n{'Union':>10}: {len(all_keys):>4} unique TPs")

    # 打印交叉统计
    print("\n=== 交叉统计 ===")
    for key in all_keys:
        membership = frozenset(name for name in tool_sets if key in tool_sets[name])
        pass  # 统计放到绘图中

    combo_counts = {}
    for key in all_keys:
        membership = frozenset(name for name in tool_sets if key in tool_sets[name])
        combo_counts[membership] = combo_counts.get(membership, 0) + 1

    for combo, count in sorted(combo_counts.items(), key=lambda x: (-len(x[0]), -x[1])):
        print(f"  {' ∩ '.join(sorted(combo))}: {count}")

    # 绘制三种图表
    print("\n--- 绘制 UpSet Plot ---")
    plot_upset(tool_sets)

    print("\n--- 绘制 supervenn ---")
    plot_supervenn(tool_sets)

    print("\n--- 绘制 Venn (5集合) ---")
    plot_venn5(tool_sets)


if __name__ == "__main__":
    main()
