import argparse, json, sys, csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import re

# ─────────────────────────────────────────────────────────────
def letter(idx: int) -> str:                # 0 → 'a'
    return chr(ord('a') + idx)

def plot_mse_distributions_from_groups(mse_groups):
    import matplotlib.pyplot as plt

    t1 = mse_groups["Top-1 Correct"]
    t2 = mse_groups["Top-2 Correct"]
    inc = mse_groups["Incorrect"]

    n1, n2, n3 = len(t1), len(t2), len(inc)

    labels = [
        f"Top-1 Correct\n(n={n1})",
        f"Top-2 Correct\n(n={n2})",
        f"Incorrect\n(n={n3})"
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot([t1, t2, inc], labels=labels)
    plt.xticks([1, 2, 3], labels, fontsize=16)
    plt.ylabel("Final MSE", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────
# New: works with 1-level (Seq_x) or 2-level (Seed*/Seq_x) hierarchy
# ─────────────────────────────────────────────────────────────
def _iter_seq_dirs(root: Path):
    """
    Yield every Seq_<n> directory that exists under `root`, regardless of
    whether there is an intermediate Seed_* directory.

    Accepts:
        root/
            Seq_00/
            Seq_01/
            ...
        root/
            Seed1/Seq_00/
            Seed1/Seq_01/
            ...
            Seed2/Seq_00/ ...
    """
    # 1) direct children: root/Seq_*
    for p in root.glob("Seq_*"):
        if p.is_dir():
            yield p

    # 2) one level deeper: root/Seed*/Seq_*
    for seed_dir in root.glob("Seed*"):
        if not seed_dir.is_dir():
            continue
        for p in seed_dir.glob("Seq_*"):
            if p.is_dir():
                yield p


# ─────────────────────────────────────────────────────────────
def analyse(root: Path, verbose=False):
    """Aggregate recognition results across *all* Seq dirs under `root`."""
    seq_stat = {i: {"total": 0, "top1": 0, "top2": 0} for i in range(26)}
    missing  = []

    # walk both styles (Seq_* or Seed*/Seq_*)
    found_seq_dirs = defaultdict(list)         # sid -> [Path, ...]
    for seq_dir in _iter_seq_dirs(root):
        m = re.match(r"Seq_(\d+)", seq_dir.name)
        if m:
            found_seq_dirs[int(m.group(1))].append(seq_dir)

    # loop over canonical ids 0-25
    for seq_id in range(26):
        dirs = found_seq_dirs.get(seq_id, [])
        if not dirs:
            missing.append(seq_id)
            if verbose:
                print(f"[WARN] missing Seq_{seq_id:02d}", file=sys.stderr)
            continue

        for seq_dir in dirs:
            for mp in seq_dir.glob("meta_*.json"):
                try:
                    with open(mp, encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception as e:
                    if verbose:
                        print(f"[ERR] read fail: {mp} ({e})", file=sys.stderr)
                    continue

                ok1 = meta.get("match_top1", False)
                ok2 = meta.get("success",    False)

                s = seq_stat[seq_id]
                s["total"] += 1
                s["top1"]  += int(ok1)
                s["top2"]  += int(ok2)

                if verbose:
                    print(f"[OK] {mp.relative_to(root)}  "
                          f"match_top1={ok1}  success={ok2}")

    total = {"total":0,"top1":0,"top2":0}
    for s in seq_stat.values():
        total["total"] += s["total"]
        total["top1"]  += s["top1"]
        total["top2"]  += s["top2"]

    return total, seq_stat, missing


# ─────────────────────────────────────────────────────────────
def collect_mse_by_recognition_with_path(root: Path):
    """
    Recursively collect MSEs from meta_*.json under both Seq_* and Seed*/Seq_*.
    """
    mse_groups = {
        "Top-1 Correct": [],
        "Top-2 Correct": [],
        "Incorrect": []
    }
    paths = {
        "Top-1 Correct": [],
        "Top-2 Correct": [],
        "Incorrect": []
    }

    for json_file in root.rglob("meta_*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            mse = meta.get("final_mse")
            if mse is None:
                continue
            top1 = meta.get("match_top1", False)
            top2 = meta.get("success", False)

            if top1:
                key = "Top-1 Correct"
            elif top2:
                key = "Top-2 Correct"
            else:
                key = "Incorrect"

            mse_groups[key].append(mse)
            paths[key].append(json_file)
        except Exception as e:
            print(f"[ERR] Failed reading: {json_file} ({e})")

    return mse_groups, paths

def print_summary(total):
    n = total["total"]
    r1 = total["top1"]/n*100 if n else 0
    r2 = total["top2"]/n*100 if n else 0
    print("\n【Result Summary】")
    print("────────────────────────────")
    print(f"Number of files      : {n}")
    print(f"Top-1 Accuracy         : {r1:6.2f}%  ({total['top1']}/{n})")
    print(f"Top-2 Accuracy         : {r2:6.2f}%  ({total['top2']}/{n})")
    print("────────────────────────────")

def print_per_seq(seq_stat, missing):
    print("\n【Seq-wise accuracy】")
    print("─────────────────────────────────────────")
    print("Seq | n data | Top-1(%) | Top-2(%)")
    print("─────────────────────────────────────────")
    for sid in range(26):
        if sid in missing:
            print(f" {letter(sid)}  |  -   |  missing |  missing")
            continue
        s = seq_stat[sid]
        n = s["total"]
        t1 = s["top1"]/n*100 if n else 0
        t2 = s["top2"]/n*100 if n else 0
        print(f" {letter(sid)}  | {n:3} | {t1:7.2f} | {t2:7.2f}")
    print("─────────────────────────────────────────")

def save_csv(path: Path, seq_stat, total, missing):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seq_id","letter","total",
                    "top1_correct","top1_acc",
                    "top2_correct","top2_acc","missing"])
        for sid in range(26):
            miss = sid in missing
            s = seq_stat[sid]
            n = s["total"] or 1
            w.writerow([sid, letter(sid), s["total"],
                        s["top1"], s["top1"]/n if not miss else "",
                        s["top2"], s["top2"]/n if not miss else "",
                        miss])
        n_all = total["total"]
        w.writerow([])
        w.writerow(["ALL","",n_all,
                    total["top1"], total["top1"]/n_all if n_all else "",
                    total["top2"], total["top2"]/n_all if n_all else ""])
    print(f"[INFO] Save CSV → {path}")
    
def mse_statistical_test(mse_groups):
    from scipy.stats import mannwhitneyu

    t1 = mse_groups["Top-1 Correct"]
    t2 = mse_groups["Top-2 Correct"]
    inc = mse_groups["Incorrect"]

    print("\n【Statistical result】")
    print("────────────────────────────")

    def test_pair(name1, a, name2, b):
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        print(f"{name1} vs {name2}:")
        print(f"  U={stat:.1f}, p={p:.4g}", "→ Statistically Different" if p < 0.05 else "→ n.s.")

    test_pair("Top-1", t1, "Top-2", t2)
    test_pair("Top-1", t1, "Incorrect", inc)
    test_pair("Top-2", t2, "Incorrect", inc)

    print("────────────────────────────")

def detect_outliers_and_report(mse_groups, paths_by_group, iqr_factor=1.5):
    print("\n【Off point】")
    print("────────────────────────────")

    for group in mse_groups:
        mses = mse_groups[group]
        paths = paths_by_group[group]
        if not mses:
            continue

        q1 = np.percentile(mses, 25)
        q3 = np.percentile(mses, 75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        print(f"\n[{group}]")
        print(f"IQR = {iqr:.6f}, Threshold: < {lower:.6f} or > {upper:.6f}")
        for mse, path in zip(mses, paths):
            if mse < lower or mse > upper:
                print(f"  {mse:.6f} → {path}")


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="Seq_x")
    ap.add_argument("--csv", metavar="PATH",
                    )
    ap.add_argument("--verbose", action="store_true",
                    )
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        ap.error(f"No directory: {root}")

    total, seq_stat, missing = analyse(root, args.verbose)

    print_summary(total)
    print_per_seq(seq_stat, missing)

    if missing:
        miss_letters = ", ".join(letter(s) for s in missing)
        print(f"\n[WARN] meta_*.json none for Seq: {miss_letters}")

    if args.csv:
        save_csv(Path(args.csv).expanduser(), seq_stat, total, missing)
        
    # MSE収集
    mse_groups, paths_by_group = collect_mse_by_recognition_with_path(root)

    # 検定と可視化
    mse_statistical_test(mse_groups)
    detect_outliers_and_report(mse_groups, paths_by_group)
    plot_mse_distributions_from_groups(mse_groups)

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
