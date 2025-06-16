import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_reports(base_dir):
    records = []
    for run_dir in sorted(glob.glob(os.path.join(base_dir, "run_*"))):
        run_name = os.path.basename(run_dir)
        for report_file in sorted(glob.glob(os.path.join(run_dir, "REPORT_*.txt"))):
            m = re.search(r'REPORT_(\d+)\.txt', report_file)
            if not m:
                continue
            fold = int(m.group(1))
            content = open(report_file).read()

            acc = float(re.search(r"accuracy\s+([\d.]+)", content).group(1))
            f1_lines = re.findall(r"^\s*([A-Za-z]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", content, re.MULTILINE)
            f1_dict = {f"{label} F1": float(f1) for label, f1 in f1_lines}
            macro_f1 = float(re.search(r"macro avg\s+[0-9.]+\s+[0-9.]+\s+([\d.]+)", content).group(1))
            weighted_f1 = float(re.search(r"weighted avg\s+[0-9.]+\s+[0-9.]+\s+([\d.]+)", content).group(1))

            rec = dict(Run=run_name, Fold=fold, Accuracy=acc,
                       **f1_dict, **{"Macro F1": macro_f1, "Weighted F1": weighted_f1})
            records.append(rec)

    df = pd.DataFrame(records)
    # Fill missing class F1s
    class_cols = sorted(c for c in df.columns if c.endswith(" F1") and c not in ("Macro F1", "Weighted F1"))
    for c in class_cols:
        df[c] = df.get(c, 0.0)
    df = df[["Run", "Fold", "Accuracy"] + class_cols + ["Macro F1", "Weighted F1"]]
    return df, class_cols

def write_aligned_report(df, class_f1_cols, base_dir, output_file):
    # Ensure Fold is integer
    df["Fold"] = df["Fold"].astype(int)

    # Numeric stats columns (exclude Run, Fold)
    stats_cols = [c for c in df.columns if c not in ("Run", "Fold") and pd.api.types.is_numeric_dtype(df[c])]

    # Compute summary stats
    mean_s = df[stats_cols].mean()
    var_s = df[stats_cols].var()    # unbiased variance :contentReference[oaicite:3]{index=3}
    sem_s = df[stats_cols].sem()    # standard error :contentReference[oaicite:4]{index=4}

    # Build summary rows
    summary = []
    for label, series in [("average", mean_s), ("variance", var_s), ("stderr", sem_s)]:
        row = {c: "" for c in df.columns}
        row["Fold"] = label
        for c in stats_cols:
            row[c] = series[c]
        summary.append(row)

    df_full = pd.concat([df, pd.DataFrame(summary)], ignore_index=True, sort=False)

    # Determine column widths
    col_widths = {}
    for c in df_full.columns:
        maxwidth = max(len(c), 10)
        for v in df_full[c]:
            s = f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
            maxwidth = max(maxwidth, len(s) + 2)
        col_widths[c] = maxwidth

    # Render lines
    lines = []
    header = "".join(c.ljust(col_widths[c]) for c in df_full.columns)
    sep = "-" * sum(col_widths.values())
    lines.extend([header, sep])

    for _, row in df_full.iterrows():
        line = ""
        for c in df_full.columns:
            v = row[c]
            if isinstance(v, int):
                s = f"{v:d}"
            elif isinstance(v, float):
                s = f"{v:.2f}"
            else:
                s = str(v)
            line += s.ljust(col_widths[c])
        lines.append(line)

    with open(output_file, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Report saved to {output_file}")

    # Plotting (unchanged logic)
    plot_df = df_full[df_full["Fold"].apply(lambda x: isinstance(x, int))]
    plt.figure(figsize=(10, 6))
    for run_name, grp in plot_df.groupby("Run"):
        plt.plot(grp["Fold"], grp["Accuracy"], marker='o', label=f"{run_name}")
    plt.ylim(0,1); plt.title(f"Accuracy Across Folds for {base_dir}")
    plt.xlabel("Fold"); plt.ylabel("Accuracy")
    plt.grid(True); plt.legend(); plt.savefig(f"REPORT_{base_dir}_accuracy.png"); plt.show()

    for col in class_f1_cols:
        plt.figure(figsize=(10, 6))
        for run_name, grp in plot_df.groupby("Run"):
            plt.plot(grp["Fold"], grp[col], marker='o', label=run_name)
        avg_val = plot_df[col].mean()
        plt.axhline(avg_val, linestyle='--', color='gray', label="Average")
        plt.title(f"{col} Across Folds - {base_dir}")
        plt.xlabel("Fold"); plt.ylabel("F1 Score")
        plt.ylim(0,1); plt.grid(True); plt.legend()
        plt.savefig(f"REPORT_{base_dir}_{col.replace(' ', '_')}.png")
        plt.show()

# === RUN ===
cwd = os.getcwd()
basename = os.path.basename(cwd)
df, class_f1_cols = parse_reports(cwd)
write_aligned_report(df, class_f1_cols, basename, f"REPORT_{basename}.txt")
