import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument("--csv-file-path", type=str, required=True)
parser.add_argument("--png-file-path", type=str, required=True)
opt = parser.parse_args()

# Read data from CSV
results_dir, filename = os.path.split(opt.csv_file_path)
metricname = filename.split('.')[0]
data = pd.read_csv(opt.csv_file_path, skipinitialspace=True)
data = data.sort_values(by="load")

# Figure the place to save
plots_dir, plot_filename = os.path.split(opt.png_file_path)
os.makedirs(plots_dir, exist_ok=True)

# Define colors for each mode
colors = ["blue", "green", "orange", "red", "brown", "yellow", "black"]
modes = {}
for i, mode in enumerate(data["mode"].unique()):
    modes[mode] = colors[i]

# Define hatch for each model
hatch_styles = ["////", "----", "\\\\\\\\", "....", "++", "xx", "----"]
model_cols = [col for col in data.columns if col not in ["load", "mode"]]
models = {}
for i, model_name in enumerate(model_cols):
    models[model_name] = hatch_styles[i]

# Few more constants
bar_width = 5
transparency = 0.6
num_loads = len(data["load"].unique())

# Plot
fig, ax = plt.subplots(figsize=(25, 10))

mode_ctr = -1
for mode, color in modes.items():
    mode_ctr += 1
    bottom = pd.Series([0] * num_loads)
    for model, hatch in models.items():
        col = data.loc[data["mode"] == mode, model]
        col = col.reset_index(drop=True)
        pos = [
            p * (len(modes) + 1) * bar_width + bar_width * mode_ctr
            for p in range(num_loads)
        ]
        bars = ax.bar(
            pos,
            col,
            bar_width,
            bottom=bottom,
            color=color,
            hatch=hatch,
            alpha=0.5,
            label=f"{mode} / {model}"
        )

        for i, bar in enumerate(bars):
            height = bar.get_height()
            x_loc = bar.get_x() + bar.get_width() / 2.
            y_loc = bottom[i] + height / 2.
            ax.text(
                x_loc, y_loc,
                "%d" % int(height),
                ha="center", va="center", color="black",
                weight="bold", fontsize=12
            )
        bottom += col


x_ticks = [
    i * (len(modes) + 1) * bar_width + (len(modes)) * bar_width / 2
    for i in range(num_loads)
]

ax.set_xlabel("Load Values")
ax.set_ylabel(f"{metricname} Metric", labelpad=10)
ax.set_title(f"{metricname}")
ax.set_xticks(x_ticks)
ax.set_xticklabels(data["load"].unique())
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(opt.png_file_path)
plt.close()
