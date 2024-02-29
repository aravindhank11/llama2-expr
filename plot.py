import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV
filename = sys.argv[1]
metricname = os.path.basename(filename).split('.')[0]
data = pd.read_csv(sys.argv[1], skipinitialspace=True)
data = data.sort_values(by="load")

# Define modes for each mode
colors = ["blue", "green", "orange", "red", "brown", "yellow", "black"]
modes = {}
for i, mode in enumerate(data["mode"].unique()):
    modes[mode] = colors[i]

# Define hatch for each model
hatch_styles = ["....", "////", "++", "xx", "\\\\\\\\", "----", "||||"]
model_cols = [col for col in data.columns if col not in ["load", "mode"]]
models = {}
for i, model_name in enumerate(model_cols):
    models[model_name] = hatch_styles[i]

bar_width = 5
transparency = 0.6
num_loads = len(data["load"].unique())


# Plot
fig, ax = plt.subplots(figsize=(100, 12))

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
            alpha=0.6,
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
                weight="bold", fontsize=15
            )
        bottom += col


x_ticks = [
    i * (len(modes) + 1) * bar_width + (len(modes)) * bar_width / 2
    for i in range(num_loads)
]

ax.set_xlabel("Load Values")
ax.set_ylabel(f"{metricname} Metric", labelpad=300)
ax.set_title(f"{metricname}")
ax.set_xticks(x_ticks)
ax.set_xticklabels(data["load"].unique())
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
