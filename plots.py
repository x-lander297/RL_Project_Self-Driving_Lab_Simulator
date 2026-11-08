"""Create learning_curve.png, heatmap.png, bar.png in results/."""
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lab_env import LabEnv

env = LabEnv()
curves = pd.read_csv("results/curves.csv")
summary = pd.read_csv("results/summary.csv").set_index("policy")

# 1. Learning curve: mean +/- 1 std across seeds
g = curves.groupby("episode")["eval_final_yield"].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(g.index, g["mean"], label="Q-learning (mean of 5 seeds)")
ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.25, label="±1 std")
for name, ls in [("Random", "--"), ("FixedRecipe", "--"), ("HillClimb", "--")]:
    ax.axhline(summary.loc[name, "mean_final_yield"], ls=ls, color={"Random": "tab:red",
               "FixedRecipe": "tab:green", "HillClimb": "tab:purple"}[name], label=f"{name} (mean)")
ax.axhline(env.y_max, ls=":", color="k", label="Global max yield")
ax.set(title="Greedy-policy final yield during training", xlabel="Training episode",
       ylabel="Mean final true yield")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
fig.savefig("results/learning_curve.png", dpi=150); plt.close(fig)

# 2. Heatmap of true yield with greedy-policy arrows (seed 0)
q = np.load("results/qtable_seed0.npy")
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(env.y_grid.T, origin="lower", cmap="viridis")   # x = T index, y = C index
fig.colorbar(im, ax=ax, label="True yield (noise-free)")
dirs = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
for s in range(121):
    i, j = divmod(s, 11)
    if not q[s].any():                       # never updated: greedy action is just a tie-break
        ax.plot(i, j, "x", color="gray", ms=5)
        continue
    a = int(q[s].argmax())
    if a == 4:
        ax.plot(i, j, "o", color="white", mec="black", ms=5)
    else:
        dx, dy = dirs[a]
        ax.arrow(i - 0.3 * dx, j - 0.3 * dy, 0.6 * dx, 0.6 * dy, head_width=0.2, length_includes_head=True,
                 fc="white", ec="black", lw=0.6)
ax.axvline(8.5, ls="--", color="red", lw=1)  # T > 90 C region starts to the right
ax.set_xticks(range(11)); ax.set_xticklabels([f"{t:.0f}" for t in env.T_vals])
ax.set_yticks(range(11)); ax.set_yticklabels([f"{c:.2f}" for c in env.C_vals])
ax.set(xlabel="Temperature (°C)", ylabel="Concentration (mol/L)",
       title="True yield and greedy policy (seed 0)\narrow = move, o = hold, x = state never visited, red line: T>90 °C")
fig.tight_layout(); fig.savefig("results/heatmap.png", dpi=150); plt.close(fig)

# 3. Bar chart with 95% CI error bars
names = list(summary.index)
m = summary["mean_final_yield"].values
err = np.vstack([m - summary["ci95_low_final_yield"].values, summary["ci95_high_final_yield"].values - m])
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(names, m, yerr=err, capsize=6, color=["tab:blue", "tab:red", "tab:green", "tab:purple"][:len(names)])
for k, v in enumerate(m):
    ax.text(k, v + 0.015, f"{v:.3f}", ha="center")
ax.set(title="Mean final true yield by policy (95% CI, 200 episodes)", xlabel="Policy",
       ylabel="Mean final true yield")
ax.grid(axis="y", alpha=0.3); fig.tight_layout()
fig.savefig("results/bar.png", dpi=150); plt.close(fig)
print("Saved results/learning_curve.png, results/heatmap.png, results/bar.png")