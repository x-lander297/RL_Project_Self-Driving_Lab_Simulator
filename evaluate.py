"""Fair comparison of Q-learning vs baselines using common random numbers."""
import numpy as np
import pandas as pd
from lab_env import LabEnv
from agents import QAgent, RandomAgent, FixedRecipeAgent, HillClimbAgent

N_EP, HORIZON, SEED_BASE = 200, 30, 100_000   # test seeds differ from training-curve eval seeds
env = LabEnv()
# "Fixed recipe" T=50 C, C=1.0 mol/L is off-grid -> nearest grid cell
FIXED_START = (int(np.argmin(abs(env.T_vals - 50))), int(np.argmin(abs(env.C_vals - 1.0))))
rng = np.random.default_rng(2024)
STARTS = [(int(a), int(b)) for a, b in rng.integers(0, 11, size=(N_EP, 2))]  # shared by policies
q_trained = np.load("results/qtable_seed0.npy")


def make_policies():
    qa = QAgent()
    qa.q = q_trained.copy()
    return {"QAgent": qa, "Random": RandomAgent(seed=0),
            "FixedRecipe": FixedRecipeAgent(), "HillClimb": HillClimbAgent()}


def evaluate_policy(name, agent, start_fn):
    rec = []
    for k in range(N_EP):
        if hasattr(agent, "reset"):
            agent.reset()
        # Episode k uses the same env seed (same noise stream) for every policy
        s, _ = env.reset(seed=SEED_BASE + k, options={"start": start_fn(k)})
        ret, unsafe, hit, found = 0.0, 0, HORIZON, False
        for t in range(1, HORIZON + 1):
            a = agent.act(s, greedy=True)
            s2, r, terminated, truncated, info = env.step(a)
            if isinstance(agent, HillClimbAgent):
                agent.update(s, a, r, s2)
            ret += r
            unsafe += int(info["T"] > 90)
            if not found and info["y_true"] >= 0.9 * env.y_max:   # counted over steps 1..30
                hit, found = t, True
            s = s2
        rec.append((info["y_true"], ret, hit, unsafe / HORIZON))
    d = np.array(rec)
    m, se = d[:, 0].mean(), d[:, 0].std(ddof=1) / np.sqrt(N_EP)
    return {"policy": name, "mean_final_yield": m,
            "ci95_low_final_yield": m - 1.96 * se, "ci95_high_final_yield": m + 1.96 * se,
            "mean_return": d[:, 1].mean(), "mean_steps_to_90": d[:, 2].mean(),
            "unsafe_fraction": d[:, 3].mean()}


def run_all(start_for):
    return pd.DataFrame([evaluate_policy(n, a, lambda k, n=n: start_for(n, k))
                         for n, a in make_policies().items()])


print(f"Global max noise-free yield: {env.y_max:.4f} | fixed-recipe start cell {FIXED_START} "
      f"= T {env.T_vals[FIXED_START[0]]:.0f} C, C {env.C_vals[FIXED_START[1]]:.2f} mol/L")

main = run_all(lambda n, k: FIXED_START if n == "FixedRecipe" else STARTS[k])
main.to_csv("results/summary.csv", index=False)
print("\nMain comparison (Random/HillClimb/QAgent: shared random starts; FixedRecipe: recipe start)")
print(main.to_string(index=False, float_format="%.4f"))

same = run_all(lambda n, k: FIXED_START)
same.to_csv("results/summary_fixed_start.csv", index=False)
print("\nSupplementary: all policies start from the fixed-recipe cell")
print(same.to_string(index=False, float_format="%.4f"))