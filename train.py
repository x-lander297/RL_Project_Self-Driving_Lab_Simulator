"""Train tabular Q-learning for 5 seeds; log learning curves; save seed-0 Q-table."""
import os
import time
import numpy as np
import pandas as pd
from lab_env import LabEnv
from agents import QAgent

N_SEEDS, N_EPISODES = 5, 3000
EVAL_EVERY, EVAL_EPISODES, EVAL_SEED_BASE = 100, 50, 10_000


def run_episode(env, agent, learn, greedy, seed=None):
    """Run one episode; return (sum of rewards, noise-free yield at final step)."""
    s, _ = env.reset(seed=seed)
    ret = 0.0
    while True:
        a = agent.act(s, greedy=greedy)
        s2, r, terminated, truncated, info = env.step(a)
        if learn:
            agent.update(s, a, r, s2)
        ret += r
        s = s2
        if terminated or truncated:
            return ret, info["y_true"]


os.makedirs("results", exist_ok=True)
rows, t0 = [], time.time()
for seed in range(N_SEEDS):
    env, eval_env, agent = LabEnv(), LabEnv(), QAgent(seed=seed)
    for ep in range(N_EPISODES):
        agent.decay_epsilon(ep, N_EPISODES)
        train_ret, _ = run_episode(env, agent, learn=True, greedy=False, seed=seed if ep == 0 else None)
        if (ep + 1) % EVAL_EVERY == 0:
            # Same eval seeds at every checkpoint -> same start cells and noise streams
            res = [run_episode(eval_env, agent, learn=False, greedy=True, seed=EVAL_SEED_BASE + k)
                   for k in range(EVAL_EPISODES)]
            rows.append({"seed": seed, "episode": ep + 1, "train_return": train_ret,
                         "eval_final_yield": float(np.mean([r[1] for r in res])),
                         "eval_return": float(np.mean([r[0] for r in res]))})
        if (ep + 1) % 500 == 0:
            print(f"seed {seed} | episode {ep + 1}/{N_EPISODES} | epsilon {agent.epsilon:.2f} "
                  f"| elapsed {time.time() - t0:.1f}s")
    if seed == 0:
        np.save("results/qtable_seed0.npy", agent.q)

pd.DataFrame(rows).to_csv("results/curves.csv", index=False)
print(f"Saved results/curves.csv and results/qtable_seed0.npy. Total runtime: {time.time() - t0:.1f}s")