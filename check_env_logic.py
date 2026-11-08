import numpy as np
from gymnasium.utils.env_checker import check_env

from lab_env import LabEnv

env = LabEnv()

print("Running Gymnasium environment checker...")
check_env(env, skip_render_check=True)
print("Environment passed Gymnasium checks.")

print("\n--- Manual logic checks ---")

# Check spaces
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Check reset returns the Gymnasium 5-tuple-compatible structure
obs, info = env.reset(seed=42)
print("Reset observation:", obs)
print("Reset info keys:", info.keys())

# Check fixed start
obs_fixed, info_fixed = env.reset(seed=42, options={"start": (3, 3)})
print("Fixed-start observation:", obs_fixed)
print("Fixed-start info:", info_fixed)

# Check episode length and truncation
steps = 0
truncated = False
terminated = False

while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(4)  # hold
    steps += 1

print("Steps taken:", steps)
print("terminated:", terminated)
print("truncated:", truncated)
print("Final info:", info)

# Check boundary behavior: repeatedly move temperature down from the lowest row
obs, info = env.reset(seed=1, options={"start": (0, 0)})
for _ in range(5):
    obs, reward, terminated, truncated, info = env.step(1)  # decrease T
print("State after repeated T-down at lower boundary:", obs)
print("Final info after boundary test:", info)

# Check reproducibility: same seed should give the same first noisy reward
env1 = LabEnv()
env2 = LabEnv()

obs1, info1 = env1.reset(seed=123)
obs2, info2 = env2.reset(seed=123)

reward1 = env1.step(0)[1]
reward2 = env2.step(0)[1]

print("Reward with seed 123, run 1:", reward1)
print("Reward with seed 123, run 2:", reward2)
print("Reproducible:", np.isclose(reward1, reward2))