# Self-Driving Lab Simulator: Tabular Q-Learning for Reaction-Yield Optimization

## Objective
Train a tabular Q-learning agent to adjust temperature and concentration on a discretized grid to maximize a
simulated reaction yield, and compare it against random, fixed-recipe, and hill-climbing baselines.
**This is a simulated system, not real chemistry**: the yield function is a made-up smooth surface with noise.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Requires Python 3.10+, CPU only.

## Run order
```bash
python train.py
python evaluate.py
python plots.py
```

## Files
- `requirements.txt`: dependencies (numpy, gymnasium, pandas, matplotlib).
- `lab_env.py`: `LabEnv`, an 11x11 T-C grid (T 20-100 °C, C 0.1-2.0 mol/L), 5 actions, 30-step episodes,
  noisy yield with a safety penalty for T > 90 °C.
- `agents.py`: `QAgent`, `RandomAgent`, `FixedRecipeAgent`, `HillClimbAgent`.
- `train.py`: trains Q-learning for 5 seeds x 3000 episodes; periodic greedy evaluation.
- `evaluate.py`: 200-episode comparison of all four policies with shared env seeds.
- `plots.py`: learning curve, policy heatmap, bar chart.

## Q-learning in brief
The agent keeps a table Q(state, action) of expected discounted return. After each step it moves Q toward
`reward + gamma * max_a' Q(next_state, a')` with step size alpha=0.1 (gamma=0.95). Exploration is epsilon-greedy,
with epsilon decaying linearly from 1.0 to 0.05 over the first 60% of episodes. Episodes end by time limit, so the
update always bootstraps from the next state. The agent optimizes cumulative reward over 30 steps, which is not
identical to maximizing the final-step yield.

## Baselines
- **Random:** uniformly random actions.
- **FixedRecipe:** always holds; evaluated from the grid cell nearest T=50 °C, C=1.0 mol/L (T=52 °C, C=1.05 mol/L).
- **HillClimb:** tries an action; if the observed reward drops, undoes it and tries the next action in the cycle
  (T up, C up, T down, C down); keeps repeating an action while it does not hurt. Moves blocked by the grid
  boundary are detected and not "undone".

## Outputs (in `results/`)
- `curves.csv`: seed, episode, train_return, eval_final_yield, eval_return (eval values are means over 50 greedy episodes).
- `qtable_seed0.npy`: final Q-table for seed 0.
- `summary.csv`: per-policy metrics (final yield with 95% CI, return, steps to 90% of max, unsafe fraction).
- `summary_fixed_start.csv`: the same metrics with every policy starting at the fixed-recipe cell.
- `learning_curve.png`, `heatmap.png`, `bar.png`.

## Fairness notes
Episode k uses the same environment seed for every policy. Random, HillClimb and QAgent share the same random start
cells; FixedRecipe starts from its recipe cell by definition. The supplementary table starts every policy there.
The reported Q-learning results use the seed-0 Q-table only; the learning curve shows variation across 5 seeds.