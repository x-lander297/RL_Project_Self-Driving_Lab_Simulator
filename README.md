# RL Project: Self-Driving Lab Simulator

This project uses **tabular Q-learning** to optimize temperature and concentration in a simulated chemical-reaction environment. The agent learns to select experimental conditions that maximize reaction yield and is compared against random, fixed-recipe, and hill-climbing baselines.

**Important:** This is a simulated system, not real chemistry. The yield function is an artificial smooth surface with Gaussian measurement noise.

## Repository

https://github.com/x-lander297/RL_Project_Self-Driving_Lab_Simulator

## Objective

Train a tabular Q-learning agent to adjust temperature and concentration on a discretized grid to maximize simulated reaction yield.

The project compares the trained agent against:

- A random-action baseline
- A fixed-recipe baseline
- A hill-climbing baseline

## Problem Setting

The environment is an 11 × 11 grid of temperature and concentration values:

- Temperature: 20 °C to 100 °C
- Concentration: 0.1 mol/L to 2.0 mol/L
- State space: 121 discrete temperature-concentration cells
- Action space: 5 actions
- Episode length: 30 steps

### Actions

| Action | Meaning |
|---|---|
| 0 | Increase temperature |
| 1 | Decrease temperature |
| 2 | Increase concentration |
| 3 | Decrease concentration |
| 4 | Hold current conditions |

Moves beyond the grid boundary leave the state unchanged.

### Reward

At each step, the agent receives the noisy measured yield as reward. A penalty of 0.2 is applied when the temperature exceeds 90 °C to represent a soft safety constraint.

The noise-free yield function has a maximum on the discretized grid of:

```text
0.9059019891
```

at approximately:

```text
Temperature: 68.0 °C
Concentration: 1.24 mol/L
```

This is the maximum available on the 11 × 11 grid, not necessarily the continuous optimum of the underlying yield function.

## Algorithm

The project uses tabular Q-learning with epsilon-greedy exploration.

The Q-table has shape:

```text
121 states × 5 actions
```

The Q-learning update rule is:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]
```

| Parameter | Value |
|---|---:|
| Learning rate, alpha | 0.1 |
| Discount factor, gamma | 0.95 |
| Initial epsilon | 1.0 |
| Final epsilon | 0.05 |
| Training episodes per seed | 3,000 |
| Steps per episode | 30 |
| Number of training seeds | 5 |

The final evaluation uses the seed-0 Q-table with greedy action selection.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project requires Python 3.10 or later and runs on CPU only.

## How to Run

Run the following commands in order:

```bash
python train.py
python evaluate.py
python plots.py
```

### 1. Train the agent

```bash
python train.py
```

This trains the Q-learning agent for 5 seeds, performs periodic greedy evaluation, and saves:

- `results/curves.csv`
- `results/qtable_seed0.npy`

### 2. Evaluate policies

```bash
python evaluate.py
```

This compares the Q-learning agent with the random, fixed-recipe, and hill-climbing baselines over 200 evaluation episodes and saves:

- `results/summary.csv`
- `results/summary_fixed_start.csv`

### 3. Generate plots

```bash
python plots.py
```

This creates:

- `results/learning_curve.png`
- `results/heatmap.png`
- `results/bar.png`

## Project Files

| File | Description |
|---|---|
| `lab_env.py` | Custom Gymnasium environment, yield function, state transitions, rewards, and episode logic |
| `agents.py` | Q-learning agent and random, fixed-recipe, and hill-climbing baselines |
| `train.py` | Trains the Q-learning agent across five seeds and records learning curves |
| `evaluate.py` | Evaluates all policies using shared environment seeds |
| `plots.py` | Generates the learning curve, policy heatmap, and baseline comparison chart |
| `report.md` | Project report, including problem formulation, methodology, results, discussion, limitations, and ethics |
| `requirements.txt` | Python dependencies |
| `results/` | CSV results, trained Q-table, and generated plots |

## Results

| Policy | Mean final yield | 95% CI | Mean return | Mean steps to 90% | Unsafe fraction |
|---|---:|---:|---:|---:|---:|
| QAgent | 0.9059 | 0.9059–0.9059 | 24.67 | 5.085 | 0.37% |
| HillClimb | 0.7549 | 0.7212–0.7886 | 15.50 | 14.67 | 9.82% |
| FixedRecipe | 0.4162 | 0.4162–0.4162 | 12.48 | 30.0 | 0% |
| Random | 0.1479 | 0.1146–0.1812 | 3.41 | 25.5 | 18.15% |

The Q-learning agent achieved the maximum yield available on the discretized grid. It outperformed all three baselines in final yield.

The hill-climbing baseline was the strongest non-learning competitor. It achieved a substantially higher final yield than the random and fixed-recipe baselines without training, but it did not match the Q-learning agent’s final yield, return, speed to the high-yield region, or unsafe-step rate.

## Learning Curve

The mean greedy evaluation yield increased from 0.596 at episode 100 to 0.881 at episode 500. It reached approximately 0.906 by episode 1,000 and remained near that level through episode 3,000.

Four of the five training seeds reached the maximum grid yield of 0.9059, while seed 4 reached 0.8992.

See `results/learning_curve.png` for the full learning curve and baseline comparison.

## Visualizations

- `results/learning_curve.png`: Mean evaluation yield across training, with baseline comparisons.
- `results/heatmap.png`: Yield surface with the learned greedy policy. Temperature is shown on the x-axis and concentration on the y-axis.
- `results/bar.png`: Final-yield comparison across all policies, with 95% confidence-interval error bars.

## Fairness Notes

- The random, hill-climbing, and Q-learning policies use shared environment seeds and start states.
- The fixed-recipe policy starts from its designated recipe cell, so its comparison reflects both its policy and starting condition.
- The main Q-learning evaluation uses the seed-0 Q-table.
- The learning curve reports variation across five training seeds.
- `summary_fixed_start.csv` provides a supplementary comparison in which all policies begin from the fixed-recipe cell.

## Limitations

- The yield function is fixed and known to the environment designer.
- The Q-learning agent learns one specific simulated landscape and does not demonstrate generalization to new reactions.
- The state space is small, discrete, and two-dimensional.
- Final evaluation uses only the seed-0 Q-table.
- Zero-width confidence intervals reflect deterministic policies evaluated on noise-free yield, not certainty.
- The fixed-recipe baseline starts from a different cell than the other policies.
- Real laboratories involve cost, safety constraints, instrumentation, time constraints, and multi-objective trade-offs that are not modeled here.

## Ethics

Autonomous laboratory optimization must include safety constraints and human oversight. In this simulation, the temperature penalty is soft; a real system would require hard safety limits, validated operating procedures, and fail-safe mechanisms.

Optimizing yield alone may ignore cost, energy use, waste, by-products, and environmental impact. Real self-driving laboratories should consider multiple objectives and constraints.

Simulated performance must not be presented as evidence of real-world chemical performance. Optimization methods can also be applied to harmful as well as beneficial chemical synthesis, so transparency, reproducibility, access controls, and human accountability are important.

## License

This project is submitted for educational purposes.