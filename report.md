# Tabular Q-Learning for Reaction-Yield Optimization in a Simulated Self-Driving Lab

## Abstract

This project investigates whether a tabular Q-learning agent can learn to select temperature and concentration settings that maximize the yield of a simulated chemical reaction. The environment is an 11 × 11 grid of temperature and concentration values, with a hidden smooth yield function, Gaussian measurement noise, and a penalty for operating above 90 °C. The agent uses tabular Q-learning with epsilon-greedy exploration and is compared against random, fixed-recipe, and hill-climbing baselines. The Q-learning agent achieved a mean final noise-free yield of 0.9059, equal to the maximum yield available on the discretized grid. It outperformed all baselines in final yield, although the hill-climbing baseline was a strong training-free competitor. The results demonstrate learning within a fixed simulated landscape and do not establish generalization to new reactions or real laboratory systems.

## 1. Introduction

Self-driving laboratories aim to reduce the time and cost of chemical experimentation by automatically selecting and evaluating experimental conditions. In many laboratory settings, each experiment provides limited and noisy feedback, making sequential decision-making important. Reinforcement learning (RL) is therefore a natural framework for studying how an agent can improve its choices through interaction with an environment.

This project tests whether a minimal RL agent can learn high-yield reaction conditions in a simulated self-driving-lab setting. The agent adjusts temperature and concentration over a fixed experimental budget and is evaluated against simple non-learning strategies. The objective is not to model real chemistry, but to create a transparent, reproducible RL problem in which the agent must learn from noisy feedback.

## 2. Problem Formulation

### Environment

The environment, `LabEnv`, is a custom Gymnasium environment representing a simulated chemical reaction. The agent controls two experimental variables:

- Temperature: 11 evenly spaced values from 20 °C to 100 °C.
- Concentration: 11 evenly spaced values from 0.1 mol/L to 2.0 mol/L.

The resulting state space contains 121 discrete temperature-concentration cells. Each episode represents an experiment budget of 30 steps. Episodes end by truncation after 30 steps, rather than by a task-specific termination condition.

### State Space

The state is a discrete index representing the current temperature-concentration cell:

\[
s = 11i + j
\]

where \(i\) indexes the temperature level and \(j\) indexes the concentration level. The state space is `Discrete(121)`. The agent observes only the current cell; it does not directly observe the underlying yield function or the number of steps remaining.

### Action Space

The action space is `Discrete(5)`:

| Action | Meaning |
|---|---|
| 0 | Increase temperature by one grid step |
| 1 | Decrease temperature by one grid step |
| 2 | Increase concentration by one grid step |
| 3 | Decrease concentration by one grid step |
| 4 | Hold the current conditions |

Moves beyond the boundary of the grid leave the state unchanged.

### Reward Function

At each step, the environment computes the noise-free yield from the current temperature and concentration, adds Gaussian measurement noise with standard deviation 0.03, and clips the observed yield to the range [0, 1]. The reward is the noisy observed yield, with an additional penalty of 0.2 when the temperature exceeds 90 °C.

\[
r_t = \text{clip}(Y_{\text{true}}(T,C) + \mathcal{N}(0, 0.03), 0, 1) - 0.2\mathbf{1}_{T > 90}
\]

The temperature penalty represents a soft safety constraint. It discourages operation in the high-temperature region but does not completely prevent the agent from entering it.

### Learning Algorithm

The project uses tabular Q-learning with epsilon-greedy exploration. The Q-table has dimensions 121 × 5, corresponding to 121 states and five actions.

The Q-learning update rule is:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
\]

The following hyperparameters were used:

| Parameter | Value |
|---|---:|
| Learning rate, \(\alpha\) | 0.1 |
| Discount factor, \(\gamma\) | 0.95 |
| Initial epsilon | 1.0 |
| Final epsilon | 0.05 |
| Epsilon decay | Linear over first 60% of episodes |
| Training episodes per seed | 3,000 |
| Steps per episode | 30 |
| Number of training seeds | 5 |

The final evaluation used the seed-0 Q-table with greedy action selection.

### Baselines

Three baselines were used:

1. **Random policy:** selects uniformly among the five available actions.
2. **Fixed-recipe policy:** holds at the grid cell nearest 50 °C and 1.0 mol/L, corresponding to 52 °C and 1.05 mol/L.
3. **Hill-climbing policy:** attempts an action, reverts it if the observed reward decreases, and then proceeds through a fixed action cycle.

The random, hill-climbing, and Q-learning policies were evaluated using shared environment seeds and start states. The fixed-recipe policy began from its own recipe cell.

## 3. Implementation

The implementation consists of the following files:

| File | Purpose |
|---|---|
| `lab_env.py` | Defines the simulated reaction environment, yield function, state transitions, rewards, and episode termination logic |
| `agents.py` | Implements the Q-learning agent and the random, fixed-recipe, and hill-climbing baselines |
| `train.py` | Trains the Q-learning agent across five seeds and records learning-curve data |
| `evaluate.py` | Evaluates all policies under a common evaluation procedure and writes `summary.csv` |
| `plots.py` | Generates the learning curve, yield heatmap with learned policy, and baseline comparison chart |
| `requirements.txt` | Lists the Python dependencies |
| `README.md` | Explains installation, execution, outputs, and project structure |

The project uses only CPU-based computation and standard Python packages: NumPy, Gymnasium, pandas, and Matplotlib. Training required 3,000 episodes per seed, with 30 steps per episode, giving 90,000 environment interactions per seed.

## 4. Results

### Final performance

| Policy | Mean final yield | 95% CI | Mean return | Mean steps to 90% | Unsafe fraction |
|---|---:|---:|---:|---:|---:|
| QAgent | 0.9059 | 0.9059–0.9059 | 24.67 | 5.085 | 0.37% |
| HillClimb | 0.7549 | 0.7212–0.7886 | 15.50 | 14.67 | 9.82% |
| FixedRecipe | 0.4162 | 0.4162–0.4162 | 12.48 | 30.0 | 0% |
| Random | 0.1479 | 0.1146–0.1812 | 3.41 | 25.5 | 18.15% |

The Q-learning agent achieved a mean final noise-free yield of 0.9059. This equals the maximum yield available on the 11 × 11 discretized grid, which is 0.9059019891 at approximately 68 °C and 1.24 mol/L. Therefore, the Q-learning agent reached the best available grid cell in every evaluated episode.

### Learning curve

Across five training seeds, the mean greedy evaluation yield increased from 0.596 at episode 100 to 0.756 at episode 200, 0.841 at episode 300, and 0.881 at episode 500. The agent reached approximately 0.906 by episode 1,000 and remained between 0.900 and 0.906 through episode 3,000.

At the final training checkpoint, four of the five seeds reached a final yield of 0.9059, while seed 4 reached 0.8992. The mean final yield across seeds was 0.9046, with a standard deviation of 0.0030. The learning curve therefore shows rapid early improvement followed by a stable plateau.

The learning curve is shown in `results/learning_curve.png`.

### Baseline comparison

The Q-learning agent outperformed the random and hill-climbing baselines in final yield. Its final yield was 0.9059, compared with 0.7549 for hill-climbing and 0.1479 for the random policy. The confidence intervals for the hill-climbing and random baselines did not overlap the Q-learning agent’s result.

The hill-climbing baseline was the strongest non-learning competitor. Without training, it achieved a final yield of 0.7549, substantially above the fixed-recipe baseline and random baseline. However, it reached only approximately 83% of the Q-learning agent’s final yield and earned 15.50 mean return, compared with 24.67 for Q-learning.

The hill-climbing agent also required 14.67 steps on average to first reach at least 90% of the maximum available yield, compared with 5.085 steps for the Q-learning agent. It spent 9.82% of its steps in the unsafe temperature region, whereas the Q-learning agent spent 0.37%.

The final-yield comparison is shown in `results/bar.png`. The learned greedy policy is visualized on the yield surface in `results/heatmap.png`, where temperature is shown on the x-axis and concentration on the y-axis.

## 5. Discussion

The results indicate that tabular Q-learning can learn an effective policy in a small, noisy, sequential optimization problem. The agent learned to move toward the high-yield region and ultimately reached the best cell available on the discretized grid.

The hill-climbing baseline was credible because it required no prior training and still achieved a much higher yield than the random and fixed-recipe baselines. However, it was not competitive with Q-learning in this experiment. It achieved 83% of the Q-learning agent’s final yield, required approximately three times as many steps to reach the 90% yield threshold, and earned 63% of the Q-learning agent’s mean return.

One possible explanation for the hill-climber’s weaker performance is that it evaluates individual noisy reward observations and may reverse or repeat actions because of measurement noise. This explanation is a hypothesis and was not separately tested. The Q-learning agent also benefited from 90,000 training interactions per seed, whereas the hill-climbing baseline used no prior training experience.

The Q-learning agent’s unsafe-step fraction of 0.37% arose only from starting cells at 100 °C. From those states, the agent’s first available move still left the temperature above 90 °C, making those unsafe steps unavoidable under the environment’s action structure.

## 6. Grid Maximum and Continuous Optimum

The maximum noise-free yield on the 11 × 11 grid was 0.9059019891, obtained at 68 °C and 1.24 mol/L. This is the **grid maximum**, not necessarily the continuous optimum of the underlying yield function.

The underlying yield function has a continuous maximum of 0.95 at 65 °C and 1.2 mol/L. The discretized grid cannot represent that exact point because 65 °C lies between the 60 °C and 68 °C grid values, while 1.2 mol/L lies between the 1.05 mol/L and 1.24 mol/L grid values.

The grid maximum represents approximately 95.4% of the continuous optimum. Therefore, the Q-learning agent is described as **optimal on the discretized grid**, not as having found the true continuous optimum.

## 7. Limitations

- The yield function is fixed and known to the environment designer. The Q-learning agent memorizes a specific landscape rather than demonstrating generalization to new reactions.
- The environment uses a coarse 11 × 11 grid, whereas real experimental conditions are continuous and may involve many more variables.
- The final evaluation used only the seed-0 Q-table. The zero-width confidence interval for QAgent reflects deterministic greedy behavior on noise-free yield, not certainty across training runs.
- The fixed-recipe baseline starts from a different cell than the other policies, so its comparison combines differences in policy and starting condition.
- Confidence intervals were reported only for final yield. The return, steps-to-90%, and unsafe-fraction results are descriptive and do not include confidence intervals.
- Training required 90,000 environment interactions per seed. This is unrealistic for many real laboratory settings, where experiments may be expensive, slow, or limited.
- The hill-climbing baseline demonstrates that simple non-learning methods can perform well in some optimization settings. RL is not automatically the best approach for every optimization problem.

## 8. Ethics

Autonomous laboratory optimization raises important safety and responsibility concerns. In this project, the temperature penalty is a soft constraint, meaning that the agent can still enter the unsafe region. A real laboratory system would require hard safety constraints, validated operating limits, human oversight, and fail-safe mechanisms.

Optimizing yield alone may also ignore other important objectives, including cost, energy use, waste generation, by-product formation, and environmental impact. A real self-driving lab should therefore use a multi-objective reward function and explicit safety constraints rather than optimizing a single metric.

The results of this simulated project must not be presented as evidence of real-world chemical performance. The environment uses an artificial yield function and does not model real reaction kinetics, instrumentation, experimental variability, or safety risks.

Finally, autonomous optimization technology can accelerate both beneficial and harmful chemical synthesis. Transparency, reproducibility, access controls, and human accountability are important safeguards. This project supports reproducibility by using fixed seeds, shared evaluation conditions, and publicly documented code.

## 9. Conclusion

Tabular Q-learning successfully learned a policy that reached the maximum yield available on the simulated 11 × 11 temperature-concentration grid. The agent achieved a final noise-free yield of 0.9059, exceeding the random, fixed-recipe, and hill-climbing baselines.

The hill-climbing baseline was a strong training-free alternative, but it achieved lower final yield, required more steps to reach the high-yield region, and incurred more unsafe-temperature steps. The project demonstrates the potential of RL for sequential experimental optimization, while also showing that its advantage depends on the problem structure and should be compared carefully against simpler baselines.

Future work could evaluate the agent on changing yield functions, continuous action spaces, multi-objective rewards, real experimental constraints, and transfer across related reactions.

## 10. Reproducibility

To reproduce the project:

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python plots.py
```

The main outputs are:

- `results/curves.csv`: training and evaluation learning-curve data.
- `results/summary.csv`: final policy comparison.
- `results/qtable_seed0.npy`: Q-table for seed 0.
- `results/learning_curve.png`: learning curve with baseline comparisons.
- `results/heatmap.png`: yield surface with learned greedy policy.
- `results/bar.png`: final-yield comparison across policies.