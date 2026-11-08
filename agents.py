"""Q-learning agent and three baselines. All policies expose act(state, greedy=False)."""
import numpy as np

OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}  # inverse of each action


class QAgent:
    def __init__(self, n_states=121, n_actions=5, alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, decay_frac=0.6, seed=0):
        self.q = np.zeros((n_states, n_actions))
        self.n_actions, self.alpha, self.gamma = n_actions, alpha, gamma
        self.eps_start, self.eps_end, self.decay_frac = eps_start, eps_end, decay_frac
        self.epsilon = eps_start
        self.rng = np.random.default_rng(seed)

    def act(self, state, greedy=False):
        # `greedy or ...` short-circuits, so greedy eval never consumes the training RNG
        if greedy or self.rng.random() >= self.epsilon:
            return int(np.argmax(self.q[state]))
        return int(self.rng.integers(self.n_actions))

    def update(self, state, action, reward, next_state):
        # Bootstrap always: episodes end by truncation (time limit), not termination
        td_target = reward + self.gamma * self.q[next_state].max()
        self.q[state, action] += self.alpha * (td_target - self.q[state, action])

    def decay_epsilon(self, episode, n_episodes):
        # Linear decay from eps_start to eps_end over the first 60% of episodes
        frac = min(1.0, episode / (self.decay_frac * n_episodes))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)


class RandomAgent:
    def __init__(self, n_actions=5, seed=0):
        self.n_actions, self.rng = n_actions, np.random.default_rng(seed)

    def act(self, state, greedy=False):
        return int(self.rng.integers(self.n_actions))


class FixedRecipeAgent:
    def act(self, state, greedy=False):
        return 4  # hold


class HillClimbAgent:
    """Greedy local search on observed (noisy) reward with a fixed action cycle."""
    CYCLE = (0, 2, 1, 3)  # T up, C up, T down, C down

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = None          # current state
        self.prev_reward = None    # reward at the reference cell
        self.last_action = None
        self.idx = 0               # position in the action cycle
        self.undo = None           # pending undo action
        self.undoing = False       # True while executing an undo step

    def act(self, state, greedy=False):
        self.state = state
        if self.undo is not None:                      # revert the worsening move first
            action, self.undo, self.undoing = self.undo, None, True
        else:                                          # keep repeating the current action while it helps
            action = self.CYCLE[self.idx % len(self.CYCLE)]
        self.last_action = action
        return action

    def update(self, state, action, reward, next_state):
        if self.undoing:                   # back at the reference cell: refresh baseline, no comparison
            self.undoing, self.prev_reward = False, reward
        elif next_state == state:          # blocked by boundary: nothing moved, nothing to undo
            self.idx += 1
            self.prev_reward = reward
        elif self.prev_reward is not None and reward < self.prev_reward:
            self.undo = OPPOSITE[action]   # worse: undo next step, then try the next cycle action
            self.idx += 1
        else:                              # first step or improvement: accept and continue
            self.prev_reward = reward
        self.state = next_state