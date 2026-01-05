import numpy as np

class QLearningAgent:
    """
    Tabular Q-learning for small action space.
    We discretize continuous state into bins, then learn Q(s,a).
    """

    def __init__(self, n_actions=3, alpha=0.2, gamma=0.95, eps=0.2):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        # bins for discretization (tune later if needed)
        # state = [carbon_n, util_n, cool_n, soc_n, flex_n] in 0..1
        self.bins = {
            "carbon": np.array([0.35, 0.55, 0.75]),  # low/med/high/veryhigh
            "util":   np.array([0.40, 0.60, 0.80]),
            "soc":    np.array([0.25, 0.50, 0.75]),
            "flex":   np.array([0.50, 0.70, 0.85]),
        }

        # Q-table shape = carbon_bins+1 x util_bins+1 x soc_bins+1 x flex_bins+1 x actions
        self.Q = np.zeros((4, 4, 4, 4, n_actions), dtype=np.float32)

    def _bin(self, x, edges):
        # returns 0..len(edges)
        return int(np.digitize([x], edges)[0])

    def state_to_idx(self, state, pred_next_util=None):
        carbon_n, util_n, cool_n, soc_n, flex_n = state

        # use predicted util if provided (this makes learning use MindSpore)
        u = float(pred_next_util) if pred_next_util is not None else float(util_n)

        c_i = self._bin(float(carbon_n), self.bins["carbon"])
        u_i = self._bin(u,              self.bins["util"])
        s_i = self._bin(float(soc_n),   self.bins["soc"])
        f_i = self._bin(float(flex_n),  self.bins["flex"])
        return (c_i, u_i, s_i, f_i)

    def choose_action(self, state_idx, valid_actions=None):
        # epsilon-greedy
        if np.random.rand() < self.eps:
            if valid_actions is None:
                return np.random.randint(self.n_actions)
            return int(np.random.choice(valid_actions))

        q = self.Q[state_idx]
        if valid_actions is None:
            return int(np.argmax(q))

        # choose best among valid_actions
        best_a = max(valid_actions, key=lambda a: q[a])
        return int(best_a)

    def update(self, s_idx, a, r, s2_idx, done):
        q_sa = self.Q[s_idx][a]
        if done:
            target = r
        else:
            target = r + self.gamma * float(np.max(self.Q[s2_idx]))
        self.Q[s_idx][a] = q_sa + self.alpha * (target - q_sa)
