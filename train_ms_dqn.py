import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops
import os
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
from datacenter import DataCenterEnv, RUN_NOW, DELAY_FLEX, USE_BATTERY
from ms_predictor import UtilPredictor1to1

ms.set_device("CPU")


# ---------- Safety filter (keeps SLA near 0) ----------
def safe_actions(state):
    carbon_n, util_n, cool_n, soc_n, flex_n = state
    acts = [RUN_NOW]
    if flex_n >= 0.60 and util_n <= 0.70:
        acts.append(DELAY_FLEX)
    if soc_n >= 0.25:
        acts.append(USE_BATTERY)
    return acts


# ---------- MindSpore Q-network ----------
class QNet(nn.Cell):
    def __init__(self, state_dim=5, n_actions=3):
        super().__init__()
        self.net = nn.SequentialCell(
            nn.Dense(state_dim, 64),
            nn.ReLU(),
            nn.Dense(64, 64),
            nn.ReLU(),
            nn.Dense(64, n_actions)
        )

    def construct(self, x):
        return self.net(x)


# ---------- Replay buffer ----------
class ReplayBuffer:
    def __init__(self, cap=20000):
        self.cap = cap
        self.buf = []
        self.i = 0

    def push(self, s, a, r, s2, done):
        item = (s, a, r, s2, done)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.i] = item
        self.i = (self.i + 1) % self.cap

    def sample(self, bs=64):
        idx = np.random.choice(len(self.buf), bs, replace=False)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
        return (np.array(s, np.float32),
                np.array(a, np.int32),
                np.array(r, np.float32),
                np.array(s2, np.float32),
                np.array(d, np.float32))

    def __len__(self):
        return len(self.buf)

class DQNLossCell(nn.Cell):
    def __init__(self, q_net, tgt_net, gamma):
        super().__init__()
        self.q_net = q_net
        self.tgt_net = tgt_net
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.gatherd = ops.GatherD()
        self.reduce_max = ops.ReduceMax()

    def construct(self, s, a, r, s2, done):
        # Q(s,a)
        qvals = self.q_net(s)
        q_sa = self.gatherd(qvals, 1, a)

        # target = r + gamma*(1-done)*max(Q_tgt(s2))
        q2 = self.tgt_net(s2)
        q2_max = self.reduce_max(q2, 1).reshape(-1, 1)
        y = r + self.gamma * (1.0 - done) * ops.stop_gradient(q2_max)

        return self.loss_fn(q_sa, y)

# ---------- DQN Agent (MindSpore) ----------
class DQN:
    def __init__(self, state_dim=5, n_actions=3, lr=1e-3, gamma=0.99):
        self.n_actions = n_actions
        self.gamma = gamma

        self.q = QNet(state_dim, n_actions)
        self.tgt = QNet(state_dim, n_actions)
        self.update_target()

        self.opt = nn.Adam(self.q.trainable_params(), learning_rate=lr)

        self.loss_cell = DQNLossCell(self.q, self.tgt, self.gamma)
        self.train_step = nn.TrainOneStepCell(self.loss_cell, self.opt)
        self.train_step.set_train()

    def update_target(self):
        for p, tp in zip(self.q.get_parameters(), self.tgt.get_parameters()):
            tp.set_data(p.data)

    def act(self, s_np, eps, valid_actions=None):
        if np.random.rand() < eps:
            return int(np.random.choice(valid_actions)) if valid_actions else int(np.random.randint(self.n_actions))

        qvals = self.q(Tensor(s_np.reshape(1, -1), ms.float32)).asnumpy()[0]
        if not valid_actions:
            return int(np.argmax(qvals))
        return int(max(valid_actions, key=lambda a: qvals[a]))

    def train_one(self, batch):
        s, a, r, s2, done = batch

        s = Tensor(s, ms.float32)
        s2 = Tensor(s2, ms.float32)
        a = Tensor(a.reshape(-1, 1), ms.int32)
        r = Tensor(r.reshape(-1, 1), ms.float32)
        done = Tensor(done.reshape(-1, 1), ms.float32)

        loss = self.train_step(s, a, r, s2, done)
        return float(loss.asnumpy())


def run_episode(env, policy_fn):
    s = env.reset()
    total_carbon = 0.0
    total_sla = 0


    while True:
        a = policy_fn(s)
        s2, r, done, info = env.step(a)
        total_carbon += info["carbon_step"]
        total_sla += info["sla_violation"]
        if done:
            break
        s = s2
    return total_carbon, total_sla

BEST_RUN_CKPT = "best_run.ckpt"      # best within THIS run
BEST_GLOBAL_CKPT = "best_dqn.ckpt"   # best across ALL runs
BEST_SCORE_FILE = "best_score.txt"  # stores best carbon number


def save_ckpt(net, path):
    save_checkpoint(net, path)


def load_ckpt_into(net, path):
    params = load_checkpoint(path)
    load_param_into_net(net, params)


def read_best_score():
    if not os.path.exists(BEST_SCORE_FILE):
        return None
    with open(BEST_SCORE_FILE, "r") as f:
        return float(f.read().strip())


def write_best_score(v):
    with open(BEST_SCORE_FILE, "w") as f:
        f.write(str(v))


if __name__ == "__main__":
    csv_path = "data/datacenter.csv"

    # 1) MindSpore predictor (Prediction requirement)
    pred = UtilPredictor1to1()
    pred.fit(csv_path, epochs=200)

    # 2) Baseline evaluation
    env_base = DataCenterEnv(csv_path)
    base_carbon, base_sla = run_episode(env_base, lambda s: RUN_NOW)

    print(f"[BASELINE] carbon={base_carbon:.2f} sla={base_sla}")

    # 3) Train MindSpore DQN (Learning requirement)
    env = DataCenterEnv(csv_path)
    agent = DQN(state_dim=5, n_actions=3, lr=1e-3, gamma=0.99)
    buf = ReplayBuffer(cap=20000)


    def learned_policy(s):
        util_n = float(s[1])
        pu = float(pred.predict_next_util(util_n))
        pu = max(0.0, min(1.0, pu))
        s_rl = np.array([s[0], pu, s[2], s[3], s[4]], dtype=np.float32)
        valid = safe_actions(s)
        return agent.act(s_rl, eps=0.0, valid_actions=valid)


    # ---- (B) Load global best so training continues across runs ----
    prev_best = read_best_score()
    if prev_best is not None and os.path.exists(BEST_GLOBAL_CKPT):
        load_ckpt_into(agent.q, BEST_GLOBAL_CKPT)
        agent.update_target()
        print(f"[Loaded GLOBAL best] carbon={prev_best:.2f}")
    else:
        print("[No GLOBAL best found] starting fresh")

    episodes = 300
    batch_size = 64
    warmup = 150
    eps = 0.30
    target_every = 25

    best_carbon_run = float("inf")
    best_sla_run = 999

    for ep in range(episodes):
        s = env.reset()
        ep_sla = 0
        ep_carbon = 0.0
        a_counts = [0, 0, 0]

        while True:
            # use predictor to build a "forecast-aware" state for RL
            util_n = float(s[1])
            pu = float(pred.predict_next_util(util_n))
            pu = max(0.0, min(1.0, pu))
            s_rl = np.array([s[0], pu, s[2], s[3], s[4]], dtype=np.float32)

            valid = safe_actions(s)
            a = agent.act(s_rl, eps=eps, valid_actions=valid)

            a_counts[a] += 1

            s2, r, done, info = env.step(a)
            ep_sla += info["sla_violation"]
            ep_carbon += info["carbon_step"]

            if done:
                s2_rl = np.zeros_like(s_rl, dtype=np.float32)
            else:
                util2 = float(s2[1])
                pu2 = float(pred.predict_next_util(util2))
                pu2 = max(0.0, min(1.0, pu2))
                s2_rl = np.array([s2[0], pu2, s2[2], s2[3], s2[4]], dtype=np.float32)

            buf.push(s_rl, a, r, s2_rl, float(done))

            if len(buf) >= max(warmup, batch_size):
                batch = buf.sample(batch_size)
                agent.train_one(batch)

            if done:
                break
            s = s2

        eps = max(0.05, eps * 0.995)
        if (ep + 1) % target_every == 0:
            agent.update_target()

        if (ep + 1) % 50 == 0:
            print(f"[ACTIONS] RUN={a_counts[0]} DELAY={a_counts[1]} BATT={a_counts[2]}")
            print(f"ep={ep + 1} eps={eps:.3f} ep_carbon={ep_carbon:.2f} ep_sla={ep_sla} buffer={len(buf)}")

            # ---- (A) Evaluate and keep best within THIS run ----
            env_tmp = DataCenterEnv(csv_path)
            eval_carbon, eval_sla = run_episode(env_tmp, learned_policy)
            print(f"[EVAL] ep={ep + 1} carbon={eval_carbon:.2f} sla={eval_sla}")

            if eval_sla == 0 and eval_carbon < best_carbon_run:
                best_carbon_run = eval_carbon
                best_sla_run = eval_sla
                save_ckpt(agent.q, BEST_RUN_CKPT)
                print(f"[BEST RUN] saved {BEST_RUN_CKPT} carbon={best_carbon_run:.2f}")
    # ================= STEP 6: PROMOTION =================
    # Promote best of THIS run to GLOBAL BEST if it beats global best
    prev_best_global = read_best_score()

    if os.path.exists(BEST_RUN_CKPT) and best_sla_run == 0:
        if (prev_best_global is None) or (best_carbon_run < prev_best_global):
            # Promote best_run -> global best
            load_ckpt_into(agent.q, BEST_RUN_CKPT)
            agent.update_target()
            save_ckpt(agent.q, BEST_GLOBAL_CKPT)
            write_best_score(best_carbon_run)
            print(f"[PROMOTED] New GLOBAL BEST! carbon={best_carbon_run:.2f}")
        else:
            print(f"[PROMOTED] Not better than global. global={prev_best_global:.2f} run_best={best_carbon_run:.2f}")
    else:
        print("[PROMOTED] No valid best_run to promote (or SLA violation)")
    # =====================================================

    # Step 5 (SERVING): always load GLOBAL BEST for system use / visualization
    if os.path.exists(BEST_GLOBAL_CKPT):
        load_ckpt_into(agent.q, BEST_GLOBAL_CKPT)
        agent.update_target()
        print("[SERVING] Loaded GLOBAL BEST policy (best_dqn.ckpt)")
    else:
        print("[SERVING] No global best yet, using current agent weights")

    env_eval = DataCenterEnv(csv_path)
    learned_carbon, learned_sla = run_episode(env_eval, learned_policy)

    
    reduction = (base_carbon - learned_carbon) / base_carbon * 100.0

    print("\n===== BASELINE vs MindSpore DQN =====")
    print(f"Baseline carbon: {base_carbon:.2f} | SLA: {base_sla}")
    print(f"Learned carbon:  {learned_carbon:.2f} | SLA: {learned_sla}")
    print(f"Reduction:       {reduction:.2f}%")
