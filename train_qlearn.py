import numpy as np
from datacenter import DataCenterEnv, RUN_NOW, DELAY_FLEX, USE_BATTERY
from ms_predictor import UtilPredictor1to1
from qlearn_agent import QLearningAgent


def baseline_policy(state):
    return RUN_NOW


def safe_actions(state):
    """
    Hard safety filter (keeps SLA violations near 0).
    RL can only choose actions that are safe in this step.
    """
    carbon_n, util_n, cool_n, soc_n, flex_n = state

    actions = [RUN_NOW]

    # Delay allowed only when flexible + not high util
    if flex_n >= 0.60 and util_n <= 0.70:
        actions.append(DELAY_FLEX)

    # Battery allowed only if SOC ok
    if soc_n >= 0.25:
        actions.append(USE_BATTERY)

    return actions


def run_episode(env, policy_fn):
    state = env.reset()
    total_carbon = 0.0
    total_sla = 0

    while True:
        action = policy_fn(state)
        next_state, reward, done, info = env.step(action)
        total_carbon += info["carbon_step"]
        total_sla += info["sla_violation"]
        if done:
            break
        state = next_state

    return total_carbon, total_sla


def train_qlearning(env, predictor, episodes=300):
    agent = QLearningAgent(n_actions=3, alpha=0.2, gamma=0.95, eps=0.25)

    for ep in range(episodes):
        state = env.reset()

        while True:
            util_n = float(state[1])
            pred_util = predictor.predict_next_util(util_n)
            pred_util = max(0.0, min(1.0, pred_util))

            s_idx = agent.state_to_idx(state, pred_next_util=pred_util)

            valid = safe_actions(state)
            a = agent.choose_action(s_idx, valid_actions=valid)

            next_state, reward, done, info = env.step(a)

            if done:
                agent.update(s_idx, a, reward, s_idx, done=True)
                break

            util2 = float(next_state[1])
            pred_util2 = predictor.predict_next_util(util2)
            pred_util2 = max(0.0, min(1.0, pred_util2))
            s2_idx = agent.state_to_idx(next_state, pred_next_util=pred_util2)

            agent.update(s_idx, a, reward, s2_idx, done=False)

            state = next_state

        # slowly reduce exploration
        agent.eps = max(0.05, agent.eps * 0.995)

    return agent


def evaluate_learned(env, predictor, agent):
    def learned_policy(state):
        util_n = float(state[1])
        pred_util = predictor.predict_next_util(util_n)
        pred_util = max(0.0, min(1.0, pred_util))

        s_idx = agent.state_to_idx(state, pred_next_util=pred_util)
        valid = safe_actions(state)
        return agent.choose_action(s_idx, valid_actions=valid)

    return run_episode(env, learned_policy)


if __name__ == "__main__":
    csv_path = "data/datacenter.csv"

    # 1) Train MindSpore predictor (required)
    predictor = UtilPredictor1to1()
    predictor.fit(csv_path, epochs=200)

    # 2) Baseline
    env_base = DataCenterEnv(csv_path)
    base_carbon, base_sla = run_episode(env_base, baseline_policy)

    # 3) Train Q-learning agent in simulation (Learn step)
    env_train = DataCenterEnv(csv_path)
    agent = train_qlearning(env_train, predictor, episodes=300)

    # 4) Evaluate learned policy
    env_eval = DataCenterEnv(csv_path)
    learned_carbon, learned_sla = evaluate_learned(env_eval, predictor, agent)

    reduction_pct = (base_carbon - learned_carbon) / base_carbon * 100.0

    print("\n===== BASELINE vs LEARNED (MindSpore + Q-learning) =====")
    print(f"Baseline total carbon: {base_carbon:.2f}")
    print(f"Learned total carbon:  {learned_carbon:.2f}")
    print(f"Carbon reduction:      {reduction_pct:.2f}%")
    print(f"Baseline SLA violations: {base_sla}")
    print(f"Learned SLA violations:  {learned_sla}")
