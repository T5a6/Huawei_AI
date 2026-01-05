from datacenter import DataCenterEnv
from policies import baseline_policy, carb_policy

def run_episode(env, policy_fn):
    state = env.reset()
    total_carbon = 0.0
    total_sla = 0
    carbon_series = []
    sla_series = []
    action_series = []

    while True:
        action = policy_fn(state)
        next_state, reward, done, info = env.step(action)

        total_carbon += info["carbon_step"]
        total_sla += info["sla_violation"]

        carbon_series.append(info["carbon_step"])
        sla_series.append(info["sla_violation"])
        action_series.append(action)

        if done:
            break
        state = next_state

    return total_carbon, total_sla, carbon_series, sla_series, action_series


if __name__ == "__main__":
    csv_path = "data/datacenter.csv"

    # BEFORE: baseline
    env_baseline = DataCenterEnv(csv_path)
    base_carbon, base_sla, base_carbon_series, base_sla_series, base_actions = run_episode(
        env_baseline, baseline_policy
    )

    # AFTER: CARB
    env_carb = DataCenterEnv(csv_path)
    carb_carbon, carb_sla, carb_carbon_series, carb_sla_series, carb_actions = run_episode(
        env_carb, carb_policy
    )

    reduction_pct = (base_carbon - carb_carbon) / base_carbon * 100.0

    print("\n===== BEFORE vs AFTER =====")
    print(f"Baseline total carbon: {base_carbon:.2f}")
    print(f"CARB total carbon:     {carb_carbon:.2f}")
    print(f"Carbon reduction:      {reduction_pct:.2f}%")
    print(f"Baseline SLA violations: {base_sla}")
    print(f"CARB SLA violations:     {carb_sla}")
