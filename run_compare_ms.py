from datacenter import DataCenterEnv, RUN_NOW, DELAY_FLEX, USE_BATTERY
from ms_predictor import UtilPredictor1to1

def baseline_policy(state):
    return RUN_NOW

def carb_policy_with_prediction(state, pred_next_util,
                               carbon_high=0.50,
                               carbon_very_high=0.70,
                               flex_high=0.65,
                               soc_ok=0.35,
                               util_now_safe=0.70,
                               util_pred_safe=0.70):
    """
    CARB decision using MindSpore prediction:
    - If predicted utilization will be high, avoid DELAY to protect SLA
    - Battery is preferred when carbon is very high
    """
    carbon_n, util_n, cool_n, soc_n, flex_n = state

    # 1) very high carbon => battery first (no SLA risk)
    if carbon_n > carbon_very_high and soc_n > soc_ok:
        return USE_BATTERY

    # 2) delay only if: carbon high + flexible + current util safe + predicted util safe
    if carbon_n > carbon_high and flex_n > flex_high and util_n < util_now_safe and pred_next_util < util_pred_safe:
        return DELAY_FLEX

    return RUN_NOW

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

if __name__ == "__main__":
    csv_path = "data/datacenter.csv"

    # 1) Train MindSpore predictor (fast)
    predictor = UtilPredictor1to1()
    predictor.fit(csv_path, epochs=200)

    # 2) Baseline run
    env_base = DataCenterEnv(csv_path)
    base_carbon, base_sla = run_episode(env_base, baseline_policy)

    # 3) CARB run with MindSpore prediction
    def policy_ms(state):
        util_n = float(state[1])  # state = [carbon_n, util_n, cool_n, soc_n, flex_n]
        pred = predictor.predict_next_util(util_n)
        return carb_policy_with_prediction(state, pred)

    env_carb = DataCenterEnv(csv_path)
    carb_carbon, carb_sla = run_episode(env_carb, policy_ms)

    reduction_pct = (base_carbon - carb_carbon) / base_carbon * 100.0

    print("\n===== BEFORE vs AFTER (MindSpore Prediction) =====")
    print(f"Baseline total carbon: {base_carbon:.2f}")
    print(f"CARB total carbon:     {carb_carbon:.2f}")
    print(f"Carbon reduction:      {reduction_pct:.2f}%")
    print(f"Baseline SLA violations: {base_sla}")
    print(f"CARB SLA violations:     {carb_sla}")
