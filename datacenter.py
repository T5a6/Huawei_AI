import pandas as pd
import numpy as np

RUN_NOW = 0
DELAY_FLEX = 1
USE_BATTERY = 2

class DataCenterEnv:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

        # parse timestamp if needed (safe)
        if "timestamp" in self.df.columns:
            try:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], format="%d/%m/%Y %H:%M")
            except Exception:
                self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])

        # ----- normalize the 5 state columns (do it here to be safe) -----
        # carbon & cooling min-max
        cmin, cmax = self.df["grid_carbon_intensity"].min(), self.df["grid_carbon_intensity"].max()
        lmin, lmax = self.df["cooling_load"].min(), self.df["cooling_load"].max()

        self.df["carbon_n"] = (self.df["grid_carbon_intensity"] - cmin) / (cmax - cmin + 1e-9)
        self.df["cool_n"]   = (self.df["cooling_load"] - lmin) / (lmax - lmin + 1e-9)

        # utilization + SOC to 0–1
        self.df["util_n"] = self.df["server_utilization"] / 100.0
        self.df["soc_n"]  = self.df["battery_soc"] / 100.0

        # already 0–1
        self.df["flex_n"] = self.df["workload_flexibility"]

        self.state_cols = ["carbon_n", "util_n", "cool_n", "soc_n", "flex_n"]

        # --- simulator parameters (simple, safe for prototype) ---
        self.base_it_kw = 50.0
        self.it_scale_kw = 150.0
        self.max_batt_kw = 150.0      # max battery discharge per step (kW)
        self.soc_drop_per_kwh = 0.002 # SOC drop factor (tune later)

        self.t = 0
        self.done = False
        self.backlog = 0.0  # ✅ add this line

        self.reset()

    def reset(self):
        self.t = 0
        self.done = False
        self.backlog = 0.0  # ✅ add this line
        return self._get_state()

    def _get_state(self):
        return self.df.loc[self.t, self.state_cols].to_numpy(dtype=np.float32)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")

        row = self.df.loc[self.t]
        carbon_intensity = float(row["grid_carbon_intensity"])  # gCO2/kWh
        util_n = float(row["util_n"])
        cool_kw = float(row["cooling_load"])
        cool_kw_effective = cool_kw
        soc_n = float(row["soc_n"])
        flex_n = float(row["flex_n"])

        # --- compute IT power with backlog shifting ---
        # Pull some delayed work back in (execute later)
        pull = min(self.backlog, 0.15)  # run up to 10% extra per step
        self.backlog -= pull

        workload_factor = 1.0 + pull
        it_kw = (self.base_it_kw + util_n * self.it_scale_kw) * workload_factor

        # --- apply action effects ---
        sla_violation = 0

        # 1) Delay flexible workload: reduces IT power a bit if flexible
        if action == DELAY_FLEX:
            if flex_n < 0.6:
                sla_violation = 1
            if util_n > 0.75:
                sla_violation = 1

            # ✅ shift part of work to later instead of deleting it
            shift = 0.25 * flex_n  # 0–15% shift based on flexibility
            self.backlog += shift  # store delayed work
            it_kw *= (1.0 - shift)  # run less now

            # keep your strong demo effect
            it_kw *= 0.80
            cool_kw_effective *= 0.90

        # 2) Use battery: offset grid power, reduce SOC
        batt_kw = 0.0
        if action == USE_BATTERY and soc_n > 0.2:
            batt_kw = self.max_batt_kw * min(1.0, soc_n)  # simple scaling
            # drop SOC a little (prototype)
            dt_hours = 5.0 / 60.0
            energy_kwh = batt_kw * dt_hours
            soc_n = max(0.0, soc_n - self.soc_drop_per_kwh * energy_kwh)

        # --- grid power ---
        grid_kw = max(0.0, it_kw + cool_kw_effective - batt_kw)

        # --- carbon this step (relative units ok for prototype) ---
        carbon_step = (grid_kw * carbon_intensity) / 1000.0

        # --- reward: minimize carbon + punish SLA ---
        reward = -carbon_step
        if sla_violation:
            reward -= 50.0  # big penalty so SLA matters

        # update SOC back into df so next state reflects changes (optional but good)
        self.df.at[self.t, "soc_n"] = soc_n

        # move time
        self.t += 1
        if self.t >= len(self.df):
            self.done = True

        next_state = None if self.done else self._get_state()

        info = {
            "carbon_step": carbon_step,
            "grid_kw": grid_kw,
            "it_kw": it_kw,
            "cool_kw": cool_kw_effective,
            "batt_kw": batt_kw,
            "sla_violation": sla_violation,
            "t": self.t,
        }

        return next_state, float(reward), self.done, info
