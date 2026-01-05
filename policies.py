from datacenter import RUN_NOW, DELAY_FLEX, USE_BATTERY

def baseline_policy(state):
    """
    BEFORE policy: does nothing smart.
    Always run workload immediately.
    """
    return RUN_NOW


def carb_policy(state,
                carbon_high=0.50,
                carbon_very_high=0.70,
                flex_high=0.65,
                soc_ok=0.35,
                util_safe=0.70):
    """
    AFTER policy: carbon-aware controller.
    Uses carbon + flexibility + battery to reduce emissions while avoiding SLA risk.

    Rules:
    1) If carbon is high AND workload is flexible AND utilization is not too high:
       -> delay flexible workload
    2) Else if carbon is high AND battery has enough charge:
       -> use battery to offset grid power
    3) Else:
       -> run now
    """
    carbon_n, util_n, cool_n, soc_n, flex_n = state

    # 1) If carbon is VERY high -> prefer battery (fast carbon drop)
    if carbon_n > carbon_very_high and soc_n > soc_ok:
        return USE_BATTERY

    # 2) If carbon is high and work is flexible -> delay
    if carbon_n > carbon_high and flex_n > flex_high and util_n < util_safe:
        return DELAY_FLEX

    # 3) Otherwise run
    return RUN_NOW
