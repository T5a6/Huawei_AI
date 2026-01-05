import re
from typing import Dict, Any, List

RE_BASELINE = re.compile(r"\[BASELINE\]\s+carbon=(?P<carbon>[\d.]+)\s+sla=(?P<sla>\d+)")
RE_LOADED_GLOBAL = re.compile(r"\[Loaded GLOBAL best\]\s+carbon=(?P<carbon>[\d.]+)")
RE_ACTIONS = re.compile(r"\[ACTIONS\]\s+RUN=(?P<run>\d+)\s+DELAY=(?P<delay>\d+)\s+BATT=(?P<batt>\d+)")
RE_EVAL = re.compile(r"\[EVAL\]\s+ep=(?P<ep>\d+)\s+carbon=(?P<carbon>[\d.]+)\s+sla=(?P<sla>\d+)")
RE_BEST_RUN = re.compile(r"\[BEST RUN\]\s+saved\s+best_run\.ckpt\s+carbon=(?P<carbon>[\d.]+)")
RE_PROMOTED_NEW = re.compile(r"\[PROMOTED\]\s+New GLOBAL BEST!\s+carbon=(?P<carbon>[\d.]+)")
RE_PROMOTED_NOT = re.compile(r"\[PROMOTED\]\s+Not better than global\.\s+global=(?P<global>[\d.]+)\s+run_best=(?P<run_best>[\d.]+)")
RE_PROMOTED_NONE = re.compile(r"\[PROMOTED\]\s+No valid best_run to promote")
RE_BASELINE_FINAL = re.compile(r"Baseline carbon:\s+(?P<carbon>[\d.]+)\s+\|\s+SLA:\s+(?P<sla>\d+)")
RE_SERVED_FINAL = re.compile(r"Served carbon:\s+(?P<carbon>[\d.]+)\s+\|\s+SLA:\s+(?P<sla>\d+)")
RE_REDUCTION = re.compile(r"Reduction:\s+(?P<pct>[\d.]+)%")

def parse_output(lines: List[str]) -> Dict[str, Any]:
    run: Dict[str, Any] = {
        "baseline_carbon": None,
        "baseline_sla": None,
        "loaded_global_best": None,
        "eval_points": [],
        "best_run_carbon": None,
        "promotion": None,
        "served_carbon": None,
        "served_sla": None,
        "reduction_pct": None,
        "notes": [],
    }

    pending_actions = None

    for raw in lines:
        line = raw.strip()

        m = RE_BASELINE.search(line)
        if m:
            run["baseline_carbon"] = float(m.group("carbon"))
            run["baseline_sla"] = int(m.group("sla"))
            continue

        m = RE_LOADED_GLOBAL.search(line)
        if m:
            run["loaded_global_best"] = float(m.group("carbon"))
            continue

        m = RE_ACTIONS.search(line)
        if m:
            pending_actions = {
                "run": int(m.group("run")),
                "delay": int(m.group("delay")),
                "batt": int(m.group("batt")),
            }
            continue

        m = RE_EVAL.search(line)
        if m:
            pt = {
                "ep": int(m.group("ep")),
                "carbon": float(m.group("carbon")),
                "sla": int(m.group("sla")),
            }
            if pending_actions:
                pt["actions"] = pending_actions
                pending_actions = None
            run["eval_points"].append(pt)
            continue

        m = RE_BEST_RUN.search(line)
        if m:
            run["best_run_carbon"] = float(m.group("carbon"))
            continue

        m = RE_PROMOTED_NEW.search(line)
        if m:
            run["promotion"] = {"type": "new_global", "global_best": float(m.group("carbon"))}
            continue

        m = RE_PROMOTED_NOT.search(line)
        if m:
            run["promotion"] = {
                "type": "kept_global",
                "global_best": float(m.group("global")),
                "run_best": float(m.group("run_best")),
            }
            continue

        if RE_PROMOTED_NONE.search(line):
            run["promotion"] = {"type": "no_valid_best_run"}
            continue

        m = RE_BASELINE_FINAL.search(line)
        if m:
            if run["baseline_carbon"] is None:
                run["baseline_carbon"] = float(m.group("carbon"))
            if run["baseline_sla"] is None:
                run["baseline_sla"] = int(m.group("sla"))
            continue

        m = RE_SERVED_FINAL.search(line)
        if m:
            run["served_carbon"] = float(m.group("carbon"))
            run["served_sla"] = int(m.group("sla"))
            continue

        m = RE_REDUCTION.search(line)
        if m:
            run["reduction_pct"] = float(m.group("pct"))
            continue

    if run["served_sla"] is not None and run["served_sla"] != 0:
        run["notes"].append("Warning: Served SLA is not zero.")

    if run["reduction_pct"] is None and run["baseline_carbon"] and run["served_carbon"]:
        run["reduction_pct"] = (run["baseline_carbon"] - run["served_carbon"]) / run["baseline_carbon"] * 100.0

    return run
