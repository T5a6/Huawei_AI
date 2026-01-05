import json
import os
from datetime import datetime
from typing import Dict, Any, List

HISTORY_PATH = "run_history.json"

def load_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_history(hist: List[Dict[str, Any]]) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def append_run(run: Dict[str, Any]) -> Dict[str, Any]:
    hist = load_history()
    run = dict(run)
    run["timestamp"] = datetime.now().isoformat(timespec="seconds")
    run["run_index"] = len(hist) + 1
    hist.append(run)
    save_history(hist)
    return run
