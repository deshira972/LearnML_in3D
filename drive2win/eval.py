from __future__ import annotations
import time
from typing import Callable
import numpy as np


def run_policy(client, policy_fn: Callable, duration: float = 60.0,
               hz: float = 20.0, on_step: Callable | None = None) -> dict:
    interval = 1.0 / hz
    time.sleep(5.0)
    start = time.time()
    steps = 0
    checkpoints_passed = 0
    crashes = 0
    last_pos = None
    stuck_streak = 0
    max_stuck = 0
    track = []
    next_log = start

    while time.time() - start < duration:
        state = client.get_latest_state()
        if not state or "sensors" not in state:
            time.sleep(0.1); continue

        nav = state["sensors"].get("navigation") or {}
        cp = nav.get("checkpoints_completed", 0) or 0
        checkpoints_passed = max(checkpoints_passed, cp)

        sp = state["sensors"].get("speed", 0.0)
        if sp < 0.3:
            stuck_streak += 1
        else:
            max_stuck = max(max_stuck, stuck_streak)
            stuck_streak = 0

        pos = state.get("position") or {}
        if last_pos is not None and pos:
            dx = pos.get("x", 0) - last_pos.get("x", 0)
            dz = pos.get("z", 0) - last_pos.get("z", 0)
            if (dx * dx + dz * dz) > 25.0:
                crashes += 1
        last_pos = pos

        throttle, steering = policy_fn(state)
        client.send_control_ws(throttle, steering)
        steps += 1
        if on_step is not None:
            on_step(steps, state, (throttle, steering))

        now = time.time()
        if now >= next_log:
            track.append({"t": now - start, "position": pos, "speed": sp})
            next_log = now + 1.0

        time.sleep(interval)

    elapsed = time.time() - start
    return {
        "steps": steps,
        "elapsed": elapsed,
        "checkpoints_passed": checkpoints_passed,
        "crashes": crashes,
        "min_speed_streak": max(max_stuck, stuck_streak),
        "track": track,
    }


def score_runs(runs: list[dict], target_checkpoints: int) -> dict:
    completed = [r for r in runs if r["checkpoints_passed"] >= target_checkpoints]
    times = [r["elapsed"] for r in completed]
    crashes_per_run = [r["crashes"] for r in runs]

    return {
        "n_runs": len(runs),
        "completion_rate": len(completed) / max(1, len(runs)),
        "median_lap_time": float(np.median(times)) if times else float("inf"),
        "mean_crashes": float(np.mean(crashes_per_run)) if crashes_per_run else 0.0,
        "max_checkpoints": max(r["checkpoints_passed"] for r in runs),
    }