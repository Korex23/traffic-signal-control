import os
import json
import numpy as np
import traci
from stable_baselines3 import PPO
from src.traffic_env import TrafficEnv
from src.webster_baseline import WebsterBaseline
from src.max_pressure_baseline import MaxPressureBaseline
from src.sotl_baseline import SOTLBaseline

# Baseline controllers to compare PPO against. Each exposes get_action(step)/reset().
BASELINES = {
    "webster":      ("Webster",  WebsterBaseline),
    "max_pressure": ("MaxPress", MaxPressureBaseline),
    "sotl":         ("SOTL",     SOTLBaseline),
}

# ── Configuration ──────────────────────────────────────────────────────────
# Prefer the best model (selected by eval reward during training); fall back to
# the final model if best_model.zip is not present.
BEST_MODEL_PATH  = "checkpoints/best_model.zip"
FINAL_MODEL_PATH = "checkpoints/ppo_traffic_final.zip"
RESULTS_DIR      = "results"
EPISODE_LENGTH   = 3600
NUM_TRIALS       = 10

# Demand scenarios mapped to SUMO --scale factors.
# The fixed eval networks (intersection1/2/3) bake in ~200 veh/hr, so:
#   low = 1x (~200), medium = 3x (~600), peak = 5x (~1000)
DEMAND_SCENARIOS = {
    "low":    {"scale": 1.0, "veh_per_hr": 200},
    "medium": {"scale": 3.0, "veh_per_hr": 600},
    "peak":   {"scale": 5.0, "veh_per_hr": 1000},
}

os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed evaluation intersections. These MUST match TrafficEnv's EVAL_INTERSECTIONS
# (used when training_mode=False): intersection1/2/3, single traffic light "C".
INTERSECTIONS = [
    {
        "id": "intersection1",
        "tl_id": "C",
        "lanes_in": ["N2C_0", "N2C_1", "S2C_0", "S2C_1",
                     "E2C_0", "E2C_1", "W2C_0", "W2C_1"],
        "num_phases": 4,
    },
    {
        "id": "intersection2",
        "tl_id": "C",
        "lanes_in": ["N2C_0", "N2C_1", "N2C_2",
                     "S2C_0", "S2C_1", "S2C_2",
                     "E2C_0", "E2C_1",
                     "W2C_0", "W2C_1",
                     "NE2C_0", "NE2C_1",
                     "SW2C_0", "SW2C_1"],
        "num_phases": 6,
    },
    {
        "id": "intersection3",
        "tl_id": "C",
        "lanes_in": ["N2C_0", "N2C_1", "S2C_0", "S2C_1",
                     "E2C_0", "E2C_1", "W2C_0", "W2C_1"],
        "num_phases": 4,
    },
]


# ── Metric collection ───────────────────────────────────────────────────────

def collect_metrics():
    """Snapshot the continuous metrics (waiting, queue, throughput) at this step."""
    total_waiting    = 0.0
    total_vehicles   = 0
    total_queue      = 0.0
    total_throughput = 0

    for i, intersection in enumerate(INTERSECTIONS):
        traci.switch(f"int{i+1}")
        lanes = intersection["lanes_in"]

        for lane in lanes:
            try:
                halting   = traci.lane.getLastStepHaltingNumber(lane)
                lane_wait = traci.lane.getWaitingTime(lane)

                # Average waiting time per vehicle in this lane
                if halting > 0:
                    total_waiting  += lane_wait / halting
                    total_vehicles += 1

                total_queue += halting
            except Exception:
                pass

        # Count vehicles that completed trips
        try:
            total_throughput += traci.simulation.getArrivedNumber()
        except Exception:
            pass

    avg_waiting = total_waiting / total_vehicles if total_vehicles > 0 else 0.0

    return avg_waiting, total_queue, total_throughput


def track_ev_waiting(ev_wait):
    """Update per-emergency-vehicle accumulated waiting time.

    For every emergency vehicle currently on a monitored approach lane, record
    its accumulated waiting time (total seconds spent stopped so far this trip;
    requires SUMO --waiting-time-memory large enough, set in TrafficEnv). We keep
    the maximum value seen per vehicle, so once the EV crosses and disappears we
    retain its final total waiting-to-clear. Keyed by (intersection_index, veh_id)
    because the 3 SUMO instances have independent vehicle IDs.

    Lower mean value = the controller cleared emergency vehicles faster.
    """
    for i, intersection in enumerate(INTERSECTIONS):
        traci.switch(f"int{i+1}")
        for lane in intersection["lanes_in"]:
            try:
                for v in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getTypeID(v) == "emergency":
                        w   = traci.vehicle.getAccumulatedWaitingTime(v)
                        key = (i, v)
                        if w >= ev_wait.get(key, 0.0):
                            ev_wait[key] = w
            except Exception:
                pass


def _make_eval_env(scale):
    """Fixed eval env: intersection1/2/3, controlled demand via --scale."""
    return TrafficEnv(
        use_gui           = False,
        episode_length    = EPISODE_LENGTH,
        scenario          = "default",
        training_mode     = False,
        eval_demand_scale = scale,
    )


# ── PPO evaluation ──────────────────────────────────────────────────────────

def run_ppo_trial(model, scale, trial_num):
    print(f"  PPO Trial {trial_num+1}/{NUM_TRIALS} | scale={scale}")

    env = _make_eval_env(scale)
    obs, _ = env.reset()

    waiting_times, queue_lengths, throughputs = [], [], []
    ev_wait = {}

    for step in range(EPISODE_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        w, q, t = collect_metrics()
        waiting_times.append(w)
        queue_lengths.append(q)
        throughputs.append(t)
        track_ev_waiting(ev_wait)

        if done:
            break

    env.close()

    ev_vals = list(ev_wait.values())
    return {
        "avg_waiting_time":   float(np.mean(waiting_times)),
        "avg_queue_length":   float(np.mean(queue_lengths)),
        "total_throughput":   int(sum(throughputs)),
        "ev_avg_wait_time":   float(np.mean(ev_vals)) if ev_vals else None,
        "ev_count":           len(ev_vals),
    }


# ── Baseline evaluation (Webster / Max-Pressure / SOTL) ─────────────────────

def run_baseline_trial(controller_cls, scale, name, trial_num):
    print(f"  {name} Trial {trial_num+1}/{NUM_TRIALS} | scale={scale}")

    env = _make_eval_env(scale)
    controller = controller_cls()
    obs, _ = env.reset()
    controller.reset()

    waiting_times, queue_lengths, throughputs = [], [], []
    ev_wait = {}

    for step in range(EPISODE_LENGTH):
        action = controller.get_action(step)
        obs, reward, done, truncated, info = env.step(action)

        w, q, t = collect_metrics()
        waiting_times.append(w)
        queue_lengths.append(q)
        throughputs.append(t)
        track_ev_waiting(ev_wait)

        if done:
            break

    env.close()

    ev_vals = list(ev_wait.values())
    return {
        "avg_waiting_time":   float(np.mean(waiting_times)),
        "avg_queue_length":   float(np.mean(queue_lengths)),
        "total_throughput":   int(sum(throughputs)),
        "ev_avg_wait_time":   float(np.mean(ev_vals)) if ev_vals else None,
        "ev_count":           len(ev_vals),
    }


# ── Main evaluation loop ────────────────────────────────────────────────────

def _load_model():
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading BEST model: {BEST_MODEL_PATH}")
        return PPO.load(BEST_MODEL_PATH)
    if os.path.exists(FINAL_MODEL_PATH):
        print(f"best_model.zip not found — loading FINAL model: {FINAL_MODEL_PATH}")
        return PPO.load(FINAL_MODEL_PATH)
    raise FileNotFoundError(
        f"No model found at {BEST_MODEL_PATH} or {FINAL_MODEL_PATH}. Train first."
    )


def evaluate():
    print("=" * 60)
    print("  Traffic Signal Control — Evaluation")
    print(f"  Trials per scenario: {NUM_TRIALS}")
    print(f"  Episode length:      {EPISODE_LENGTH} steps")
    print(f"  Eval intersections:  intersection1/2/3 (fixed)")
    print("=" * 60)

    print()
    model = _load_model()
    print("Model loaded OK\n")

    all_results = {}

    for scenario, cfg in DEMAND_SCENARIOS.items():
        scale = cfg["scale"]
        print(f"\n{'='*40}")
        print(f"Scenario: {scenario.upper()} (~{cfg['veh_per_hr']} veh/hr, scale={scale})")
        print(f"{'='*40}")

        print("\nRunning PPO agent...")
        ppo_results = [run_ppo_trial(model, scale, t) for t in range(NUM_TRIALS)]

        baseline_results = {}
        for key, (label, cls) in BASELINES.items():
            print(f"\nRunning {label} baseline...")
            baseline_results[key] = [
                run_baseline_trial(cls, scale, label, t) for t in range(NUM_TRIALS)
            ]

        def summarise(results):
            metrics = ["avg_waiting_time", "avg_queue_length",
                       "total_throughput", "ev_avg_wait_time"]
            summary = {}
            for m in metrics:
                vals = [r[m] for r in results if r[m] is not None]
                if vals:
                    summary[m] = {"mean": float(np.mean(vals)),
                                  "std":  float(np.std(vals))}
            return summary

        summaries = {"ppo": summarise(ppo_results)}
        for key, res in baseline_results.items():
            summaries[key] = summarise(res)

        metrics    = ["avg_waiting_time", "avg_queue_length",
                      "total_throughput", "ev_avg_wait_time"]
        columns    = ["ppo"] + list(BASELINES.keys())
        col_labels = {"ppo": "PPO", **{k: v[0] for k, v in BASELINES.items()}}

        # Means table
        header = f"{'Metric':<20}" + "".join(f"{col_labels[c]:>12}" for c in columns)
        print("\n" + header)
        print("-" * len(header))
        for m in metrics:
            row = f"{m:<20}"
            for c in columns:
                row += f"{summaries[c].get(m, {}).get('mean', 0):>12.2f}"
            print(row)

        # PPO improvement vs each baseline (positive = PPO better)
        print("\nPPO improvement vs each baseline (positive = PPO better):")
        for m in metrics:
            ppo_val = summaries["ppo"].get(m, {}).get("mean", 0)
            line = f"  {m:<20}"
            for key in BASELINES:
                bval = summaries[key].get(m, {}).get("mean", 0)
                if m == "total_throughput":       # higher is better
                    imp = ((ppo_val - bval) / bval * 100) if bval else 0
                else:                              # lower is better
                    imp = ((bval - ppo_val) / bval * 100) if bval else 0
                line += f"  {col_labels[key]}: {imp:+6.1f}%"
            print(line)

        all_results[scenario] = summaries

    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()
