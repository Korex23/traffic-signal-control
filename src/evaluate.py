import os
import sys
import json
import numpy as np
import traci
from stable_baselines3 import PPO
from src.traffic_env import TrafficEnv
from src.webster_baseline import WebsterBaseline

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH     = "checkpoints/ppo_traffic_final.zip"
RESULTS_DIR    = "results"
EPISODE_LENGTH = 3600
NUM_TRIALS     = 10

# Demand scenarios — veh/hr per approach
DEMAND_SCENARIOS = {
    "low":    200,
    "medium": 600,
    "peak":   1000,
}

os.makedirs(RESULTS_DIR, exist_ok=True)

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

def collect_metrics(step):
    """Collect metrics from all 3 intersections at current step."""
    total_waiting    = 0.0
    total_vehicles   = 0
    total_queue      = 0.0
    total_throughput = 0
    ev_clearance     = None

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

        # EV clearance time
        try:
            for lane in lanes:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                for v in vehicles:
                    if traci.vehicle.getTypeID(v) == "emergency":
                        ev_clearance = step
        except Exception:
            pass

    # Average waiting time across all lanes that had vehicles
    avg_waiting = total_waiting / total_vehicles if total_vehicles > 0 else 0.0

    return avg_waiting, total_queue, total_throughput, ev_clearance


# ── PPO evaluation ──────────────────────────────────────────────────────────

def run_ppo_trial(model, demand, trial_num):
    """Run one trial with the trained PPO agent."""
    print(f"  PPO Trial {trial_num+1}/{NUM_TRIALS} | demand={demand}")

    env = TrafficEnv(use_gui=False, episode_length=EPISODE_LENGTH)
    obs, _ = env.reset()

    waiting_times = []
    queue_lengths = []
    throughputs   = []
    ev_times      = []

    for step in range(EPISODE_LENGTH):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        w, q, t, ev = collect_metrics(step)
        waiting_times.append(w)
        queue_lengths.append(q)
        throughputs.append(t)
        if ev is not None:
            ev_times.append(ev)

        if done:
            break

    env.close()

    return {
        "avg_waiting_time":  np.mean(waiting_times),
        "avg_queue_length":  np.mean(queue_lengths),
        "total_throughput":  sum(throughputs),
        "ev_clearance_time": np.mean(ev_times) if ev_times else None,
    }


# ── Webster evaluation ──────────────────────────────────────────────────────

def run_webster_trial(demand, trial_num):
    """Run one trial with Webster fixed-time baseline."""
    print(f"  Webster Trial {trial_num+1}/{NUM_TRIALS} | demand={demand}")

    env = TrafficEnv(use_gui=False, episode_length=EPISODE_LENGTH)
    webster = WebsterBaseline()
    obs, _ = env.reset()
    webster.reset()

    waiting_times = []
    queue_lengths = []
    throughputs   = []
    ev_times      = []

    for step in range(EPISODE_LENGTH):
        action = webster.get_action(step)
        obs, reward, done, truncated, info = env.step(action)

        w, q, t, ev = collect_metrics(step)
        waiting_times.append(w)
        queue_lengths.append(q)
        throughputs.append(t)
        if ev is not None:
            ev_times.append(ev)

        if done:
            break

    env.close()

    return {
        "avg_waiting_time":  np.mean(waiting_times),
        "avg_queue_length":  np.mean(queue_lengths),
        "total_throughput":  sum(throughputs),
        "ev_clearance_time": np.mean(ev_times) if ev_times else None,
    }


# ── Main evaluation loop ────────────────────────────────────────────────────

def evaluate():
    print("=" * 60)
    print("  Traffic Signal Control — Evaluation")
    print(f"  Trials per scenario: {NUM_TRIALS}")
    print(f"  Episode length:      {EPISODE_LENGTH} steps")
    print("=" * 60)

    # Load trained PPO model
    print("\nLoading PPO model...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded OK\n")

    all_results = {}

    for scenario, demand in DEMAND_SCENARIOS.items():
        print(f"\n{'='*40}")
        print(f"Scenario: {scenario.upper()} ({demand} veh/hr)")
        print(f"{'='*40}")

        ppo_results     = []
        webster_results = []

        # Run PPO trials
        print("\nRunning PPO agent...")
        for t in range(NUM_TRIALS):
            result = run_ppo_trial(model, demand, t)
            ppo_results.append(result)

        # Run Webster trials
        print("\nRunning Webster baseline...")
        for t in range(NUM_TRIALS):
            result = run_webster_trial(demand, t)
            webster_results.append(result)

        # Summarise results
        def summarise(results):
            metrics = ["avg_waiting_time", "avg_queue_length",
                       "total_throughput", "ev_clearance_time"]
            summary = {}
            for m in metrics:
                vals = [r[m] for r in results if r[m] is not None]
                if vals:
                    summary[m] = {
                        "mean": float(np.mean(vals)),
                        "std":  float(np.std(vals)),
                    }
            return summary

        ppo_summary     = summarise(ppo_results)
        webster_summary = summarise(webster_results)

        # Print results table
        print(f"\n{'Metric':<25} {'PPO Mean':>12} {'PPO Std':>10} {'Webster Mean':>14} {'Webster Std':>12} {'Improvement':>12}")
        print("-" * 90)

        for metric in ["avg_waiting_time", "avg_queue_length", "total_throughput"]:
            ppo_val = ppo_summary.get(metric, {}).get("mean", 0)
            ppo_std = ppo_summary.get(metric, {}).get("std", 0)
            web_val = webster_summary.get(metric, {}).get("mean", 0)
            web_std = webster_summary.get(metric, {}).get("std", 0)

            # Throughput: higher PPO is better
            # Waiting time & queue: lower PPO is better
            if metric == "total_throughput":
                improvement = ((ppo_val - web_val) / web_val) * 100 if web_val > 0 else 0
            else:
                improvement = ((web_val - ppo_val) / web_val) * 100 if web_val > 0 else 0

            print(f"{metric:<25} {ppo_val:>12.2f} {ppo_std:>10.2f} {web_val:>14.2f} {web_std:>12.2f} {improvement:>11.1f}%")

        all_results[scenario] = {
            "ppo":     ppo_summary,
            "webster": webster_summary,
        }

    # Save results to JSON
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}") 
    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()