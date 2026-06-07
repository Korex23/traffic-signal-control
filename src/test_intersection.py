import os
import sys
import numpy as np
import traci
from stable_baselines3 import PPO

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME")

MODEL_PATH   = "checkpoints/ppo_traffic_final.zip"
SUMO_CONFIG  = "networks/intersection_test/intersection_test.sumocfg"
EPISODE_LENGTH = 3600
PORT = 8820

LANES = [
    "N2C_0", "N2C_1", "N2C_2",
    "S2C_0", "S2C_1",
    "E2C_0", "E2C_1",
    "W2C_0", "W2C_1",
    "NW2C_0",
]

def build_state():
    """Build a 73-dim state vector — pad unused slots with zeros."""
    state = np.zeros(73, dtype=np.float32)

    queue_lengths = []
    waiting_times = []

    for lane in LANES:
        try:
            q = traci.lane.getLastStepHaltingNumber(lane)
            w = traci.lane.getWaitingTime(lane)
        except:
            q, w = 0, 0
        queue_lengths.append(q / 50.0)
        waiting_times.append(min(w, 300.0) / 300.0)

    # Fill intersection 1 slot (first 21 values)
    # 8 queue + 8 wait + 4 phase + 1 EV
    for j in range(min(8, len(queue_lengths))):
        state[j] = queue_lengths[j]
    for j in range(min(8, len(waiting_times))):
        state[8 + j] = waiting_times[j]

    # Active phase one-hot (4 phases)
    try:
        phase = traci.trafficlight.getPhase("C")
        if phase < 4:
            state[16 + phase] = 1.0
    except:
        pass

    # EV flag
    try:
        for lane in LANES:
            for v in traci.lane.getLastStepVehicleIDs(lane):
                if traci.vehicle.getTypeID(v) == "emergency":
                    state[20] = 1.0
                    break
    except:
        pass

    return state


def run_test(use_gui=True):
    print("Loading PPO model...")
    model = PPO.load(MODEL_PATH)
    print("Model loaded OK\n")

    sumo_binary = "sumo-gui" if use_gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", SUMO_CONFIG,
        "--no-warnings",
        "--no-step-log",
        "--random",
        "--time-to-teleport", "-1",
        "--end", "999999",
    ]

    traci.start(sumo_cmd, port=PORT, label="test")

    waiting_times = []
    queue_lengths = []
    throughputs   = []
    ev_times      = []

    for step in range(EPISODE_LENGTH):
        state = build_state()

        # Agent picks action for all 3 intersection slots
        # We only use the first one (intersection 1 = 4 phases)
        action, _ = model.predict(state, deterministic=True)
        phase = int(action[0])  # use intersection 1's decision

        # Enforce minimum green (10s)
        try:
            current = traci.trafficlight.getPhase("C")
            if step % 10 == 0 and phase != current:
                traci.trafficlight.setPhase("C", phase % 4)
        except:
            pass

        traci.simulationStep()

        # Collect metrics
        total_wait  = 0.0
        total_queue = 0.0
        vehicles    = 0
        for lane in LANES:
            try:
                h = traci.lane.getLastStepHaltingNumber(lane)
                w = traci.lane.getWaitingTime(lane)
                if h > 0:
                    total_wait += w / h
                    vehicles   += 1
                total_queue += h
            except:
                pass

        avg_wait = total_wait / vehicles if vehicles > 0 else 0
        waiting_times.append(avg_wait)
        queue_lengths.append(total_queue)
        throughputs.append(traci.simulation.getArrivedNumber())

        # EV detection
        for lane in LANES:
            try:
                for v in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getTypeID(v) == "emergency":
                        ev_times.append(step)
            except:
                pass

        if step % 500 == 0:
            print(f"Step {step:>4} | Avg wait: {avg_wait:.2f}s | Queue: {total_queue:.1f}")

    traci.close()

    print("\n" + "="*50)
    print("  PPO on Unseen Intersection — Results")
    print("="*50)
    print(f"Avg waiting time:  {np.mean(waiting_times):.2f}s")
    print(f"Avg queue length:  {np.mean(queue_lengths):.2f} vehicles")
    print(f"Total throughput:  {sum(throughputs)} vehicles")
    if ev_times:
        print(f"EV detections:     {len(ev_times)} times")
        print(f"Avg EV clearance:  {np.mean(ev_times):.0f}s")
    else:
        print("EV detections:     0")
    print("="*50)


if __name__ == "__main__":
    run_test(use_gui=True)