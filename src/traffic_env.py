import os
import sys
import gymnasium as gym
import numpy as np
import traci

# SUMO setup
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")

# Paths to SUMO config files
SUMO_CONFIGS = [
    os.path.join("networks", "intersection1", "intersection1.sumocfg"),
    os.path.join("networks", "intersection2", "intersection2.sumocfg"),
    os.path.join("networks", "intersection3", "intersection3.sumocfg"),
]

# Intersection definitions
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

# State vector size
# Int 1 & 3: 8 queue + 8 wait + 4 phase + 1 EV = 21
# Int 2:    12 queue + 12 wait + 6 phase + 1 EV = 31
# Total: 21 + 31 + 21 = 73
STATE_SIZE = 73

# Reward weights from thesis
ALPHA = 5.0  # EV bonus weight
BETA = 0.5   # Queue growth penalty weight


class TrafficEnv(gym.Env):
    """
    Custom Gymnasium environment for adaptive traffic signal control.
    Controls 3 intersections simultaneously via TraCI.
    State:  73-dimensional vector (queue lengths, waiting times, phases, EV flags)
    Action: discrete phase selection per intersection
    Reward: R = -ΣW + α·EV - β·ΔQ
    """

    def __init__(self, use_gui=False, episode_length=3600):
        super().__init__()

        self.use_gui = use_gui
        self.episode_length = episode_length
        self.step_count = 0
        self.prev_queue_length = 0.0

        # Action space: [4, 6, 4] — one phase choice per intersection
        self.action_space = gym.spaces.MultiDiscrete([4, 6, 4])

        # Observation space: 73-dimensional normalized vector
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STATE_SIZE,),
            dtype=np.float32
        )

        # TraCI ports — one per SUMO instance
        self.ports = [8813, 8814, 8815]

        # Minimum green time enforcement (10 seconds from thesis)
        self.min_green = 10
        self.green_timer = [0, 0, 0]
        self.current_phase = [0, 0, 0]

        print("TrafficEnv initialized")
        print(f"Action space: {self.action_space}")
        print(f"Observation space: {self.observation_space}")

    def _start_sumo(self):
        """Start 3 SUMO instances on separate ports"""
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        for i, (config, port) in enumerate(zip(SUMO_CONFIGS, self.ports)):
            sumo_cmd = [
                sumo_binary,
                "-c", config,
                "--no-warnings",
                "--no-step-log",
                "--random",
            ]
            traci.start(sumo_cmd, port=port, label=f"int{i+1}")
            print(f"Started SUMO instance {i+1} on port {port}")

    def _close_sumo(self):
        """Close all SUMO instances"""
        for i in range(3):
            try:
                traci.switch(f"int{i+1}")
                traci.close()
            except Exception:
                pass

    def _get_state(self):
        """Build 73-dimensional state vector from all 3 intersections"""
        state = []

        for i, intersection in enumerate(INTERSECTIONS):
            traci.switch(f"int{i+1}")
            lanes = intersection["lanes_in"]

            # Queue lengths per lane (normalized by max 50 vehicles)
            queue_lengths = []
            for lane in lanes:
                try:
                    q = traci.lane.getLastStepHaltingNumber(lane)
                except Exception:
                    q = 0
                queue_lengths.append(q / 50.0)

            # Waiting times per lane (normalized by max 300 seconds)
            waiting_times = []
            for lane in lanes:
                try:
                    w = traci.lane.getWaitingTime(lane)
                except Exception:
                    w = 0
                waiting_times.append(w / 300.0)

            # Pad to fixed size
            max_lanes = 12
            queue_lengths = queue_lengths[:max_lanes]
            waiting_times = waiting_times[:max_lanes]
            while len(queue_lengths) < max_lanes:
                queue_lengths.append(0.0)
            while len(waiting_times) < max_lanes:
                waiting_times.append(0.0)

            # Active phase one-hot encoded
            max_phases = 6
            phase_one_hot = [0.0] * max_phases
            try:
                current_phase = traci.trafficlight.getPhase(intersection["tl_id"])
                if current_phase < max_phases:
                    phase_one_hot[current_phase] = 1.0
            except Exception:
                pass

            # Emergency vehicle flag
            ev_flag = self._detect_emergency_vehicle(i, lanes)

            # Build per-intersection state with thesis dimensions
            if i == 1:  # Intersection 2: 12+12+6+1 = 31
                state.extend(queue_lengths)       # 12
                state.extend(waiting_times)       # 12
                state.extend(phase_one_hot)       # 6
                state.append(float(ev_flag))      # 1
            else:       # Intersections 1 & 3: 8+8+4+1 = 21
                state.extend(queue_lengths[:8])   # 8
                state.extend(waiting_times[:8])   # 8
                state.extend(phase_one_hot[:4])   # 4
                state.append(float(ev_flag))      # 1

        return np.array(state, dtype=np.float32)

    def _detect_emergency_vehicle(self, intersection_idx, lanes):
        """Detect if emergency vehicle is present on any approach lane"""
        try:
            for lane in lanes:
                vehicles = traci.lane.getLastStepVehicleIDs(lane)
                for v in vehicles:
                    if traci.vehicle.getTypeID(v) == "emergency":
                        return 1
        except Exception:
            pass
        return 0

    def _compute_reward(self, actions, ev_flags):
        """Compute reward: R = -ΣW + α·EV - β·ΔQ"""
        total_waiting = 0.0
        total_queue = 0.0
        ev_bonus = 0.0

        for i, intersection in enumerate(INTERSECTIONS):
            traci.switch(f"int{i+1}")
            lanes = intersection["lanes_in"]

            for lane in lanes:
                try:
                    total_waiting += traci.lane.getWaitingTime(lane)
                    total_queue += traci.lane.getLastStepHaltingNumber(lane)
                except Exception:
                    pass

            # EV bonus if chosen phase serves an EV approach
            if ev_flags[i]:
                ev_bonus += ALPHA

        # Queue growth penalty
        delta_queue = total_queue - self.prev_queue_length
        self.prev_queue_length = total_queue

        reward = -total_waiting + ev_bonus - BETA * delta_queue
        return reward, ev_flags

    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)

        # Close existing SUMO instances
        self._close_sumo()

        # Reset trackers
        self.step_count = 0
        self.prev_queue_length = 0.0
        self.green_timer = [0, 0, 0]
        self.current_phase = [0, 0, 0]

        # Start fresh SUMO instances
        self._start_sumo()

        # Get initial state
        state = self._get_state()
        info = {}

        return state, info

    def step(self, actions):
        """
        Execute one step in the environment.
        actions: array of [phase_int1, phase_int2, phase_int3]
        """
        ev_flags = [0, 0, 0]

        for i, intersection in enumerate(INTERSECTIONS):
            traci.switch(f"int{i+1}")
            tl_id = intersection["tl_id"]
            action = actions[i]

            # Enforce minimum green time
            if self.green_timer[i] >= self.min_green:
                if action != self.current_phase[i]:
                    # Switch to new phase
                    traci.trafficlight.setPhase(tl_id, action)
                    self.current_phase[i] = action
                    self.green_timer[i] = 0
            else:
                self.green_timer[i] += 1

            # Check for emergency vehicles
            ev_flags[i] = self._detect_emergency_vehicle(
                i, intersection["lanes_in"]
            )

            # Advance simulation by 1 step
            traci.simulationStep()

        # Get new state
        state = self._get_state()

        # Compute reward
        reward, ev_flags = self._compute_reward(actions, ev_flags)

        # Check if episode is done
        self.step_count += 1
        done = self.step_count >= self.episode_length
        truncated = False

        info = {
            "step": self.step_count,
            "ev_flags": ev_flags,
            "reward": reward,
        }

        return state, reward, done, truncated, info

    def close(self):
        """Clean up SUMO instances"""
        self._close_sumo()