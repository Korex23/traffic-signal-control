import os
import sys
import random
import gymnasium as gym
import numpy as np
import traci

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")

# ── Intersection pool ──────────────────────────────────────────────────────
INTERSECTION_POOL = [
    {
        "id": "intersection1",
        "tl_ids": ["C"],
        "lanes_in": [
            "N2C_0", "N2C_1", "S2C_0", "S2C_1",
            "E2C_0", "E2C_1", "W2C_0", "W2C_1",
        ],
        "num_phases": 4,
        "config": os.path.join("networks", "intersection1", "intersection1.sumocfg"),
    },
    {
        "id": "intersection2",
        "tl_ids": ["C"],
        "lanes_in": [
            "N2C_0", "N2C_1", "N2C_2",
            "S2C_0", "S2C_1", "S2C_2",
            "E2C_0", "E2C_1",
            "W2C_0", "W2C_1",
            "NE2C_0", "NE2C_1",
            "SW2C_0", "SW2C_1",
        ],
        "num_phases": 6,
        "config": os.path.join("networks", "intersection2", "intersection2.sumocfg"),
    },
    {
        "id": "intersection3",
        "tl_ids": ["C"],
        "lanes_in": [
            "N2C_0", "N2C_1", "S2C_0", "S2C_1",
            "E2C_0", "E2C_1", "W2C_0", "W2C_1",
        ],
        "num_phases": 4,
        "config": os.path.join("networks", "intersection3", "intersection3.sumocfg"),
    },
    {
        "id": "intersection5",
        "tl_ids": ["C"],
        "lanes_in": [
            "N2C_0", "N2C_1",
            "E2C_0", "E2C_1", "E2C_2",
            "W2C_0", "W2C_1", "W2C_2",
        ],
        "num_phases": 3,
        "config": os.path.join("networks", "intersection5", "intersection5.sumocfg"),
    },
    {
        "id": "intersection6",
        "tl_ids": ["C"],
        "lanes_in": [
            "S2C_0", "S2C_1",
            "NW2C_0", "NW2C_1",
            "NE2C_0", "NE2C_1",
        ],
        "num_phases": 3,
        "config": os.path.join("networks", "intersection6", "intersection6.sumocfg"),
    },
    {
        "id": "intersection7",
        "tl_ids": ["C"],
        "lanes_in": [
            "NE2C_0", "NE2C_1",
            "SW2C_0", "SW2C_1",
            "NW2C_0", "NW2C_1",
            "SE2C_0", "SE2C_1",
        ],
        "num_phases": 4,
        "config": os.path.join("networks", "intersection7", "intersection7.sumocfg"),
    },
    {
        "id": "intersection8",
        "tl_ids": ["C"],
        "lanes_in": [
            "N2C_0", "N2C_1",
            "S2C_0", "S2C_1",
            "E2C_0", "E2C_1", "E2C_2", "E2C_3",
            "W2C_0", "W2C_1", "W2C_2", "W2C_3",
        ],
        "num_phases": 4,
        "config": os.path.join("networks", "intersection8", "intersection8.sumocfg"),
    },
    {
        "id": "intersection11",
        "tl_ids": ["C"],
        "lanes_in": [
            "E2C_0", "E2C_1", "E2C_2", "E2C_3",
            "W2C_0", "W2C_1", "W2C_2", "W2C_3",
            "N2C_0", "S2C_0",
        ],
        "num_phases": 4,
        "config": os.path.join("networks", "intersection11", "intersection11.sumocfg"),
    },
]

# ── Evaluation configs (demand scenarios) ──────────────────────────────────
SUMO_CONFIGS = {
    "low": [
        os.path.join("networks", "intersection1", "intersection1_low.sumocfg"),
        os.path.join("networks", "intersection2", "intersection2_low.sumocfg"),
        os.path.join("networks", "intersection3", "intersection3_low.sumocfg"),
    ],
    "medium": [
        os.path.join("networks", "intersection1", "intersection1_medium.sumocfg"),
        os.path.join("networks", "intersection2", "intersection2_medium.sumocfg"),
        os.path.join("networks", "intersection3", "intersection3_medium.sumocfg"),
    ],
    "peak": [
        os.path.join("networks", "intersection1", "intersection1_peak.sumocfg"),
        os.path.join("networks", "intersection2", "intersection2_peak.sumocfg"),
        os.path.join("networks", "intersection3", "intersection3_peak.sumocfg"),
    ],
    "default": [
        os.path.join("networks", "intersection1", "intersection1.sumocfg"),
        os.path.join("networks", "intersection2", "intersection2.sumocfg"),
        os.path.join("networks", "intersection3", "intersection3.sumocfg"),
    ],
}

# ── Eval intersection definitions ──────────────────────────────────────────
EVAL_INTERSECTIONS = [
    {
        "lanes_in": [
            "N2C_0", "N2C_1", "S2C_0", "S2C_1",
            "E2C_0", "E2C_1", "W2C_0", "W2C_1",
        ]
    },
    {
        "lanes_in": [
            "N2C_0", "N2C_1", "N2C_2",
            "S2C_0", "S2C_1", "S2C_2",
            "E2C_0", "E2C_1", "W2C_0", "W2C_1",
            "NE2C_0", "NE2C_1", "SW2C_0", "SW2C_1",
        ]
    },
    {
        "lanes_in": [
            "N2C_0", "N2C_1", "S2C_0", "S2C_1",
            "E2C_0", "E2C_1", "W2C_0", "W2C_1",
        ]
    },
]

# ── Constants ──────────────────────────────────────────────────────────────
STATE_SIZE   = 93
ALPHA        = 50.0    # EV priority weight — raised so EV bonus is a meaningful reward term
BETA         = 0.5
REWARD_SCALE = 1000.0


class TrafficEnv(gym.Env):
    """
    Adaptive traffic signal control environment.
    Training mode : randomly samples 3 intersections from pool each episode.
    Evaluation mode: uses fixed demand-specific sumocfg files.
    State  : 93-dimensional vector (queue, wait, phase, EV flag)
    Action : MultiDiscrete([6, 6, 6]) — one phase per intersection slot
    Reward : R = (−avg_wait + α·EV − β·Δavg_queue) / REWARD_SCALE
    """

    def __init__(
        self,
        use_gui            = False,
        episode_length     = 3600,
        scenario           = "default",
        training_mode      = True,
        ports              = None,
        label_prefix       = "int",
        demand_scale_range = (0.5, 5.0),
        eval_demand_scale  = 1.0,
    ):
        super().__init__()

        self.use_gui            = use_gui
        self.episode_length     = episode_length
        self.scenario           = scenario
        self.training_mode      = training_mode
        self.demand_scale_range = demand_scale_range
        self.eval_demand_scale  = eval_demand_scale
        self.step_count         = 0
        self.prev_avg_queue     = 0.0

        # Action space: 6 phases max across all intersections in pool
        self.action_space = gym.spaces.MultiDiscrete([6, 6, 6])

        # Observation space: 93-dimensional normalised vector
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(93,), dtype=np.float32
        )

        self.ports          = ports if ports is not None else [8813, 8814, 8815]
        self.labels         = [f"{label_prefix}{i+1}" for i in range(3)]
        self.min_green      = 10
        self.green_timer    = [0, 0, 0]
        self.current_phase  = [0, 0, 0]

        self.selected_intersections = None

        print(f"TrafficEnv initialized | mode={'TRAINING' if training_mode else 'EVAL'}")
        print(f"Action space:      {self.action_space}")
        print(f"Observation space: {self.observation_space}")

    # ── SUMO lifecycle ─────────────────────────────────────────────────────

    def _start_sumo(self):
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        if self.training_mode:
            self.selected_intersections = random.sample(INTERSECTION_POOL, 3)
            configs = [i["config"] for i in self.selected_intersections]
            ids     = [i["id"]     for i in self.selected_intersections]
            print(f"  Sampled: {ids}")
        else:
            configs = SUMO_CONFIGS[self.scenario]
            self.selected_intersections = None

        for i, (config, port) in enumerate(zip(configs, self.ports)):
            sumo_cmd = [
                sumo_binary,
                "-c", config,
                "--no-warnings",
                "--no-step-log",
                "--random",
                "--time-to-teleport", "-1",
                "--end", "999999",
                "--quit-on-end", "false",
            ]

            # Demand scaling via SUMO --scale.
            #   training: random factor per episode (domain randomisation, low → peak)
            #   eval:     fixed factor so demand scenarios are controlled/reproducible
            if self.training_mode:
                scale = random.uniform(*self.demand_scale_range)
            else:
                scale = self.eval_demand_scale
            sumo_cmd += ["--scale", f"{scale:.2f}"]

            traci.start(sumo_cmd, port=port, label=self.labels[i])

    def _close_sumo(self):
        for i in range(3):
            try:
                traci.switch(self.labels[i])
                traci.close()
            except Exception:
                pass

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_lanes(self, slot):
        if self.training_mode and self.selected_intersections:
            return self.selected_intersections[slot]["lanes_in"]
        return EVAL_INTERSECTIONS[slot]["lanes_in"]

    def _get_tl_ids(self, slot):
        if self.training_mode and self.selected_intersections:
            return self.selected_intersections[slot]["tl_ids"]
        return ["C"]

    def _get_max_phases(self, slot):
        if self.training_mode and self.selected_intersections:
            return self.selected_intersections[slot]["num_phases"]
        return [4, 6, 4][slot]

    # ── State ──────────────────────────────────────────────────────────────

    def _get_state(self):
        state = []

        for i in range(3):
            traci.switch(self.labels[i])
            lanes = self._get_lanes(i)

            # Queue lengths — normalised by max 50 vehicles
            queue_lengths = []
            for lane in lanes:
                try:
                    q = traci.lane.getLastStepHaltingNumber(lane)
                except Exception:
                    q = 0
                queue_lengths.append(q / 50.0)

            # Waiting times — normalised by max 300 seconds
            waiting_times = []
            for lane in lanes:
                try:
                    w = traci.lane.getWaitingTime(lane)
                except Exception:
                    w = 0
                waiting_times.append(min(w, 300.0) / 300.0)

            # Pad / truncate to fixed size
            max_lanes = 12
            queue_lengths = (queue_lengths + [0.0] * max_lanes)[:max_lanes]
            waiting_times = (waiting_times + [0.0] * max_lanes)[:max_lanes]

            # Active phase one-hot
            max_phases    = 6
            phase_one_hot = [0.0] * max_phases
            try:
                cp = traci.trafficlight.getPhase(self._get_tl_ids(i)[0])
                if cp < max_phases:
                    phase_one_hot[cp] = 1.0
            except Exception:
                pass

            # EV flag
            ev_flag = self._detect_ev(i, lanes)

            state.extend(queue_lengths)
            state.extend(waiting_times)
            state.extend(phase_one_hot)
            state.append(float(ev_flag))

        return np.array(state, dtype=np.float32)

    # ── EV detection ───────────────────────────────────────────────────────

    def _detect_ev(self, slot, lanes):
        try:
            for lane in lanes:
                for v in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getTypeID(v) == "emergency":
                        return 1
        except Exception:
            pass
        return 0

    # ── Reward ─────────────────────────────────────────────────────────────

    def _compute_reward(self, actions, ev_flags):
        total_waiting = 0.0
        total_queue   = 0.0
        total_lanes   = 0
        ev_bonus      = 0.0

        for i in range(3):
            traci.switch(self.labels[i])
            lanes = self._get_lanes(i)

            for lane in lanes:
                try:
                    total_waiting += traci.lane.getWaitingTime(lane)
                    total_queue   += traci.lane.getLastStepHaltingNumber(lane)
                    total_lanes   += 1
                except Exception:
                    pass

            if ev_flags[i]:
                ev_bonus += ALPHA

        # Per-lane averages — makes reward comparable across intersection sizes
        avg_waiting = total_waiting / max(total_lanes, 1)
        avg_queue   = total_queue   / max(total_lanes, 1)

        delta_avg           = avg_queue - self.prev_avg_queue
        self.prev_avg_queue = avg_queue

        reward = (-avg_waiting + ev_bonus - BETA * delta_avg) / REWARD_SCALE

        return reward, ev_flags

    # ── Gymnasium interface ────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._close_sumo()

        self.step_count     = 0
        self.prev_avg_queue = 0.0
        self.green_timer    = [0, 0, 0]
        self.current_phase  = [0, 0, 0]

        self._start_sumo()

        return self._get_state(), {}

    def step(self, actions):
        ev_flags = [0, 0, 0]

        for i in range(3):
            traci.switch(self.labels[i])
            action     = int(actions[i])
            max_phases = self._get_max_phases(i)
            tl_ids     = self._get_tl_ids(i)

            # Enforce minimum green time
            if self.green_timer[i] >= self.min_green:
                safe_action = action % max_phases
                if safe_action != self.current_phase[i]:
                    try:
                        traci.trafficlight.setPhase(tl_ids[0], safe_action)
                    except Exception:
                        pass
                    self.current_phase[i] = safe_action
                    self.green_timer[i]   = 0
            else:
                self.green_timer[i] += 1

            ev_flags[i] = self._detect_ev(i, self._get_lanes(i))
            traci.simulationStep()

        state             = self._get_state()
        reward, ev_flags  = self._compute_reward(actions, ev_flags)
        self.step_count  += 1
        done              = self.step_count >= self.episode_length

        return state, reward, done, False, {
            "step":     self.step_count,
            "ev_flags": ev_flags,
            "reward":   reward,
        }

    def close(self):
        self._close_sumo()