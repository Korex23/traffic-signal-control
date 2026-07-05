import traci
from src.signal_utils import MAX_PHASES, build_phase_map, queue_of


class MaxPressureBaseline:
    """Max-pressure control (Varaiya, 2013), queue-based approximation.

    At every decision step, for each intersection independently, activate the green
    phase whose served incoming lanes hold the greatest total queue ("pressure").
    Downstream/outgoing queues are ~0 here (movements exit to sinks), so pressure
    reduces to the served upstream queue.

    Greedy and per-step (contrast with SOTL's hysteresis). The env still enforces a
    minimum green time, so the chosen phase is held at least min_green steps before
    it can change. Operates in the env's action space (reachable protected greens).
    """

    def __init__(self, tl_id="C"):
        self.tl_id        = tl_id
        self.green_phases = [[], [], []]   # slot -> candidate green phase indices
        self.served_lanes = [{}, {}, {}]   # slot -> {phase: [served incoming lanes]}

    def reset(self):
        for i in range(3):
            traci.switch(f"int{i+1}")
            greens, served = build_phase_map(self.tl_id, MAX_PHASES[i])
            self.green_phases[i] = greens
            self.served_lanes[i] = served

    def get_action(self, step):
        actions = []
        for i in range(3):
            traci.switch(f"int{i+1}")
            greens = self.green_phases[i]

            best_p, best_pressure = greens[0], -1.0
            for p in greens:
                pressure = queue_of(self.served_lanes[i][p])
                if pressure > best_pressure:
                    best_pressure, best_p = pressure, p

            actions.append(best_p)
        return actions
