import traci
from src.signal_utils import MAX_PHASES, build_phase_map, queue_of


class SOTLBaseline:
    """Self-Organizing Traffic Lights (Cools, Gershenson & D'Hooghe, 2013), simplified.

    Holds the current green phase and integrates demand on the RED approaches into an
    accumulator (kappa). Each step kappa grows by the number of vehicles queued on the
    non-served green phases. Once kappa exceeds a threshold theta *and* the current
    phase has been green at least min_green steps, it switches to the green phase with
    the most waiting demand and resets kappa.

    Demand-reactive and hysteretic — it holds a phase until pressure builds, rather
    than re-optimising greedily every step like max-pressure. Operates in the env's
    action space (reachable protected greens).
    """

    def __init__(self, tl_id="C", theta=60, min_green=10):
        self.tl_id     = tl_id
        self.theta     = theta
        self.min_green = min_green
        self.green_phases = [[], [], []]
        self.served_lanes = [{}, {}, {}]
        self.current = [0, 0, 0]
        self.kappa   = [0, 0, 0]
        self.elapsed = [0, 0, 0]

    def reset(self):
        self.current = [0, 0, 0]
        self.kappa   = [0, 0, 0]
        self.elapsed = [0, 0, 0]
        for i in range(3):
            traci.switch(f"int{i+1}")
            greens, served = build_phase_map(self.tl_id, MAX_PHASES[i])
            self.green_phases[i] = greens
            self.served_lanes[i] = served
            if greens:
                self.current[i] = greens[0]

    def get_action(self, step):
        actions = []
        for i in range(3):
            traci.switch(f"int{i+1}")
            greens = self.green_phases[i]
            served = self.served_lanes[i]
            cur    = self.current[i]

            self.elapsed[i] += 1

            # Accumulate demand waiting on the red (non-current) green phases.
            red_demand = sum(queue_of(served[p]) for p in greens if p != cur)
            self.kappa[i] += red_demand

            if (len(greens) > 1
                    and self.elapsed[i] >= self.min_green
                    and self.kappa[i] >= self.theta):
                # Switch to the red phase serving the most waiting demand.
                best_p = max((p for p in greens if p != cur),
                             key=lambda p: queue_of(served[p]))
                self.current[i] = best_p
                self.kappa[i]   = 0
                self.elapsed[i] = 0

            actions.append(self.current[i])
        return actions
