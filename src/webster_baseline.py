import traci

class WebsterBaseline:
    """
    Fixed-time signal controller based on Webster's method.
    Timing plans calculated offline for medium demand (600 veh/hr).
    Runs the same fixed cycle regardless of real-time conditions.
    """

    def __init__(self):
        # Webster's optimal cycle length formula:
        # C = (1.5L + 5) / (1 - Y)
        # where L = total lost time, Y = sum of critical flow ratios
        # For medium demand (600 veh/hr), 2-phase intersection:
        # Cycle = 60s, Green splits: NS=27s, EW=27s, Amber=3s each

        # Intersection 1 & 3: 4 phases, 60s cycle
        self.phases_int13 = [
            {"phase": 0, "duration": 27},  # NS through
            {"phase": 1, "duration": 27},  # EW through
            {"phase": 2, "duration": 15},  # NS left-turn
            {"phase": 3, "duration": 15},  # EW left-turn
        ]

        # Intersection 2: 6 phases, 90s cycle (more approaches)
        self.phases_int2 = [
            {"phase": 0, "duration": 27},  # NS through
            {"phase": 1, "duration": 27},  # EW through
            {"phase": 2, "duration": 12},  # NS left-turn
            {"phase": 3, "duration": 12},  # EW left-turn
            {"phase": 4, "duration": 10},  # NE diagonal
            {"phase": 5, "duration": 10},  # SW diagonal
        ]

        # Phase timers per intersection
        self.timers = [0, 0, 0]
        self.phase_idx = [0, 0, 0]

    def get_action(self, step):
        """Return current phase for each intersection based on fixed schedule."""
        actions = []

        for i in range(3):
            if i == 1:
                phases = self.phases_int2
            else:
                phases = self.phases_int13

            # Increment timer
            self.timers[i] += 1

            # Check if current phase duration exceeded
            current = phases[self.phase_idx[i]]
            if self.timers[i] >= current["duration"]:
                self.timers[i] = 0
                self.phase_idx[i] = (self.phase_idx[i] + 1) % len(phases)

            actions.append(phases[self.phase_idx[i]]["phase"])

        return actions

    def reset(self):
        """Reset timers for new episode."""
        self.timers = [0, 0, 0]
        self.phase_idx = [0, 0, 0]