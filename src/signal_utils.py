import traci

# Env action space per eval slot (intersection1, intersection2, intersection3).
# The env applies action % max_phases and calls setPhase, so only program phases
# 0..max_phases-1 are reachable. Baselines must choose within this same space to be
# a fair comparison with the RL agent.
MAX_PHASES = [4, 6, 4]


def build_phase_map(tl_id, max_phases):
    """Read the traffic-light program live and return the reachable protected-green
    phases plus the incoming lanes each one serves.

    Returns (green_phase_indices, served) where:
      green_phase_indices : list of phase indices p (p < max_phases) whose state has
                            a protected green 'G'
      served              : {p: [incoming lane ids that get green in phase p]}

    Reading it from TraCI (getAllProgramLogics + getControlledLinks) keeps the
    baselines correct for any of the eval intersections without hardcoding movements.
    """
    logic  = traci.trafficlight.getAllProgramLogics(tl_id)[0]
    phases = logic.phases
    links  = traci.trafficlight.getControlledLinks(tl_id)

    greens, served = [], {}
    for p in range(min(max_phases, len(phases))):
        state = phases[p].state
        if "G" not in state:            # skip yellow / all-red / permissive-only phases
            continue
        lanes = set()
        for j, s in enumerate(state):
            if s in ("G", "g") and j < len(links) and links[j]:
                lanes.add(links[j][0][0])   # incoming lane of this movement
        if lanes:
            greens.append(p)
            served[p] = sorted(lanes)

    # Fallback (shouldn't trigger for int1/2/3): allow every reachable index.
    if not greens:
        greens = list(range(max_phases))
        served = {p: [] for p in greens}

    return greens, served


def queue_of(lanes):
    """Total halting (queued) vehicles across the given lanes."""
    total = 0
    for lane in lanes:
        try:
            total += traci.lane.getLastStepHaltingNumber(lane)
        except Exception:
            pass
    return total
