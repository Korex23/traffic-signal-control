# Training Improvements — Changelog

This document records the changes made to improve training quality before evaluation,
with the **before** and **after** for each change and the reasoning behind it.

Date: 2026-07-01

---

## Summary

The goal of this project is a **single, transferable traffic-signal policy** that
generalizes across intersections of different geometry (trained via domain
randomization: 3 intersections sampled per episode from a pool) and is compared
against a Webster fixed-time baseline.

The changes below target the biggest gaps that were preventing the trained policy
from being as good as it could be *before* evaluation.

Files touched:
- `src/traffic_env.py`
- `src/train.py`

Also: previous trained artifacts were deleted (`checkpoints/ppo_traffic_final.zip`
and `checkpoints/ppo_traffic_final/`) since they predated this redesign.

---

## 1. Demand randomization during training

**Why:** Each network's `.rou.xml` bakes in a single fixed traffic volume
(e.g. intersection1 ≈ 200 veh/hr). Training only ever saw one demand level per
network, so the policy would not generalize to the low/medium/peak demand range it
is meant to be evaluated on. This was the single highest-impact gap.

### Before
`src/traffic_env.py` — `_start_sumo()` built a fixed SUMO command with no demand scaling:

```python
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
    traci.start(sumo_cmd, port=port, label=f"int{i+1}")
```

### After
Every episode scales all flows by a random factor (default `0.5x`–`5x`) via SUMO's
`--scale`, so the policy sees the full low→peak volume range:

```python
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

    # Demand randomisation: scale all flows by a per-episode factor so the
    # policy sees the full range of traffic volumes (low -> peak) during training.
    if self.training_mode:
        scale = random.uniform(*self.demand_scale_range)
        sumo_cmd += ["--scale", f"{scale:.2f}"]

    traci.start(sumo_cmd, port=port, label=self.labels[i])
```

The range is configurable via the new `demand_scale_range` constructor argument
(default `(0.5, 5.0)`).

---

## 2. Best-model selection via `EvalCallback`

**Why:** Training previously only saved the *final* policy (timestep 5.4M). RL
training is noisy and non-monotonic, so the final policy is frequently worse than
an intermediate one. There was no principled way to keep the best.

### Before
`src/train.py` used only a `CheckpointCallback` (periodic dumps, never evaluated)
and saved whatever the model looked like at the end:

```python
checkpoint_callback = CheckpointCallback(
    save_freq   = STEPS_PER_EP * 10,
    save_path   = CHECKPOINT_DIR,
    name_prefix = "ppo_traffic",
    verbose     = 1,
)

episode_logger = EpisodeLoggerCallback()

model.learn(
    total_timesteps     = TOTAL_TIMESTEPS,
    callback            = [checkpoint_callback, episode_logger],
    ...
)

model.save("checkpoints/ppo_traffic_final")
```

### After
An `EvalCallback` periodically evaluates on a held-out eval env and saves
`checkpoints/best_model.zip` whenever the mean reward improves:

```python
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path = CHECKPOINT_DIR,
    log_path             = LOG_DIR,
    eval_freq            = STEPS_PER_EP * 25,
    n_eval_episodes      = 3,
    deterministic        = True,
    verbose              = 1,
)

model.learn(
    total_timesteps     = TOTAL_TIMESTEPS,
    callback            = [checkpoint_callback, eval_callback, episode_logger],
    ...
)

model.save("checkpoints/ppo_traffic_final")     # final policy
train_env.save("checkpoints/vecnormalize.pkl")  # normalisation stats
```

Result: after training you get both `ppo_traffic_final.zip` (last) and
`best_model.zip` (best-by-eval). Use `best_model.zip` for evaluation.

---

## 3. Dropped multi-signal intersections from the pool

**Why:** Intersections 9, 10, and 12 have multiple traffic lights, but the env only
set/read a phase on the *first* traffic light (`tl_ids[0]`). The other signals ran
on uncontrolled default programs, yet their queues still fed into the reward — i.e.
the agent was scored on things it could not control, adding noise that slows and
destabilizes learning.

**Decision:** Drop them for a clean training signal (multi-junction control can be
revisited later).

### Before
`INTERSECTION_POOL` contained 11 entries, including:
- `intersection9`  — `tl_ids = ["R1", "R2", "R3", "R4"]`
- `intersection10` — `tl_ids = ["C1", "C2"]`
- `intersection12` — `tl_ids = ["C", "PW", "PE"]`

### After
`INTERSECTION_POOL` contains **8 single-signal intersections**:
`intersection1, 2, 3, 5, 6, 7, 8, 11` (all `tl_ids = ["C"]`).

Intersections 9, 10, and 12 were removed. Their network files remain on disk;
they are simply no longer sampled during training.

---

## 4. Rebalanced the reward (EV priority weight)

**Why:** `avg_wait` ranges 0–300, while the emergency-vehicle bonus used
`ALPHA = 5`. That made EV prioritization roughly 1–2% of the reward magnitude, so
the agent had almost no incentive to learn it — even though EV clearance is a
stated goal and a reported metric.

### Before
```python
STATE_SIZE   = 93
ALPHA        = 5.0
BETA         = 0.5
REWARD_SCALE = 1000.0
```

### After
```python
STATE_SIZE   = 93
ALPHA        = 50.0    # EV priority weight — raised so EV bonus is a meaningful reward term
BETA         = 0.5
REWARD_SCALE = 1000.0
```

The reward formula is unchanged; only the EV weight was raised:
`R = (-avg_wait + ALPHA * EV - BETA * delta_avg_queue) / REWARD_SCALE`.

---

## 5. Exploration, reproducibility, and reward normalization

**Why:** Small, low-cost changes that make PPO training more stable and repeatable.

### Before
`src/train.py` created PPO with default exploration and no seed, on a raw
(un-normalized) single env:

```python
model = PPO(
    policy          = "MlpPolicy",
    env             = env,
    learning_rate   = 3e-4,
    n_steps         = 2048,
    gamma           = 0.99,
    clip_range      = 0.2,
    gae_lambda      = 0.95,
    n_epochs        = 10,
    batch_size      = 64,
    verbose         = 1,
    tensorboard_log = LOG_DIR,
)
```

### After
Added a small entropy coefficient (encourages phase exploration for the discrete
action space), a fixed seed, and `VecNormalize` reward normalization (observations
are already normalized to `[0,1]` inside the env, so only reward is normalized):

```python
set_random_seed(SEED)

train_env = DummyVecEnv([make_env(TRAIN_PORTS, "int")])
train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, gamma=0.99)

model = PPO(
    policy          = "MlpPolicy",
    env             = train_env,
    learning_rate   = 3e-4,
    n_steps         = 2048,
    gamma           = 0.99,
    clip_range      = 0.2,
    gae_lambda      = 0.95,
    n_epochs        = 10,
    batch_size      = 64,
    ent_coef        = 0.01,   # encourage phase exploration for discrete actions
    seed            = SEED,   # SEED = 42
    verbose         = 1,
    tensorboard_log = LOG_DIR,
)
```

---

## 6. Configurable SUMO ports and labels (enabling change for #2)

**Why:** The env hardcoded ports `8813–8815` and TraCI labels `int1..int3`. The
`EvalCallback` needs a *second* env whose SUMO instances run at the same time as the
training env's — with hardcoded ports/labels the two would collide. Making these
configurable lets train and eval envs coexist.

### Before
```python
def __init__(self, use_gui=False, episode_length=3600,
             scenario="default", training_mode=True):
    ...
    self.ports = [8813, 8814, 8815]
    ...
# TraCI connections used hardcoded labels, e.g.:
traci.start(sumo_cmd, port=port, label=f"int{i+1}")
traci.switch(f"int{i+1}")
```

### After
```python
def __init__(self, use_gui=False, episode_length=3600,
             scenario="default", training_mode=True,
             ports=None, label_prefix="int", demand_scale_range=(0.5, 5.0)):
    ...
    self.ports  = ports if ports is not None else [8813, 8814, 8815]
    self.labels = [f"{label_prefix}{i+1}" for i in range(3)]
    ...
# All TraCI calls now use the configurable labels:
traci.start(sumo_cmd, port=port, label=self.labels[i])
traci.switch(self.labels[i])
```

In `train.py`, the train env uses ports `8813–8815` (prefix `int`) and the eval env
uses ports `8816–8818` (prefix `eval`).

---

## Verification performed

- Both `src/traffic_env.py` and `src/train.py` byte-compile.
- Env smoke test (live SUMO 1.25): resets, samples from the 8-intersection pool,
  produces valid 93-dim observations in `[0, 1]`.
- Full pipeline test (~200 real timesteps, live SUMO): train (8813–8815) and eval
  (8816–8818) instances ran simultaneously without collision; `EvalCallback` fired,
  "New best mean reward!" triggered, and `best_model.zip` was written. Test
  artifacts were cleaned up afterward.

---

## Outputs after a real training run

- `checkpoints/ppo_traffic_final.zip` — final policy (last timestep)
- `checkpoints/best_model.zip` — **best policy by eval reward (use this)**
- `checkpoints/vecnormalize.pkl` — reward-normalization statistics
- `checkpoints/ppo_traffic_*_steps.zip` — periodic checkpoints
- `logs/` — TensorBoard logs (watch the eval curve to spot plateaus)

---

## Known remaining issue (not yet addressed)

`src/evaluate.py` still assumes a fixed layout (hardcoded `intersection1/2/3`,
`tl_id="C"`) and references `*_low/_medium/_peak.sumocfg` files that do not exist.
It does not yet measure the general, demand-randomized policy this training now
produces. It should be updated to evaluate on the pool and use `--scale` for the
demand scenarios.
