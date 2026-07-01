import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from src.traffic_env import TrafficEnv

# ── Paths ──────────────────────────────────────────────────────────────────
LOG_DIR        = "logs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Training config ────────────────────────────────────────────────────────
SEED            = 42
EPISODES        = 1500
STEPS_PER_EP    = 3600
TOTAL_TIMESTEPS = EPISODES * STEPS_PER_EP  # 5,400,000

# Separate SUMO ports/labels for the train vs eval env so their SUMO instances
# (which stay open simultaneously) never collide.
TRAIN_PORTS = [8813, 8814, 8815]
EVAL_PORTS  = [8816, 8817, 8818]


# ── Custom callback ────────────────────────────────────────────────────────
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward  = 0.0
        self.episode_count   = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]

        if self.locals["dones"][0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_reward)

            self.logger.record("episode/reward",      self.current_reward)
            self.logger.record("episode/count",       self.episode_count)
            self.logger.record("episode/mean_reward",
                               np.mean(self.episode_rewards[-10:]))

            print(
                f"Episode {self.episode_count:>4} | "
                f"Reward: {self.current_reward:>10.2f} | "
                f"Mean(last 10): {np.mean(self.episode_rewards[-10:]):>10.2f}"
            )

            self.current_reward = 0.0

        return True


# ── Env factory ────────────────────────────────────────────────────────────
def make_env(ports, label_prefix):
    def _init():
        env = TrafficEnv(
            use_gui        = False,
            episode_length = STEPS_PER_EP,
            scenario       = "default",
            training_mode  = True,
            ports          = ports,
            label_prefix   = label_prefix,
        )
        return Monitor(env)
    return _init


def train():
    print("=" * 60)
    print("  Traffic Signal Control — PPO Retrain")
    print(f"  Episodes:            {EPISODES}")
    print(f"  Steps per episode:   {STEPS_PER_EP}")
    print(f"  Total timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"  Intersection pool:   8 single-signal intersections")
    print(f"  Sampling:            random 3 per episode")
    print(f"  Demand:              randomised 0.5x-5x per episode (--scale)")
    print(f"  Seed:                {SEED}")
    print("=" * 60)

    set_random_seed(SEED)

    # ── Training env: normalise reward (obs already normalised inside env) ──
    train_env = DummyVecEnv([make_env(TRAIN_PORTS, "int")])
    train_env = VecNormalize(
        train_env, norm_obs=False, norm_reward=True, gamma=0.99
    )

    # ── Eval env: raw reward, separate ports/labels ────────────────────────
    eval_env = DummyVecEnv([make_env(EVAL_PORTS, "eval")])
    eval_env = VecNormalize(
        eval_env, norm_obs=False, norm_reward=False, training=False
    )

    # ── Create PPO model ───────────────────────────────────────────────────
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
        seed            = SEED,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
    )

    print("\nPPO Model created")
    print(f"Policy network: {model.policy}")

    # ── Callbacks ──────────────────────────────────────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq   = STEPS_PER_EP * 10,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "ppo_traffic",
        verbose     = 1,
    )

    # Periodically evaluate and keep the best policy by mean reward — the final
    # policy is often worse than an intermediate one in noisy RL training.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = CHECKPOINT_DIR,
        log_path             = LOG_DIR,
        eval_freq            = STEPS_PER_EP * 25,
        n_eval_episodes      = 3,
        deterministic        = True,
        verbose              = 1,
    )

    episode_logger = EpisodeLoggerCallback()

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = [checkpoint_callback, eval_callback, episode_logger],
        tb_log_name         = "PPO_traffic",
        reset_num_timesteps = True,
    )

    # ── Save final model + normalisation stats ─────────────────────────────
    model.save("checkpoints/ppo_traffic_final")
    train_env.save("checkpoints/vecnormalize.pkl")
    print("\nTraining complete!")
    print("Final model: checkpoints/ppo_traffic_final.zip")
    print("Best model:  checkpoints/best_model.zip  (selected by eval reward)")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
