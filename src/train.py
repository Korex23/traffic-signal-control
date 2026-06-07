import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from src.traffic_env import TrafficEnv
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LOG_DIR        = "logs"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Training config ────────────────────────────────────────────────────────
EPISODES        = 1500
STEPS_PER_EP    = 3600
TOTAL_TIMESTEPS = EPISODES * STEPS_PER_EP  # 5,400,000


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


def train():
    print("=" * 60)
    print("  Traffic Signal Control — PPO Retrain")
    print(f"  Episodes:            {EPISODES}")
    print(f"  Steps per episode:   {STEPS_PER_EP}")
    print(f"  Total timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"  Intersection pool:   11 diverse intersections")
    print(f"  Sampling:            random 3 per episode")
    print("=" * 60)

    # ── Create environment with training mode ──────────────────────────────
    env = TrafficEnv(
        use_gui       = False,
        episode_length = STEPS_PER_EP,
        scenario      = "default",
        training_mode = True,
    )

    # ── Create PPO model ───────────────────────────────────────────────────
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

    print("\nPPO Model created")
    print(f"Policy network: {model.policy}")

    # ── Callbacks ──────────────────────────────────────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq   = STEPS_PER_EP * 10,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "ppo_traffic",
        verbose     = 1,
    )

    episode_logger = EpisodeLoggerCallback()

    # ── Train ──────────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    model.learn(
        total_timesteps     = TOTAL_TIMESTEPS,
        callback            = [checkpoint_callback, episode_logger],
        tb_log_name         = "PPO_traffic",
        reset_num_timesteps = True,
    )

    # ── Save final model ───────────────────────────────────────────────────
    model.save("checkpoints/ppo_traffic_final")
    print("\nTraining complete!")
    print("Model saved to checkpoints/ppo_traffic_final")

    env.close()


if __name__ == "__main__":
    train()