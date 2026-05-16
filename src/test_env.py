from src.traffic_env import TrafficEnv
import numpy as np

def test_env(episodes=5, use_gui=False):
    env = TrafficEnv(use_gui=use_gui, episode_length=100)
    
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        step = 0

        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if done:
                break

        print(f"Episode {ep+1} | Steps: {step} | Total Reward: {total_reward:.2f}")

    env.close()
    print("All episodes complete!")

if __name__ == "__main__":
    test_env(episodes=5, use_gui=False)