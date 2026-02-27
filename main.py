import gymnasium as gym
import ale_py
import time
import random
import os
from asteroid_state import AsteroidState
from brain import Brain
from itertools import count
import torch

gym.register_envs(ale_py)
os.environ['SDL_AUDIODRIVER'] = 'dummy'

RENDER_EVERY = 100  # Show a human-rendered game every N episodes
N_EPISODES = 500

env_fast = gym.make("ALE/Asteroids-v5", obs_type="ram")
env_render = gym.make("ALE/Asteroids-v5", obs_type="ram", render_mode="human")
brain = Brain()

try:
    for episode in range(N_EPISODES):
        render = (episode % RENDER_EVERY == 0)
        env = env_render if render else env_fast

        obs, info = env.reset()
        state = AsteroidState()
        obs_custom = state.get_custom_obs()

        if render:
            print(f"\n{'='*40}")
            print(f"  🎮 EPISODE {episode} (RENDERED)")
            print(f"  eps={brain.eps_threshold:.3f}")
            print(f"{'='*40}")

        for step in count():
            state_tensor = torch.tensor(obs_custom, dtype=torch.float32).unsqueeze(0)

            action = brain.select_action(obs_custom)
            obs, reward, terminated, truncated, info = env.step(action)
            state.update(obs, reward, info)

            obs_custom = state.get_custom_obs()
            reward_tensor = torch.tensor([state.custom_reward])
            action_tensor = torch.tensor([[action - 1]], dtype=torch.long)

            if terminated or truncated:
                next_state_tensor = None
            else:
                next_state_tensor = torch.tensor(obs_custom, dtype=torch.float32).unsqueeze(0)

            brain.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
            brain.optimize_model()
            brain.soft_target_net_update()

            if render:
                print(state)
                time.sleep(0.03)

            if terminated or truncated or step >= 400:
                print(f"Episode {episode} ended after {step} steps")
                break

except KeyboardInterrupt:
    print("\nSimulation Terminated.")
finally:
    env_fast.close()
    env_render.close()