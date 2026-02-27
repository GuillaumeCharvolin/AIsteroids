import gymnasium as gym
import ale_py
import time
import random
import os
from collections import deque
from asteroid_state import AsteroidState
from brain import Brain
from itertools import count
import torch

gym.register_envs(ale_py)
os.environ['SDL_AUDIODRIVER'] = 'dummy'

RENDER_EVERY = 50  # Show a human-rendered game every N episodes
N_EPISODES = 1000

env_fast = gym.make("ALE/Asteroids-v5", obs_type="ram", max_episode_steps=400)
brain = Brain()
env_render = None  # Created on-demand to avoid window freeze

MA_WINDOW = 20  # Moving average window (episodes)
reward_history = deque(maxlen=MA_WINDOW)   # raw game score
custom_history = deque(maxlen=MA_WINDOW)   # shaped reward

try:
    for episode in range(N_EPISODES):
        render = (episode % RENDER_EVERY == 0)
        if render:
            env_render = gym.make("ALE/Asteroids-v5", obs_type="ram", render_mode="human", max_episode_steps=100)
            env = env_render
        else:
            env = env_fast

        obs, info = env.reset()
        state = AsteroidState()
        obs_custom = state.get_custom_obs()

        total_custom_reward = 0

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
            total_custom_reward += state.custom_reward

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
                #time.sleep(0.01)

            if terminated or truncated:
                custom_history.append(total_custom_reward)
                ma_custom = sum(custom_history) / len(custom_history)


                print(
                    f"Ep {episode:3d} | steps: {step:3d} | "
                    f"custom: {int(total_custom_reward):3d} (MA{MA_WINDOW}: {ma_custom:4.2f}) | "
                    f"eps: {brain.eps_threshold:.3f}"
                )
                if render:
                    env_render.close()
                    env_render = None
                break

except KeyboardInterrupt:
    print("\nSimulation Terminated.")
finally:
    env_fast.close()
    if env_render is not None:
        env_render.close()