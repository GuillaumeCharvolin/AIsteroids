import gymnasium as gym
import ale_py
import time
import random
import os
import numpy as np
from collections import deque
from asteroid_state import AsteroidState
from brain import Brain
from itertools import count
import torch
from settings import *

gym.register_envs(ale_py)
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# ── Configure PyTorch threading ──
torch.set_num_threads(TORCH_THREADS)
try:
    torch.set_num_interop_threads(max(1, TORCH_THREADS // 2))
except RuntimeError:
    pass  # Thread pool already initialized

RENDER_EVERY = 200   # Show a human-rendered game every N episodes
N_EPISODES = 1000

MA_WINDOW = 40
reward_history = deque(maxlen=MA_WINDOW)
custom_history = deque(maxlen=MA_WINDOW)

# ── Create vectorized environment (parallel workers) ──
def make_env():
    """Factory function for AsyncVectorEnv."""
    def _init():
        env = gym.make("ALE/Asteroids-v5", obs_type="ram", max_episode_steps=400)
        return env
    return _init

vec_env = gym.vector.AsyncVectorEnv(
    [make_env() for _ in range(NUM_WORKERS)],
)

brain = Brain()
env_render = None  # Created on-demand for rendered episodes

# One AsteroidState per parallel worker
states = [AsteroidState() for _ in range(NUM_WORKERS)]
worker_total_rewards = [0.0] * NUM_WORKERS  # Cumulative reward per worker's current episode

# Track episodes completed across all workers
episodes_done = 0
global_step = 0
next_render_ep = RENDER_EVERY  # Threshold-based trigger (handles episode count jumps)

print(f"🚀 Starting parallel training: {NUM_WORKERS} workers, "
      f"{TORCH_THREADS} PyTorch threads, optimize every {OPTIMIZE_EVERY} steps")

try:
    # ── Initial reset ──
    obs_batch, infos = vec_env.reset()
    obs_custom_batch = np.array([
        states[i].get_custom_obs() for i in range(NUM_WORKERS)
    ], dtype=np.float32)

    while episodes_done < N_EPISODES:
        # ── Rendered episode (single env, separate) ──
        if episodes_done >= next_render_ep:
            next_render_ep += RENDER_EVERY
            render_ep = episodes_done
            env_render = gym.make("ALE/Asteroids-v5", obs_type="ram",
                                 render_mode="human", max_episode_steps=500)
            r_obs, r_info = env_render.reset()
            r_state = AsteroidState()
            r_obs_custom = r_state.get_custom_obs()
            r_total_custom = 0

            print(f"\n{'='*40}")
            print(f"  🎮 EPISODE {render_ep} (RENDERED)")
            print(f"  eps={brain.eps_threshold:.3f}")
            print(f"{'='*40}")

            for step in count():
                action = brain.select_action(r_obs_custom, False)
                r_obs, reward, terminated, truncated, r_info = env_render.step(action)
                r_state.update(r_obs, reward, r_info)
                r_total_custom += r_state.custom_reward
                r_obs_custom = r_state.get_custom_obs()
                print(r_state)

                if terminated or truncated:
                    print(f"  Rendered ep done | custom: {int(r_total_custom)}")
                    break

            env_render.close()
            env_render = None

        # ── Save state tensors before stepping ──
        state_tensors = [
            torch.tensor(obs_custom_batch[i], dtype=torch.float32).unsqueeze(0)
            for i in range(NUM_WORKERS)
        ]

        # ── Select actions for all workers ──
        actions = brain.select_actions(obs_custom_batch)

        # ── Step all environments in parallel ──
        obs_batch, rewards, terminateds, truncateds, infos = vec_env.step(actions)

        # ── Process each worker's transition ──
        for i in range(NUM_WORKERS):
            # Check if this worker's episode just ended (auto-reset happened)
            done = terminateds[i] or truncateds[i]

            # Extract per-worker info from vectorized info dict
            worker_info = {}
            if done:
                # Auto-reset already happened: lives is back to 4.
                # Force death detection so DEATH_PENALTY is applied.
                worker_info["lives"] = 0
            elif "lives" in infos:
                worker_info["lives"] = int(infos["lives"][i])

            states[i].update(obs_batch[i], rewards[i], worker_info)

            custom_reward = states[i].custom_reward
            worker_total_rewards[i] += custom_reward

            obs_custom_new = states[i].get_custom_obs()
            reward_tensor = torch.tensor([custom_reward])
            action_tensor = torch.tensor([[actions[i] - 1]], dtype=torch.long)

            if done:
                next_state_tensor = None
            else:
                next_state_tensor = torch.tensor(obs_custom_new, dtype=torch.float32).unsqueeze(0)

            brain.memory.push(state_tensors[i], action_tensor, next_state_tensor, reward_tensor)

            if done:
                # Log the full episode reward, then reset
                ep_total = worker_total_rewards[i]
                episodes_done += 1
                states[i] = AsteroidState()
                worker_total_rewards[i] = 0.0

                custom_history.append(ep_total)
                ma_custom = sum(custom_history) / len(custom_history)

                print(
                    f"Ep {episodes_done:3d} | worker {i:2d} | "
                    f"custom: {int(ep_total):3d} (MA{MA_WINDOW}: {ma_custom:4.2f}) | "
                    f"eps: {brain.eps_threshold:.3f}"
                )

            # Update the custom obs for next step
            obs_custom_batch[i] = states[i].get_custom_obs()

        # ── Optimize (less frequently to save CPU for envs) ──
        global_step += 1
        if global_step % OPTIMIZE_EVERY == 0:
            brain.optimize_model()
            brain.soft_target_net_update()

except KeyboardInterrupt:
    print("\nSimulation Terminated.")
finally:
    vec_env.close()
    if env_render is not None:
        env_render.close()