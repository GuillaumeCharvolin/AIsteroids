import gymnasium as gym
import ale_py
import time
import random
import os
from asteroid_state import AsteroidState

gym.register_envs(ale_py)
os.environ['SDL_AUDIODRIVER'] = 'dummy' # To kill the sound

def get_action():
    action = random.randint(1, 4)
    if (action != 1):
        action = random.randint(1,4)
    return action

env = gym.make("ALE/Asteroids-v5", obs_type="ram", render_mode="human", difficulty=3, mode=31)
obs, info = env.reset()

state = AsteroidState()

try:
    while True:
        action = get_action()
        obs, reward, terminated, truncated, info = env.step(action)
        state.update(obs)
        print(state)

        if terminated or truncated:
            obs, info = env.reset()
            print("Ship Destroyed! Resetting...")

        time.sleep(0.05) # 50ms

except KeyboardInterrupt:
    print("\nSimulation Terminated.")
finally:
    env.close()