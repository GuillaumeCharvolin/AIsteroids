import gymnasium as gym
import ale_py
import time
import random
import os

gym.register_envs(ale_py)
os.environ['SDL_AUDIODRIVER'] = 'dummy' # To kill the sound

def get_action():
    return random.randint(1, 4)

env = gym.make("ALE/Asteroids-v5", obs_type="ram", render_mode="human")
obs, info = env.reset()

try:
    while True:
        
        action = get_action()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
            print("Ship Destroyed! Resetting...")

        time.sleep(0.05) # 50ms

except KeyboardInterrupt:
    print("\nSimulation Terminated.")
finally:
    env.close()