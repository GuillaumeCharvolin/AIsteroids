import math
import numpy as np
from settings import *

def _x_position(value):
    """Decode the non-linear X position encoding used by Atari 2600 Asteroids.
    Raw RAM byte -> screen pixel X coordinate (approx 0-160)."""
    value = int(value)  # cast from numpy uint8 to avoid overflow
    ls = value & 15
    add = 8 * ((value >> 7) & 1)
    sub = (value >> 4) & 7
    if value == 0:
        return 64
    elif value == 1:
        return 4
    elif ls % 2 == 0:
        mult = (ls / 2) - 1
        return int(97 + 15 * mult + add - sub)
    else:
        mult = ((ls - 1) / 2) - 1
        return int(10 + 15 * mult + add - sub)

def _y_position(raw_y):
    """Convert raw RAM Y value to screen pixel Y coordinate."""
    return 184 - 2 * (80 - int(raw_y))  # cast from numpy uint8


class AsteroidState:
    AST_SLOTS = list(range(AST_Y_START, AST_Y_END))

    def __init__(self):
        self.ship_x = 0
        self.ship_y = 0
        self.ship_rot = 0
        self.asteroids = []
        self.lives = STARTING_LIVES
        self.is_dead = False
        self.custom_reward = 0

    def _get_polar_coords(self, obj_x, obj_y):
        dx = math.remainder(int(obj_x) - int(self.ship_x), SCREEN_WIDTH)
        dy = math.remainder(int(obj_y) - int(self.ship_y), SCREEN_HEIGHT)

        dist = math.sqrt(dx**2 + dy**2)
        abs_angle = math.atan2(dy, dx)

        # Ship rotation: 0=up, increases counterclockwise, ROT_STEPS per full circle.
        # atan2 frame: 0=right, π/2=down, -π/2=up.
        ship_forward = -math.pi / 2 - self.ship_rot * (2 * math.pi / ROT_STEPS)
        # Relative angle: 0 = directly ahead, positive = right, negative = left
        rel_angle = math.remainder(abs_angle - ship_forward, 2 * math.pi)

        return dist, rel_angle

    def process_slots(self, obs):
        raw_polar_list = []
        seen_xy = []
        
        for idx in self.AST_SLOTS:
            raw_y = obs[idx]
            raw_x = obs[idx + AST_X_OFFSET]

            if raw_x == 0 or (raw_y & INACTIVE_BIT):
                continue

            x = _x_position(raw_x)
            y = _y_position(raw_y)

            seen_xy.append((x, y))
            dist, rel_angle = self._get_polar_coords(x, y)
            raw_polar_list.append((dist, rel_angle))
            
        return raw_polar_list

    def update(self, obs, reward, info):

        self.ship_x = _x_position(obs[RAM_SHIP_X])
        self.ship_y = _y_position(obs[RAM_SHIP_Y]) if obs[RAM_SHIP_Y] != SHIP_DEAD_Y else 0
        self.ship_rot = obs[RAM_SHIP_ROT] & ROT_MASK

        raw_list = self.process_slots(obs)
        self.asteroids = sorted(raw_list, key=lambda r: r[0])

        current_lives = info.get("lives", self.lives)
        self.is_dead = current_lives < self.lives
        self.lives = current_lives
        
        if self.is_dead:
            self.custom_reward = DEATH_PENALTY
        else:
            self.custom_reward = min(MAX_REWARD, reward) if reward > 0 else 0

    def get_custom_obs(self):
        """Returns [dist, angle] for the N_SENSORS closest rocks, normalized."""
        obs_list = []
        for i in range(N_SENSORS):
            if i < len(self.asteroids):
                dist, angle = self.asteroids[i]
                obs_list.extend([dist / MAX_DIST_NORM, angle / math.pi])
            else:
                obs_list.extend([SENSOR_PAD_VALUE, SENSOR_PAD_VALUE])
        return np.array(obs_list, dtype=np.float32)

    def __str__(self):
        rock_list = " ".join(f"[{d:3.0f}, {math.degrees(a):4.0f}°]" for d, a in self.asteroids)
        return (f"ROCKS: {len(self.asteroids):2} | "f"REWARD: {self.custom_reward:2} | \n"f"SENSORS: {rock_list}") 