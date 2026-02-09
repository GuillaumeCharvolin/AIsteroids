import math

class AsteroidState:
    def __init__(self):
        self.ship_x = 0
        self.ship_y = 0
        self.ship_rot = 0
        self.asteroids = []
        self.score = 0
        self.lives = 0
        self.is_dead = False

        self.WIDTH = 160
        self.HEIGHT = 210

    def _get_relative_coords(self, obj_x, obj_y):
        dx = math.remainder(obj_x - self.ship_x, self.WIDTH)
        dy = math.remainder(obj_y - self.ship_y, self.HEIGHT)
        
        return int(dx), int(dy)

    def update(self, obs):
        """Update with a full-sweep of Atari object memory."""
        self.ship_x = obs[73]
        self.ship_y = obs[74]
        self.ship_rot = obs[60]
        
        raw_list = []
        
        # 1. Sweep Primary Slots (3-17)
        for i in range(15):
            x, y = obs[3 + i], obs[12 + i]
            # Ignore 0,0 (empty) and the ship's own position
            if (x != 0 or y != 0) and not (x == self.ship_x and y == self.ship_y):
                dx, dy = self._get_relative_coords(x, y)
                raw_list.append({'dx': dx, 'dy': dy})

        # 2. Sweep Debris Slots (30-34)
        for i in range(5):
            x, y = obs[30 + i], obs[35 + i]
            if (x != 0 or y != 0):
                dx, dy = self._get_relative_coords(x, y)
                raw_list.append({'dx': dx, 'dy': dy})
        
        self.asteroids = raw_list
        self.score = obs[102]
        self.lives = obs[107]
        self.is_dead = (obs[70] > 0)

    def __str__(self):
        res = f"SHIP: ({self.ship_x:3}, {self.ship_y:3}) | ROCKS: {len(self.asteroids):2} | SCORE: {self.score}\n"
        rock_list = " ".join([f"[{r['dx']:3},{r['dy']:3}]" for r in self.asteroids])
        return res + f"SENSORS: {rock_list}"