
class AsteroidState:
    def __init__(self):
        self.ship_x = 0
        self.ship_y = 0
        self.ship_rot = 0
        self.asteroids = []
        self.score = 0
        self.lives = 0
        self.is_dead = False

    def update(self, obs):
        """Update with a full-sweep of Atari object memory."""
        self.ship_x = obs[73]
        self.ship_y = obs[74]
        self.ship_rot = obs[60]
        
        new_asteroids = []
        
        for i in range(15):
            x = obs[3 + i]
            y = obs[12 + i]
            
            if x > 0 and y > 0 and not (x == self.ship_x and y == self.ship_y):
                new_asteroids.append({'id': i, 'x': x, 'y': y})

        for i in range(5):
            x_small = obs[30 + i]
            y_small = obs[35 + i]
            if x_small > 0 and y_small > 0:
                new_asteroids.append({'id': 100 + i, 'x': x_small, 'y': y_small})

        self.asteroids = new_asteroids
        self.score = obs[102]
        self.lives = obs[107]
        self.is_dead = (obs[70] > 0)

    def __str__(self):
        res = f"SHIP: ({self.ship_x:3}, {self.ship_y:3}) | ROCKS: {len(self.asteroids):2} | SCORE: {self.score}\n"
        rock_list = " ".join([f"[{r['x']},{r['y']}]" for r in self.asteroids])
        return res + f"SENSORS: {rock_list}"