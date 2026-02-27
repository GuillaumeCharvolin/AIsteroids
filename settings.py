
# ── Game Constants ──

SCREEN_WIDTH = 160
SCREEN_HEIGHT = 210

# ── RAM Addresses (from OCAtari reverse-engineering) ──

AST_Y_START = 3       # First asteroid Y slot index
AST_Y_END = 20        # End (exclusive) of asteroid Y slots → range(3, 20)
AST_X_OFFSET = 18     # X index = Y index + 18
RAM_SHIP_X = 73
RAM_SHIP_Y = 74
RAM_SHIP_ROT = 60
SHIP_DEAD_Y = 224     # ram[74] value when ship is dead
INACTIVE_BIT = 128    # Bit 7 of Y byte = inactive slot
ROT_MASK = 0x0F       # Lower 4 bits of ram[60]
ROT_STEPS = 16        # Number of rotation steps in a full circle

# ── Reward Shaping ──

DEATH_PENALTY = -20
MAX_REWARD = 1.0
STARTING_LIVES = 4

# ── Observation (Polar Sensors) ──

N_SENSORS = 8            # Number of closest rocks tracked
MAX_DIST_NORM = 150.0    # Normalization factor for distance
SENSOR_PAD_VALUE = 1.0   # Padding for empty sensor slots
