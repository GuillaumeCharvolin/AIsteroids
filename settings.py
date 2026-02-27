
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

DEATH_PENALTY = -3
MAX_REWARD = 1.0
STARTING_LIVES = 4

# ── Observation (Polar Sensors) ──

N_SENSORS = 4            # Number of closest rocks tracked
OBS_SIZE = N_SENSORS * 2 # [rocks distance, rocks polar] for every rocks      
MAX_DIST_NORM = 132.0    # Max toroidal distance: sqrt(80² + 105²)
SENSOR_PAD_DIST = 1.0    # Empty sensor distance = far away
SENSOR_PAD_ANGLE = -1.0  # Empty sensor angle = behind the ship (−π)

# ── DQN Hyperparameters ──

BATCH_SIZE = 256 # Number of transitions sampled from the replay buffer
GAMMA = 0.99 # Discount factor of predicted future ations reward

# Epsilon decay variable - Each action has epsilon time chance to be ignored and replaced by exploration action
EPS_START = 0.3
EPS_END = 0.01
EPS_DECAY = 10000 # Controls the rate of exponential decay of epsilon, higher means a slower decay

TAU = 0.005 # Update rate of target network

LR = 0.001

N_ACTIONS = 4 # Number of actions possible

NN_LAYER_SIZE = 128

MEMORY_SIZE = 30000

# ── Parallelism ──

NUM_WORKERS = 20          # Number of parallel Atari environments (≤ 32 cores)
TORCH_THREADS = 8         # Threads for PyTorch CPU ops (leave cores for envs)
OPTIMIZE_EVERY = 2        # Run optimize_model every N steps (amortize cost)