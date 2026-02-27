import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from settings import *
from collections import namedtuple, deque
import random
import math


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations=OBS_SIZE, n_actions=N_ACTIONS):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, NN_LAYER_SIZE)
        self.layer2 = nn.Linear(NN_LAYER_SIZE, NN_LAYER_SIZE)
        self.layer3 = nn.Linear(NN_LAYER_SIZE, NN_LAYER_SIZE)
        self.layer4 = nn.Linear(NN_LAYER_SIZE, n_actions)

    # Called with either one element to determine next action, or a batch
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class Brain:

    def __init__(self):
        self.policy_net = DQN(OBS_SIZE, N_ACTIONS)
        self.target_net = DQN(OBS_SIZE, N_ACTIONS)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.action_taken_count = 0
        self.eps_threshold = EPS_START

    def decay_exploration(self):
        self.eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.action_taken_count / EPS_DECAY)

    def select_action(self, obs, explore=True):
        """Select action given observation (numpy array from get_custom_obs).
        Returns an int action for env.step()."""
        self.decay_exploration()
        self.action_taken_count += 1

        epsilon_calculated = random.random() if explore else self.eps_threshold + 1

        if epsilon_calculated > self.eps_threshold:
            with torch.no_grad():
                # obs is a numpy array → convert to tensor, add batch dim [1, OBS_SIZE]
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                # policy_net returns [1, N_ACTIONS] → indices 0-3, env expects 1-4
                return self.policy_net(obs_tensor).max(1).indices.item() + 1

        # Random exploration (biased toward fire)
        action = random.randint(1, 4)
        if action != 1:
            action = random.randint(1, 4)
        return action

    def select_actions(self, obs_batch):
        """Select actions for a batch of observations (numpy array [N, OBS_SIZE]).
        Returns numpy array of int actions for env.step()."""
        n = len(obs_batch)
        self.decay_exploration()
        self.action_taken_count += 1  # Count per batch, not per worker, to match original decay rate

        if random.random() > self.eps_threshold:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
                return self.policy_net(obs_tensor).max(1).indices.numpy() + 1

        # Random exploration (biased toward fire)
        actions = np.array([random.randint(1, 4) for _ in range(n)])
        for i in range(n):
            if actions[i] != 1:
                actions[i] = random.randint(1, 4)
        return actions

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_target_net_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
