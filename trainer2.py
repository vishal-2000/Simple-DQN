'''Complete Information DQN (CIdqn)
'''

# from tqdm import tqdm
import progressbar

import glob
import imp
import math
import gc
import os
from sre_constants import SUCCESS
import time
import datetime
import pybullet as p
import cv2
import numpy as np
from graphviz import Digraph
import argparse
import random
import torch
import matplotlib.pyplot as plt
from time import sleep
import copy

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from Config.constants import (
    GRIPPER_PUSH_RADIUS,
    PIXEL_SIZE,
    PUSH_DISTANCE,
    WORKSPACE_LIMITS,
    TARGET_LOWER,
    TARGET_UPPER,
    orange_lower,
    orange_upper,
    BG_THRESHOLD,
    MIN_GRASP_THRESHOLDS
)

from Environments.environment_sim import Environment
from Environments.utils import sample_goal, get_pose_distance
import Environments.utils as env_utils
from V1_destination_prediction.Test_cases.tc1 import TestCase1

from create_env import get_push_start, get_max_extent_of_target_from_bottom


from collections import namedtuple, deque


import wandb

# wandb setup
number = 5
NAME = "model" + str(number)
ID = '16_action_6_obs' + str(number)
run = wandb.init(project='DQN_complete', name = NAME, id = ID)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

from V2_next_best_action.models.dqn_v2 import pushDQN2

torch.cuda.empty_cache()

# Hyperparameters
BATCH_SIZE = 256
REPLAY_MEMORY_SIZE=10000
GAMMA = 0 # 0.999 # Discount factor
EPS_START = 0.9 # Random action choosing probability starts with this value and decays until EPS_END
EPS_END = 0.05 # Random action choosing probability starts at EPS_START and decays until EPS_END
EPS_DECAY = 200 # Decay rate of random action choosing probability, with the passage of episodes and time
TARGET_UPDATE = 10
TARGET_SAVE_CHECKPOINTS = [500, 1000, 2000, 5000, 10000, 15000, 25000, 35000, 45000, 50000]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


n_observations = 6 # 3 for initial state, 3 for goal state
n_actions = 16 # 16 push + 1 grasp

policy_net = pushDQN2(n_observations, n_actions, use_cuda=True).to(device)
target_net = pushDQN2(n_observations, n_actions, use_cuda=True).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(REPLAY_MEMORY_SIZE) # 10000

steps_done = 0

def select_action(state):
    '''Select the next best action 
    state: tensor(shape=(6))
    '''
    global steps_done
    sample = random.uniform(0.0, 1.0) # random.randint(a=0, b=16) 
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0*steps_done / EPS_DECAY)
    steps_done += 1

    if sample>eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def get_reward(prev_state, current_state):
    '''
    prev_state: (x1, y1, theta1, x2, y2, theta2)
    current_state: (x3, y3, theta3, _, _, _)
    '''
    reward = np.exp(-1 * np.sqrt(np.linalg.norm(current_state[0:3] - prev_state[3:6])))
    return reward

def get_reward2(prev_state, current_state):
    '''
    prev_state: (x1, y1, theta1, x2, y2, theta2)
    current_state: (x3, y3, theta3, _, _, _)
    '''
    reward = np.linalg.norm(prev_state[0:2] - prev_state[3:5]) - np.linalg.norm(current_state[0:2] - prev_state[3:5]) # prev distance - current distance
    return reward

def optimize_model(timestep=0, batch_num=0, reward=0):
    if len(memory) < BATCH_SIZE:
        return 
    print("Optimization!")
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # wandb.log({'loss': loss, 'timestep': timestep, 'batch': batch_num})
    wandb.log({'loss': loss, 'reward': reward_np, 'timestep': timestep}) #, 'batch': t})

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


from itertools import count

from Config.constants import MIN_GRASP_THRESHOLDS

is_viz = False

# env = Environment()
env = Environment(gui=False)
num_of_envs = 10
max_num_of_actions = 15
is_viz = False
max_extent_threshold = 1 # Max extent threshold of the target object in pixel units
push_directions = [0, np.pi/8, np.pi/4, 3*np.pi/8, 
                    np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, 
                    np.pi, 9*np.pi/8, 5*np.pi/4, 11*np.pi/8,  
                    3*np.pi/2, 13*np.pi/8, 7*np.pi/4, 15*np.pi/8] # 16 standard directions
# num_episodes = 50 # 10
max_timesteps = 50050
timestep = 0

wandb.config.update({
    'max_timesteps': max_timesteps,
    'batch_size': BATCH_SIZE,
    'optimizer': 'Adam',
    'learning_rate': 'default',
    'replay_memory': REPLAY_MEMORY_SIZE, # 10000
    'n_actions': n_actions,
    'n_observations': n_observations,
    'action_types': 'Only push in 16 different directions'
})

      
widgets = ['Training: ', progressbar.Bar('-')]# progressbar.AnimatedMarker()]
bar = progressbar.ProgressBar(max_value=max_timesteps+1, widgets=widgets) #, widget_kwargs={}).start()
        

while timestep < max_timesteps:
    # Initialize the environment and state
    env.reset()
    testcase1 = TestCase1(env)
    body_ids, success = testcase1.sample_test_case(bottom_obj='random') #'random') # testcase1.create_standard()

    target_pos, target_orn = p.getBasePositionAndOrientation(body_ids[1])
    target_euler = p.getEulerFromQuaternion(target_orn)

    marker_pos, marker_orn = None, None
    goal_suc = False
    while not goal_suc:
        marker_pos, marker_orn = sample_goal(target_pos, target_orn)
        # marker_obj, goal_suc = testcase1.add_marker_obj(marker_pos, marker_orn, half_extents=testcase1.current_target_size/2)
        goal_suc = testcase1.check_target_within_bottom_bounds(marker_pos)

    marker_euler = p.getEulerFromQuaternion(marker_orn)
    cur_target_st = np.array([target_pos[0], target_pos[1], target_euler[2]], dtype=np.float64)
    cur_target_goal = np.array([marker_pos[0], marker_pos[1], marker_euler[2]], dtype=np.float64)
    cur_state = np.hstack((cur_target_st, cur_target_goal))
    state = {
        'cur_state': torch.tensor(cur_state, dtype=torch.float, device=device).unsqueeze(0),
    }
    done = False

    for t in count():
        # Select and perform an action
        timestep += 1
        bar.update(timestep)
        action = select_action(state['cur_state']) # select_action(state['rgb'], state['height_map'])
        color_image, depth_image, _ = env_utils.get_true_heightmap(env)
        if action.item() in range(0, 16): # push action
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            target_mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            push_dir = push_directions[action.item()] # Sample push directions
            push_start, push_end = get_push_start(push_dir, target_mask, body_ids[1])
            env.push(push_start, push_end) # Action performed 
            
            target_pos, target_orn = p.getBasePositionAndOrientation(body_ids[1])
            euler_orn = p.getEulerFromQuaternion(target_orn)

            new_target_st = np.array([target_pos[0], target_pos[1], euler_orn[2]], dtype=np.float)
            new_state = np.hstack((new_target_st, cur_target_goal))
            reward = get_reward(current_state=new_state, prev_state=state['cur_state'].squeeze().cpu().numpy())
        elif action.item()==16:
            print("Invalid Action!!!!!")
            exit()
            
        targetPos, _ = p.getBasePositionAndOrientation(body_ids[1])
        bottomPos, _ = p.getBasePositionAndOrientation(body_ids[0])
        if targetPos[2] < bottomPos[2] + testcase1.current_bottom_size[2]/2 + testcase1.current_target_size[2]/2 - 0.01:
            done = True
        # _, reward, done, _, _ = env.step(action.item())
        reward_np = reward
        reward = torch.tensor([reward], dtype=torch.float, device=device)
        
        if not done:
            target_pos, target_orn = p.getBasePositionAndOrientation(body_ids[1])
            euler_orn = p.getEulerFromQuaternion(target_orn)
            
            new_target_st = np.array([target_pos[0], target_pos[1], euler_orn[2]], dtype=float)
            new_state = np.hstack((new_target_st, cur_target_goal))
            next_state = {
                'cur_state': torch.tensor(new_state, dtype=torch.float, device=device).unsqueeze(0),
            }
        else:
            next_state = {
                'cur_state': None,
            }

        # Store the transition in memory
        memory.push(state['cur_state'], action, next_state['cur_state'], reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        
        optimize_model(timestep=timestep, batch_num=t, reward=reward_np)

    # Update the target network, copying all weights and biases in DQN
        if timestep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # print("Target updated")
            

        # if i_episode % TARGET_SAVE == 0 or i_episode==10:
        if timestep in TARGET_SAVE_CHECKPOINTS:
            print("Saved")
            SAVE_PATH = './V2_next_best_action/models/model_checkpoints/{}.pt'.format(timestep)
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), SAVE_PATH)

        torch.cuda.empty_cache()

        if t>=100:
            done = True
        
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.savefig('durations_count.png')
# plt.show()
run.finish()
