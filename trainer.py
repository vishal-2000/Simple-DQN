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
from typing import OrderedDict
from collections import namedtuple, deque

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
import Environments.utils as env_utils
from V1_destination_prediction.Test_cases.tc1 import TestCase1
from create_env import get_push_start, get_max_extent_of_target_from_bottom
from V2_next_best_action.models.model import pushDQN

import wandb

# wandb setup
number = 1
NAME = "model_corrected_big_obj_" + str(number)
ID = 'DQN_next_best_simple_big_objects_corrected' + str(number)
run = wandb.init(project='DQN_next_best_action_simple_case', name = NAME, id = ID)


Transition = namedtuple('Transition', ('state_rgb', 'state_height', 'action', 'next_state_rgb', 'next_state_height', 'reward'))

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

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hyperparameters
BATCH_SIZE = 10
REPLAY_MEMORY_SIZE = 128 
GAMMA = 0.99 # Discount factor 0.999
EPS_START = 0.9 # Random action choosing probability starts with this value and decays until EPS_END
EPS_END = 0.05 # Random action choosing probability starts at EPS_START and decays until EPS_END
EPS_DECAY = 200 # Decay rate of random action choosing probability, with the passage of episodes and time
TARGET_UPDATE = 10 # policy net update rate
# TARGET_SAVE = 100 # net save rate
TARGET_SAVE_CHECKPOINTS = [50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]

# Number of actions
n_actions = 16 # (only push this time)  #17 # 16 push + 1 grasp

policy_net = pushDQN(use_cuda=True).to(device)
target_net = pushDQN(use_cuda=True).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(REPLAY_MEMORY_SIZE) # 10000

steps_done = 0

def select_action(state_rgb, state_height):
    '''Select the next best action 
    state: {
        'rgb': tensor(shape(3*224*224)),
        'height_map': tensor(shape(3*224*224))
    }
    '''
    global steps_done
    sample = random.uniform(0.0, 1.0) # random.randint(a=0, b=16) 
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0*steps_done / EPS_DECAY)
    steps_done += 1

    if sample>eps_threshold:
        with torch.no_grad():
            return policy_net(state_rgb, state_height).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


"""REWARD SPECIFICATION

1. If action=grasp:
        if prev_max_extents>THRESHOLD: # Max extents before grasping
            reward = 1
        else:
            reward = -1 
2. If action=push:
        if cur_max_entents>THRESHOLD: # Max extents after pushing
            reward = 1
        else:
            reward = -1
Bellman equation:
    Q(s, a) = r + gamma*max(Q(s', a'))
"""

def get_reward(action, max_extents, MIN_GRASP_EXTENT_THRESH):
    '''
    '''
    if action=='push':
        # if (max_extents[0] > MAX_EXTENT_THRESH) or (max_extents[1] > MAX_EXTENT_THRESH[1]): # check if object fell on the ground
        #     return -1.0
        if (max_extents[0] > MIN_GRASP_EXTENT_THRESH[0]) or (max_extents[1] > MIN_GRASP_EXTENT_THRESH[1]):
            return 1.5
        else:
            return -0.01 # for fast achievement of goal
    elif action=='grasp':
        # if (max_extents[0] > MAX_EXTENT_THRESH) or (max_extents[1] > MAX_EXTENT_THRESH[1]): # check if object fell on the ground
        #     return -1.0
        if (max_extents[0] > MIN_GRASP_EXTENT_THRESH[0]) or (max_extents[1] > MIN_GRASP_EXTENT_THRESH[1]):
            return 1.0
        else:
            return -1.0 # end the episode if it tries to grasp here
    else:
        return 0.0

def optimize_model(episode=0, batch_num=0):
    print("Replay memory: {}".format(len(memory)))
    if len(memory) < BATCH_SIZE:
        return
    print("Back propagation done ---------------------------------------")
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state_rgb)), device=device, dtype=torch.bool)


    non_final_next_states_rgb = None
    non_final_next_states_height = None
    if torch.sum(non_final_mask)>0:
        non_final_next_states_rgb = torch.cat([s for s in batch.next_state_rgb
                                                    if s is not None])
        non_final_next_states_height = torch.cat([s for s in batch.next_state_height
                                                    if s is not None])

    state_rgb_batch = torch.cat(batch.state_rgb)
    state_height_batch = torch.cat(batch.state_height)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

#    with torch.no_grad():
#        next_state_values = torch.zeros(BATCH_SIZE, device=device)
#        next_state_values[non_final_mask] = policy_net(non_final_next_states_rgb, non_final_next_states_height).max(1)[0].detach()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_rgb_batch, state_height_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if torch.sum(non_final_mask) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states_rgb, non_final_next_states_height).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    print("Batch size: {}-------------------".format(next_state_values.size()))

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    wandb.log({'loss': loss, 'epoch': episode, 'batch': batch_num})

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        if param.grad == None:
            continue
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# import matplotlib

# # set up matplotlib
# is_ipython = 'inline' in matplotlib.get_backend()
# if is_ipython:
#     from IPython import display

# plt.ion()

# episode_durations = []

# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())

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
num_episodes = 5010

wandb.config.update({
    'epochs': num_episodes,
    'batch_size': BATCH_SIZE,
    'optimizer': 'Adam',
    'learning_rate': 'default',
    'replay_memory': REPLAY_MEMORY_SIZE, # 10000
    'n_actions': n_actions,
    'action_types': 'Only push in 16 different directions'
})

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    testcase1 = TestCase1(env)
    body_ids, success = testcase1.sample_test_case(bottom_obj='random') #'random') # testcase1.create_standard()
    color_image, depth_image, _ = env_utils.get_true_heightmap(env)
    depth_image = np.stack((depth_image, )*3, axis=-1)
    # print("Returned body ids: {}, success: {}".format(body_ids, success))
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen
    state = {
        'rgb': torch.tensor(np.array([np.transpose(color_image, (2, 0, 1))]), dtype=torch.float, device=device), # transpose used in order to convert (224, 224, 3) to (3, 224, 224)
        'height_map': torch.tensor(np.array([np.transpose(depth_image, (2, 0, 1))]), dtype=torch.float, device=device) # torch.tensor([np.transpose(depth_image, (2, 0, 1))], device=device) # transpose used in order to convert (224, 224, 3) to (3, 224, 224)
    }
    done = False

    for t in count():
        # Select and perform an action
        action = select_action(state['rgb'], state['height_map'])
        color_image, depth_image, _ = env_utils.get_true_heightmap(env)
        if action.item() in range(0, 16): # push action
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            target_mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            push_dir = push_directions[action.item()] # Sample push directions
            push_start, push_end = get_push_start(push_dir, target_mask, body_ids[1])
            env.push(push_start, push_end) # Action performed 

            color_image, depth_image, _ = env_utils.get_true_heightmap(env) # Evaluating the new state for calculating the reward
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            target_mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            bottom_mask = cv2.inRange(temp, orange_lower, orange_upper)
            depth_image = np.stack((depth_image, )*3, axis=-1)
            target_mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            max_extents = get_max_extent_of_target_from_bottom(target_mask=target_mask, bottom_mask=bottom_mask, 
                                        bottom_obj_body_id=body_ids[0], 
                                        current_bottom_obj_size=testcase1.current_bottom_size, 
                                        is_viz=False)
            
            reward = get_reward(action='push', max_extents=max_extents, MIN_GRASP_EXTENT_THRESH=MIN_GRASP_THRESHOLDS) # get_reward(action, max_extents, MAX_EXTENT_THRESH, MIN_GRASP_EXTENT_THRESH)
            # belman_update_val = get_belman_update_value()
        elif action.item()==16:
            # Check if the state is graspable and reward the agent
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            target_mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            bottom_mask = cv2.inRange(temp, orange_lower, orange_upper)
            max_extents = get_max_extent_of_target_from_bottom(target_mask=target_mask, bottom_mask=bottom_mask, 
                                        bottom_obj_body_id=body_ids[0], 
                                        current_bottom_obj_size=testcase1.current_bottom_size, 
                                        is_viz=False)
            
            reward = get_reward(action='grasp', max_extents=max_extents, MIN_GRASP_EXTENT_THRESH=MIN_GRASP_THRESHOLDS)
            if reward==1:
                done = True
            # done = True
        targetPos, _ = p.getBasePositionAndOrientation(body_ids[1])
        bottomPos, _ = p.getBasePositionAndOrientation(body_ids[0])
        if targetPos[2] < bottomPos[2] + testcase1.current_bottom_size[2]/2 + testcase1.current_target_size[2]/2 - 0.01:
            reward = -0.75
            done = True
        # _, reward, done, _, _ = env.step(action.item())
        reward = torch.tensor([reward], dtype=torch.float, device=device)
        if reward == 1:
            done=True

        # Observe new state
        # last_screen = current_screen
        # current_screen = get_screen()
        if not done:
            next_state = {
                'rgb': torch.tensor(np.array([np.transpose(color_image, (2, 0, 1))]), dtype=torch.float, device=device), # transpose used in order to convert (224, 224, 3) to (3, 224, 224)
                'height_map': torch.tensor(np.array([np.transpose(depth_image, (2, 0, 1))]), dtype=torch.float, device=device) # transpose used in order to convert (224, 224, 3) to (3, 224, 224)
            }
        else:
            next_state = {
                'rgb': None,
                'height_map': None
            }

        # Store the transition in memory
        memory.push(state['rgb'], state['height_map'], action, next_state['rgb'], next_state['height_map'], reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model(episode=i_episode, batch_num=t)
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

        if t>=10:
            done = True

        torch.cuda.empty_cache()
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        

    # if i_episode % TARGET_SAVE == 0 or i_episode==10:
    if i_episode in TARGET_SAVE_CHECKPOINTS:
        SAVE_PATH = './V2_next_best_action/models/model_checkpoints/{}.pt'.format(i_episode)
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(policy_net.state_dict(), SAVE_PATH)

print('Complete')
# env.render()
# env.close()
plt.ioff()
plt.savefig('durations_count.png')
# plt.show()
run.finish()
