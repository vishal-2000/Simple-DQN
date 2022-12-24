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



def get_max_extent_of_target_from_bottom(target_mask, bottom_mask, bottom_obj_body_id, current_bottom_obj_size, is_viz=False):
    '''Calculates in pixels, the max outward extent of the target object from the bottom object
    '''
    bottomPos, bottomOrn = p.getBasePositionAndOrientation(bottom_obj_body_id)
    euler_orn = p.getEulerFromQuaternion(bottomOrn)
    # print(bottomPos, bottomOrn, euler_orn)
    for i in range(0, 224):
        for j in range(0, 224):
            if j==112:
                continue
            tan_val = np.tan(euler_orn[2])
            val1 = np.sqrt(((j-112) - (tan_val)*(i-112))**2)
            val2 = np.sqrt(((j-112)*(tan_val) + (i-112))**2)
            if val1 <= 1:
                bottom_mask[i, j] = 255
            if val2 <= 1:
                bottom_mask[i, j] = 255    
    if is_viz:
        cv2.imshow("bottom mask with it's principle axes", bottom_mask) # Bottom object is being masked properly - checked
        cv2.waitKey(0)

    # Find the corners for the target object
    dst = cv2.cornerHarris(target_mask, 2, 9, 0.04) # Corner detection done perfectly!!
    temp = np.zeros(shape=(224, 224))
    temp[dst>0.01*dst.max()] = 255
    if is_viz==True:
        cv2.imshow("Corners in the target", temp) # Corner detection done perfectly!! - Checked
        cv2.waitKey(0)
    corners = np.where(temp==255)

    ######## Start here
        ## Code up the max extent calculation and return the extent ##
    orn_vec = np.array([np.cos(euler_orn[2]), np.sin(euler_orn[2])])
    orn_perpendicular = np.array([-np.sin(euler_orn[2]), np.cos(euler_orn[2])])
    max_extents = np.ones(shape=(2,))*(-10.0)
    for corner in zip(corners[0], corners[1]):
        corner_in_real = np.array([
            corner[0]*PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            corner[1]*PIXEL_SIZE + WORKSPACE_LIMITS[1][0]
        ])
        corner_wrt_bottom_pos = corner_in_real - bottomPos[0:2]
        ext1 = abs(np.dot(orn_vec, corner_wrt_bottom_pos)) - current_bottom_obj_size[0]/2 # length component
        ext2 = abs(np.dot(orn_perpendicular, corner_wrt_bottom_pos)) - current_bottom_obj_size[1]/2 # breadth/width component
        if ext1 >= max_extents[0]:
            max_extents[0] = ext1
        if ext2 >= max_extents[1]:
            max_extents[1] = ext2
    # current_bottom_obj_size[0]
    return max_extents
        # max_extents[0] = np.max(np.dot(orn_vec, ))
    ######## Ends here
    # return 0

def get_push_start(push_dir, target_mask, target_obj_id, is_viz=False):
    '''Get the start point for push
    '''
    dst = cv2.cornerHarris(target_mask, 2, 9, 0.04) # Corner detection done perfectly!!
    temp = np.zeros(shape=(224, 224))
    temp[dst>0.01*dst.max()] = 255
    if is_viz==True:
        cv2.imshow("Corners in the target", temp) # Corner detection done perfectly!! - Checked
        cv2.waitKey(0)
    corners = np.where(temp==255)
    # print(corners)

    # Find the right corner - corner whose dot product gives the max value with negative sign, that is the farthest corner from the centoid for the
    # given orientation
    targetPos, targetOrn = p.getBasePositionAndOrientation(target_obj_id)
    euler_orn = p.getEulerFromQuaternion(targetOrn)
    orn_vec = np.array([np.cos(euler_orn[2]), np.sin(euler_orn[2])])
    normal_to_push_vec = np.array([-np.sin(push_dir), np.cos(push_dir)])
    push_vec = np.array([np.cos(push_dir), np.sin(push_dir)])
    max_neg_val = 0
    desired_corner = np.array(corners[0][0], corners[0][1])
    desired_push_start = np.array([0, 0, 0.1])
    for corner in zip(corners[0], corners[1]):
        corner_in_real = np.array([
            corner[0]*PIXEL_SIZE + WORKSPACE_LIMITS[0][0],
            corner[1]*PIXEL_SIZE + WORKSPACE_LIMITS[1][0]
        ])
        # print("Corner: {}, target pos: {}".format(corner_in_real, targetPos))
        corner_in_target_fr = corner_in_real - targetPos[0:2]
        val = np.dot(corner_in_target_fr, push_vec)# orn_vec)
        if val<=max_neg_val:
            max_neg_val = val
            desired_corner = corner
            desired_push_start[0] = corner_in_real[0] - (GRIPPER_PUSH_RADIUS+0.05)*push_vec[0] #orn_vec[0]
            desired_push_start[1] = corner_in_real[1] - (GRIPPER_PUSH_RADIUS+0.05)*push_vec[1] #orn_vec[1]
        # print(corner)
    # print(corner)

    desired_push_end = copy.deepcopy(desired_push_start)
    desired_push_end[:2] = desired_push_start[:2] + PUSH_DISTANCE*push_vec # orn_vec
    desired_push_start[2]=desired_push_end[2]=targetPos[2]-0.01
    return desired_push_start, desired_push_end


if __name__=='__main__':
    random.seed(12345)
    torch.manual_seed(12345)

    env = Environment(gui=True)
    num_of_envs = 10
    max_num_of_actions = 10
    is_viz = False
    max_extent_threshold = 1 # Max extent threshold of the target object in pixel units
    push_directions = [0, np.pi/8, np.pi/4, 3*np.pi/8, 
                       np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, 
                       np.pi, 9*np.pi/8, 5*np.pi/4, 11*np.pi/8,  
                       3*np.pi/2, 13*np.pi/8, 7*np.pi/4, 15*np.pi/8] # 16 standard directions

    # Test samples loop - creates "num_of_envs" samples of environment
    while num_of_envs>0:
        # Create a new test sample
        env.reset()
        testcase1 = TestCase1(env)
        body_ids, success = testcase1.sample_test_case(bottom_obj='random') #'random') # testcase1.create_standard()
        print("Returned body ids: {}, success: {}".format(body_ids, success))

        # print("Positions of the bodies:")
        # print("bottom obj: {}".format(p.getBasePositionAndOrientation(body_ids[0])))
        # print("target obj: {}".format(p.getBasePositionAndOrientation(body_ids[1])))
        # sleep(10)
        # exit()

        # Main action loop starts
        n_actions = 0
        while n_actions < max_num_of_actions:
            # Obtain the height map and the orthographic image for the given scene
            color_image, depth_image, _ = env_utils.get_true_heightmap(env)
            if is_viz:
                cv2.imshow("Color image", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                plt.figure(1)
                plt.imshow(depth_image)
                plt.title("Depth height map")
                plt.show()
                plt.close(1)
            n_actions += 1

            # 2. Create a mask to track the target object (blue colored object is the target)
            temp = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            target_mask = cv2.inRange(temp, TARGET_LOWER, TARGET_UPPER)
            bottom_mask = cv2.inRange(temp, orange_lower, orange_upper) # Shape: 224*224, Range: (0<black>, 255<white>)
            if is_viz:
                cv2.imshow("target mask", target_mask) # Target is being masked properly - checked
                cv2.waitKey(0)
            if is_viz:
                cv2.imshow("bottom mask", bottom_mask) # Bottom object is being masked properly - checked
                cv2.waitKey(0)

            max_extents = get_max_extent_of_target_from_bottom(target_mask, bottom_mask, body_ids[0], testcase1.current_bottom_size)
            print('========================================================================================================')
            print(max_extents)
            print(MIN_GRASP_THRESHOLDS)
            if (max_extents[0] >= MIN_GRASP_THRESHOLDS[0]) or (max_extents[1] >= MIN_GRASP_THRESHOLDS[1]):
                print("Object can be successfully grapsed now!! ------------------------------------")
            else:
                print("Object can't be grasped yet - needs to be pushed ---------------------------------------------")
            print('========================================================================================================')
            sleep(2)
            # 3. Push primitives
            push_dir = push_directions[13] # Sample push directions
            push_start, push_end = get_push_start(push_dir, target_mask, body_ids[1])
            # push_start[2] = push_end[2] = 
            env.push(push_start, push_end)

            


            

            
# filename = 'chessboard.png'
# img = cv2.imread(filename)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
            # print(np.sum(mask))
            # sleep(10)
            # exit()
        
        # for i in range(7):
        #     env.step()
        #     sleep(0.5)
        #     env.reset()
        #     body_ids, success = testcase1.sample_test_case(bottom_obj='random')

            