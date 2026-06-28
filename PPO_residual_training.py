
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import (
    build_correlated_noise_covariance,
    build_sensor_matrix,
    create_plots_and_gifs,
    grid_measure,
    importance_filter,
    initialize_gp,
    kalman_update,
    noise_model,
    sample_correlated_sensor_noise,
)
from evalmetrics import compute_task_completion, compute_coverage_efficiency, compute_reconstruction_rmse, compute_rmse_time_metrics
import torch
import time
from CMAES_classic_singlemap import compute_fov, dynamics_3d, waypoint_3d

"""
gymnasium import
"""

import gymnasium as gym
from gymnasium import spaces 
from stable_baselines3 import PPO 
from stable_baselines3.common.env_checker import check_env 





SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = SCRIPT_DIR / "Diffusion"
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

from sample_3d_sparse_trans_diffusion import diffusion, load_model, sample_sparse

"""
def one_training_trial(): 
    initial_conditions = varmap, mean_map, (cx, cy), (vx, vy)
    initial_traj = sample_diffusion(varmap, mean_map, (cx, cy), (vx, vy))
    observation = initial_conditions, initial_traj
    residual_changes = PPO_policy.sample(observation)
    residual_changes = max_change * np.tanh(residual_changes)
    final_traj = initial_traj + residual_changes 
    final_conditions = simulate(final_traj)
    reward = reward_fn(
    initial_conditions, final_conditions, final_traj, residual_changes)

    PPO_store(observation, residual_changes, reward)

def_train(): 
    for i in range(number_of_trials): 
        one_training_trial()

    improve_PPO()


def main(): 

    

    residual_changes = PPO_network(initial_traj, initial_conditions) #PPO network can't be
      based on initial_reward, because we don't have that in the actual rollout, i think? 
    
    residual_changes = tanh() * residual_changes ##keeping the changes small

    final_traj = initial_traj + residual_changes 

    final_conditions = simulate(final_traj)

    reward = reward_fn(
    residual_changes, 
    initial traj,
    final_traj,
    initial_conditions, 
    final_conditions)

    improve_PPO(reward)


"""


'''
Map, planner, dynamics settings
'''

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_MODULE_CANDIDATES = (
    SCRIPT_DIR / "threeDSparseTransDiffusion.py",
)
DEFAULT_CHECKPOINT = (
    SCRIPT_DIR
    / "checkpoints"
    / "sparse_trans_waypoints_epoch_1900_multimodal_3d.pth"
)
CONDITON_INDEX = 1000
TRUTH_INDEX = 1000
SEED = None
CLIP = True
NUM_STEPS = None





def one_training_trial(CONDITION_INDEX): 
    model = load_model(DEFAULT_CHECKPOINT, diffusion.device)
    device = next(model.parameters()).device

    meanvarmarker_map = diffusion.meanvarmarkermaps[CONDITON_INDEX:CONDITON_INDEX+1].to(device)
    current_position = diffusion.conditions[
        CONDITION_INDEX : CONDITION_INDEX + 1
    ].to(device)
    initial_heading_velocity = diffusion.initial_heading_velocities[
        CONDITION_INDEX : CONDITION_INDEX + 1
    ].to(device)

    initial_conditions = [meanvarmarker_map, current_position, initial_heading_velocity]

    diffusion_sample = sample_sparse(
        model, 
        condition_index = CONDITON_INDEX, 
        seed = SEED, 
        num_steps = NUM_STEPS, 
        clip_x0 = CLIP)
    
    PPO_observes = [initial_conditions, diffusion_sample]




    

    







