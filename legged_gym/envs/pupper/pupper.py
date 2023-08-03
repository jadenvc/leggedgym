# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .pupper_config import PupperFlatCfg

# NUM_ACTUATORS = 4
# NUM_PHYSICAL_JOINTS = 6

# MIN_LEG_POS = -0.35
# MAX_LEG_POS = -0.15
# MAX_WHEEL_VEL = 30.0
# ACTION_MIN = [MIN_LEG_POS, -MAX_WHEEL_VEL, MIN_LEG_POS, -MAX_WHEEL_VEL]
# ACTION_MAX = [MAX_LEG_POS, MAX_WHEEL_VEL, MAX_LEG_POS, MAX_WHEEL_VEL]

# DEFAULT_DOF_POS = torch.tensor([-0.25, 0.0, -0.25, 0.0], dtype=torch.float, device="cuda", requires_grad=False)

# ACTION_MIN = torch.tensor(ACTION_MIN, dtype=torch.float, device="cuda", requires_grad=False)
# ACTION_MAX = torch.tensor(ACTION_MAX, dtype=torch.float, device="cuda", requires_grad=False)


RESET_PROJECTED_GRAVITY_Z = 0#np.cos(1.04) # Pitch/roll angle to trigger resets

# PITCH_OFFSET_RANGE = [0.0, 0.0] #[-0.05, 0.05]

class Pupper(LeggedRobot):
    cfg : PupperFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.motor_temperatures = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    #     self.position_joints = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    #     self.velocity_joints = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

    #     # self.pitch_offsets = torch_rand_float(PITCH_OFFSET_RANGE[0], PITCH_OFFSET_RANGE[1], (self.num_envs, 1), device=self.device)#.squeeze(1)

    #     self.action_scales = torch.tensor([self.cfg.control.action_scale, self.cfg.control.velocity_action_scale, self.cfg.control.action_scale, self.cfg.control.velocity_action_scale], dtype=torch.float, device="cuda", requires_grad=False)

    #     self.command_lower_bound = torch.tensor([self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_y"][0], self.command_ranges["ang_vel_yaw"][0]], dtype=torch.float, device="cuda", requires_grad=False)
    #     self.command_upper_bound = torch.tensor([self.command_ranges["lin_vel_x"][1], self.command_ranges["lin_vel_y"][1], self.command_ranges["ang_vel_yaw"][1]], dtype=torch.float, device="cuda", requires_grad=False)
    #     # self.command_lower_bound = torch.tensor([self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_y"][0]], dtype=torch.float, device="cuda", requires_grad=False)
    #     # self.command_upper_bound = torch.tensor([self.command_ranges["lin_vel_x"][1], self.command_ranges["lin_vel_y"][1]], dtype=torch.float, device="cuda", requires_grad=False)

    #     # self.joint_friction_strengths = torch_rand_float(0.9, 1.1, (self.num_envs, self.num_dof), device=self.device)
    #     # self.actuator_strengths = torch_rand_float(0.9, 1.1, (self.num_envs, self.num_dof), device=self.device)
    #     self.joint_friction_strengths = torch_rand_float(1.0, 1.0, (self.num_envs, self.num_dof), device=self.device)
    #     self.actuator_strengths = torch_rand_float(1.0, 1.0, (self.num_envs, self.num_dof), device=self.device)

    #     self.delayed_actions = torch.zeros((self.num_envs, self.num_dof, self.cfg.control.delay_range[1] + 1), dtype=torch.float, device=self.device, requires_grad=False)
    #     self.delayed_action_indices = torch.randint(self.cfg.control.delay_range[0], self.cfg.control.delay_range[1] + 1, (self.num_envs, 1, 1), device="cuda", requires_grad=False)
    #     self.delayed_action_indices = self.delayed_action_indices.expand(self.num_envs, self.num_dof, 1)

    #     for i in range(self.num_dofs):
    #         name = self.dof_names[i]
    #         for dof_name in self.cfg.control.joint_control_types.keys():
    #             if dof_name in name:
    #                 if self.cfg.control.joint_control_types[dof_name] == "P":
    #                     self.position_joints[i] = 1.0
    #                 elif self.cfg.control.joint_control_types[dof_name] == "V":
    #                     self.velocity_joints[i] = 1.0

    # def check_termination(self):
    #     """ Check if environments need to be reset
    #     """
    #     self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    #     self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
    #     self.reset_buf |= self.time_out_buf
    #     self.reset_buf |= torch.abs(self.projected_gravity[:, 2]) < RESET_PROJECTED_GRAVITY_Z

    # def compute_observations(self):
    #     """ Computes observations
    #     """
    #     actions_clipped = torch.clamp(self.actions, (ACTION_MIN - DEFAULT_DOF_POS) / self.action_scales, (ACTION_MAX - DEFAULT_DOF_POS) / self.action_scales)

    #     self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
    #                                 self.projected_gravity,
    #                                 self.commands[:, :3] * self.commands_scale,
    #                                 self.position_joints * (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                 self.dof_vel * self.obs_scales.dof_vel,
    #                                 actions_clipped
    #                                 ),dim=-1)
    #     # add perceptive inputs if not blind
    #     if self.cfg.terrain.measure_heights:
    #         heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
    #         self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
    #     # add noise if needed
    #     if self.add_noise:
    #         self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     if self.cfg.commands.heading_command:
    #         self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     else:
    #         self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, :] *= (torch.norm(self.commands[env_ids, :], dim=1) > self.cfg.commands.deadband).unsqueeze(1)

    #     # self.commands[env_ids, 0] = torch.where(self.commands[env_ids, 0] < 0, torch.zeros_like(self.commands[env_ids, 0]), self.command_ranges["lin_vel_x"][1] * torch.ones_like(self.commands[env_ids, 0]))

    #     # # Remapping small commands to zero:
    #     # upper_bound = self.command_upper_bound
    #     # lower_bound = self.command_lower_bound
    #     # deadband = self.cfg.commands.deadband

    #     # # Compute the absolute values
    #     # abs_commands = torch.abs(self.commands[env_ids, :])

    #     # # Create masks for values within the deadband
    #     # mask_within_deadband = (abs_commands <= deadband)

    #     # # For values within deadband, simply set them to 0
    #     # self.commands[env_ids, :] = torch.where(mask_within_deadband, torch.zeros_like(self.commands[env_ids, :]), self.commands[env_ids, :])

    #     # # Create masks for values outside the deadband
    #     # mask_outside_deadband = ~mask_within_deadband

    #     # # Calculate scale factors for remapping
    #     # upper_scale = (upper_bound - deadband) / (upper_bound - deadband)
    #     # lower_scale = (lower_bound + deadband) / (lower_bound + deadband)

    #     # # Create masks for values greater than deadband and less than -deadband
    #     # mask_greater_deadband = self.commands[env_ids, :] > deadband
    #     # mask_less_neg_deadband = self.commands[env_ids, :] < -deadband

    #     # # Remap the values
    #     # self.commands[env_ids, :] = torch.where(mask_greater_deadband, (self.commands[env_ids, :] - deadband) * upper_scale, self.commands[env_ids, :])
    #     # self.commands[env_ids, :] = torch.where(mask_less_neg_deadband, (self.commands[env_ids, :] + deadband) * lower_scale, self.commands[env_ids, :])

    def _process_rigid_body_props(self, props, env_id):
            if self.cfg.domain_rand.randomize_base_mass:
                rng_mass = self.cfg.domain_rand.added_mass_range
                rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
                props[0].mass += rand_mass
            else:
                rand_mass = np.zeros(1)
            if self.cfg.domain_rand.randomize_base_com:
                rng_com_x = self.cfg.domain_rand.added_com_range_x
                rng_com_y = self.cfg.domain_rand.added_com_range_y
                rng_com_z = self.cfg.domain_rand.added_com_range_z
                rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
                props[0].com += gymapi.Vec3(*rand_com)
            else:
                rand_com = np.zeros(3)
            mass_params = np.concatenate([rand_mass, rand_com])
            return props

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)

        # Calculate thermal losses
        currents = torques / self.cfg.control.motor_torque_constant
        thermal_losses = currents ** 2 * self.cfg.control.motor_electrical_resistance * self.cfg.sim.dt # Losses in Joules

        # Add thermal losses to motor temperatures
        self.motor_temperatures += thermal_losses / (self.cfg.control.motor_specific_heat * self.cfg.control.motor_mass)

        # Apply thermal cooling
        self.motor_temperatures -= self.cfg.control.motor_thermal_conductivity * (self.motor_temperatures - self.cfg.control.motor_ambient_temperature) * self.cfg.sim.dt
        return torques

    #     # actions_clipped = actions
    #     actions_clipped = torch.clamp(actions, (ACTION_MIN - DEFAULT_DOF_POS) / self.action_scales, (ACTION_MAX - DEFAULT_DOF_POS) / self.action_scales)

    #     pos_actions_scaled = self.position_joints * actions_clipped * self.cfg.control.action_scale
    #     # actions_scaled = actions_scaled @ self.actuation_matrix
    #     velocity_actions_scaled = self.velocity_joints * actions_clipped * self.cfg.control.velocity_action_scale

    #     actions_scaled = pos_actions_scaled + velocity_actions_scaled

    #     current_delayed_actions = torch.gather(self.delayed_actions, -1, self.delayed_action_indices).squeeze(-1)
    #     self.delayed_actions[:,:,1:] = self.delayed_actions[:,:,:-1]
    #     self.delayed_actions[:,:,0] = actions_scaled

    #     # print(f'actions_scaled: {actions_scaled.size()}, default dof: {self.default_dof_pos.size()}, dgains: {self.d_gains.size()}, dof vel: {self.dof_vel.size()}')
    #     actuator_position_torques = self.p_gains*(current_delayed_actions + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
    #     actuator_position_torques = self.position_joints * actuator_position_torques

    #     # velocity_torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
    #     actuator_velocity_torques = self.d_gains*(current_delayed_actions - self.dof_vel)
    #     actuator_velocity_torques = self.velocity_joints * actuator_velocity_torques

    #     actuator_torques = actuator_position_torques + actuator_velocity_torques
    #     actuator_torques = torch.clip(actuator_torques, -self.torque_limits, self.torque_limits)
    #     actuator_torques *= self.actuator_strengths
    #     # actuator_torques -= self.cfg.control.joint_friction * self.dof_vel
    #     actuator_torques -= self.cfg.control.joint_friction * self.joint_friction_strengths * self.dof_vel

    #     return actuator_torques