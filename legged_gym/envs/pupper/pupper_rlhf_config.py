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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.pupper.pupper_config import PupperFlatCfg, PupperFlatCfgPPO

class PupperRLHFCfg( PupperFlatCfg ):
        
    class init_state( PupperFlatCfg.init_state ):
        pos = [0.0, 0.0, 0.22] # x,y,z [m]
        rot = [0, 0, 0.7071068, 0.7071068]
        pos = [0.0, 0.0, 0.056] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
        'leg2_leftFrontLegMotor': 0.0,   # [rad]
        'leg4_leftRearLegMotor': 0.0,   # [rad]
        'leg1_rightFrontLegMotor': -0.0 ,  # [rad]
        'leg3_rightRearLegMotor': -0.0,   # [rad]

        'leftFrontUpperLegMotor': 0.0,     # [rad]
        'leftRearUpperLegMotor': 0.0,   # [rad]
        'rightFrontUpperLegMotor': 0.0,     # [rad]
        'rightRearUpperLegMotor': 0.0,   # [rad]

        'leftFrontLowerLegMotor': -1.57,   # [rad]
        'leftRearLowerLegMotor': -1.57,    # [rad]
        'rightFrontLowerLegMotor': -1.57,  # [rad]
        'rightRearLowerLegMotor': -1.57,    # [rad]
    }

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pupper/pupper_v2a.urdf'
        name = "pupper"
        foot_name = "Toe"
        collapse_fixed_joints = False
        penalize_contacts_on = ["UpperLeg"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.85
        base_height_target = 0.11
        forward_velocity_clip = 1.0
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.2
            tracking_lin_vel = 2.0
            forward_velocity = 0.0
            termination = -0.0
            tracking_ang_vel = 1.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = -5.
            dof_vel = -0.
            dof_acc = -0.0
            base_height = -0. 
            feet_air_time =  0.1
            collision = -1.0
            feet_stumble = -0.0 
            action_rate = 0.0
            stand_still = -0.

    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        curriculum = False
        max_curriculum = 2.0
        class ranges:
            # lin_vel_x = [-0.6, 0.6] # min max [m/s]
            # lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            # ang_vel_yaw = [-1, 1]    # min max [rad/s]
            # heading = [-3.14, 3.14]

            lin_vel_x = [0.0, 0.0] # min max [m/s]
            lin_vel_y = [-0.4, -0.4]   # min max [m/s]
            ang_vel_yaw = [0.0, 0.0]    # min max [rad/s]
            heading = [1.57, 1.57]
            
            #lin_vel_x = [0, 0]
            #lin_vel_y = [-1.5, -1.5]
            #ang_vel_yaw = [-0,-0]

            
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        # friction_range = [0.7, 1.25]
        friction_range = [0.0, 0.0]
        randomize_base_mass = True
        # added_mass_range = [-0.3, 0.3]
        added_mass_range = [0.0, 0.0]
        push_robots = False
        push_interval_s = 8
        max_push_vel_xy = 1.0
        # stiffness_delta_range = [-2.0, 2.0]
        # damping_delta_range = [-0.05, 0.05]
        stiffness_delta_range = [-0.0, 0.0]
        damping_delta_range = [0.0, 0.0]
        randomize_base_com = True
        # added_com_range_x = [-0.01, 0.01]
        # added_com_range_y = [-0.01, 0.01]
        # added_com_range_z = [-0.01, 0.01]
        added_com_range_x = [0.0, 0.0]
        added_com_range_y = [0.0, 0.0]
        added_com_range_z = [0.0, 0.0]

class PupperRLHFCfgPPO( PupperFlatCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'init'
        experiment_name = 'pupper_rlhf'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid