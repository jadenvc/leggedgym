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

class PupperFlatCfg( LeggedRobotCfg ):
    # def _process_rigid_body_props(self, props, env_id):
    #         if self.cfg.domain_rand.randomize_base_mass:
    #             rng_mass = self.cfg.domain_rand.added_mass_range
    #             rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
    #             props[0].mass += rand_mass
    #         else:
    #             rand_mass = np.zeros(1)
    #         if self.cfg.domain_rand.randomize_gripper_mass:
    #             gripper_rng_mass = self.cfg.domain_rand.gripper_added_mass_range
    #             gripper_rand_mass = np.random.uniform(gripper_rng_mass[0], gripper_rng_mass[1], size=(1, ))
    #             props[self.gripper_idx].mass += gripper_rand_mass
    #         else:
    #             gripper_rand_mass = np.zeros(1)
    #         if self.cfg.domain_rand.randomize_base_com:
    #             rng_com_x = self.cfg.domain_rand.added_com_range_x
    #             rng_com_y = self.cfg.domain_rand.added_com_range_y
    #             rng_com_z = self.cfg.domain_rand.added_com_range_z
    #             rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
    #             props[0].com += gymapi.Vec3(*rand_com)
    #         else:
    #             rand_com = np.zeros(3)
    #         mass_params = np.concatenate([rand_mass, rand_com, gripper_rand_mass])
    #         return props


    class env( LeggedRobotCfg.env ):
        num_observations = 31# + 15
  
    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'trimesh'
        # curriculum = True
        horizontal_scale = 0.05 # [m]
        vertical_scale = 0.0025 # [m]
        # # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [1.0, 0.0, 0, 0, 0.0]
        terrain_length = 8.
        terrain_width = 8.
        mesh_type = 'plane'
        measure_heights = False
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.22] # x,y,z [m]
        rot = [0, 0, 0.7071068, 0.7071068]
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'leg2_leftFrontLegMotor': 0.2,   # [rad]
        #     'leg4_leftRearLegMotor': 0.2,   # [rad]
        #     'leg1_rightFrontLegMotor': -0.2 ,  # [rad]
        #     'leg3_rightRearLegMotor': -0.2,   # [rad]

        #     'leftFrontUpperLegMotor': 0.5,     # [rad]
        #     'leftRearUpperLegMotor': 0.5,   # [rad]
        #     'rightFrontUpperLegMotor': 0.5,     # [rad]
        #     'rightRearUpperLegMotor': 0.5,   # [rad]

        #     'leftFrontLowerLegMotor': -1.2,   # [rad]
        #     'leftRearLowerLegMotor': -1.2,    # [rad]
        #     'rightFrontLowerLegMotor': -1.2,  # [rad]
        #     'rightRearLowerLegMotor': -1.2,    # [rad]
        # }

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'leg2_leftFrontLegMotor': 0.2,   # [rad]
            'leg4_leftRearLegMotor': 0.2,   # [rad]
            'leg1_rightFrontLegMotor': -0.2 ,  # [rad]
            'leg3_rightRearLegMotor': -0.2,   # [rad]

            'leftFrontUpperLegMotor': 0.0,     # [rad]
            'leftRearUpperLegMotor': 0.0,   # [rad]
            'rightFrontUpperLegMotor': 0.0,     # [rad]
            'rightRearUpperLegMotor': 0.0,   # [rad]

            'leftFrontLowerLegMotor': -1.2,   # [rad]
            'leftRearLowerLegMotor': -1.2,    # [rad]
            'rightFrontLowerLegMotor': -1.2,  # [rad]
            'rightRearLowerLegMotor': -1.2,    # [rad]
        }
        
    class sim( LeggedRobotCfg.sim ):
        dt =  0.002
        substeps = 1

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'LegMotor': 4.}  # [N*m/rad]
        damping = {'LegMotor': 0.2}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.3
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 15

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/pupper/pupper_v2a.urdf'
        name = "pupper"
        foot_name = "Toe"
        collapse_fixed_joints = False
        penalize_contacts_on = ["UpperLeg"]
        terminate_after_contacts_on = ["chassis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.85
        base_height_target = 0.4
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.00085

            dof_pos_limits = 0.0 # penalize movement outside allowed bounds
            action_magnitude = 0.0 # not sure what it is
            dof_acc = -8e-7 # penalize high accelerations

            action_rate = 0.0
            lin_vel_z = 0#-1 #-2
            ang_vel_xy = 0#-0.03
            tracking_lin_vel = 0#2.0 # main important
            feet_air_time = 0#0.035 # maybe i should care more about this...
            tracking_ang_vel = 0#1.2
            orientation = 0.0 # why is this so high, penalize joint velocities and accelerations
            base_height = 2.0 # looks bad
            # torques = -0.0002
            # dof_pos_limits = -10.0
            # action_magnitude = -0.035
            # dof_acc = -3.5e-7
            # action_rate = -0.07
            # tracking_lin_vel = 2.0
            # feet_air_time = 0.1
            # tracking_ang_vel = 1.2
            # orientation = -15.0
            # base_height = -15.0
            # feet_clearance = 4.0
            termination = -0.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            lin_vel_z = 0
            ang_vel_xy = 0
            orientation = -0.
            torques = -0.0002
            dof_vel = -0.
            dof_acc = -0
            # base_height = -0. 
            # feet_air_time =  1.0
            collision = 0.
            feet_stumble = -0.0 
            action_rate = -0.0
            # stand_still = -5.
            
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
            lin_vel_y = [-0.8, -0.8]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
            #lin_vel_x = [0, 0]
            #lin_vel_y = [-1.5, -1.5]
            #ang_vel_yaw = [-0,-0]

            
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.7, 1.25]
        randomize_base_mass = True
        added_mass_range = [-0.3, 0.3]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.0
        stiffness_delta_range = [-2.0, 2.0]
        damping_delta_range = [-0.05, 0.05]
        randomize_base_com = True
        added_com_range_x = [-0.01, 0.01]
        added_com_range_y = [-0.01, 0.01]
        added_com_range_z = [-0.01, 0.01]

class PupperFlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.005
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'init'
        experiment_name = 'flat_pupper'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

  
