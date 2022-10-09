from collections.abc import Sequence

import actor_critic
import torch


class IsaacGymPolicy(object):

  def __init__(self, checkpoint_path, device):

    num_obs = 48  # Need to find the list of sensors being used for pupper.
    num_critic_obs = 48  # If no priviledge info is specified
    num_actions = 12

    # To match the training config.
    policy_cfg = {
        'activation': 'elu',
        'actor_hidden_dims': [512, 256, 128],
        'critic_hidden_dims': [512, 256, 128],
        'init_noise_std': 1.0
    }
    actor_critic_policy = actor_critic.ActorCritic(num_obs, num_critic_obs,
                                                   num_actions,
                                                   **policy_cfg).to(device)

    loaded_dict = torch.load(checkpoint_path)
    actor_critic_policy.load_state_dict(loaded_dict['model_state_dict'])

    actor_critic_policy.eval()

    self.device = device
    self.policy = actor_critic_policy.act_inference

  def step(self, obs):
    # The following information will be pulled from the pupper training config
    clip_actions = 100
    action_scale = 0.3
    default_dof_pos = torch.tensor([0, 0.5, -1.2] * 4, device=self.device)

    actions = self.policy(obs.detach())

    # Now needs to convert the action to the joint angles.

    actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

    actions_scaled = actions * action_scale
    motor_targets = actions_scaled + default_dof_pos

    return motor_targets


def main() -> None:
  path = '/usr/local/google/home/tingnan/src/pupper/legged_gym/logs/flat_pupper/Oct01_21-12-46_/model_1500.pt'
  device = 'cpu'

  policy = IsaacGymPolicy(path, device)

  obs = torch.tensor([
      0.3135, -0.2561, 0.0493, -0.1321, -0.1175, -0.0096, 0.2240, 0.1411,
      -0.9643, -0.3414, -2.0000, 0.0214, 0.3170, 0.5583, -0.6898, -0.2991,
      1.0475, -1.2112, 0.1258, 0.8773, -0.4002, -0.2655, -0.1646, -0.0513,
      -0.1057, 0.2348, -0.0140, -0.2070, 0.2207, -0.0648, -0.0802, 0.0719,
      0.0080, -0.0778, 0.0720, -0.0351, 1.2710, 3.5572, 4.3091, 2.2097, 0.2297,
      4.3505, 1.1126, 4.6963, -0.4752, 2.6046, 0.9741, 3.3854
  ],
                     device=device)

  motor_angles = policy.step(obs.detach())

  return motor_angles
