import torch

def state_error(states, trajectories):
    # Position error
    pos_error = torch.norm(states[:, :, :2] - trajectories[:, :, :2], dim=-1).mean(dim=1) / 100.0
    # Speed error
    speed_error = torch.abs(states[:, :, 2] - trajectories[:, :, 2]).mean(dim=1)
    # Heading error (angular difference using cosine-sine method)
    theta = states[:, :, 3]
    theta_target = trajectories[:, :, 3]
    cos_diff_theta = torch.cos(theta) * torch.cos(theta_target) + torch.sin(theta) * torch.sin(theta_target)
    sin_diff_theta = torch.cos(theta) * torch.sin(theta_target) - torch.sin(theta) * torch.cos(theta_target)
    heading_loss = ((1 - cos_diff_theta).pow(2) + sin_diff_theta.pow(2)).mean(dim=1)
    # Combine all errors with weights
    return pos_error + 0.5 * heading_loss + speed_error
