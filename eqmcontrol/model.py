import torch
from torch import nn
from loguru import logger

class BicycleModel(nn.Module):
    def __init__(self, drag_coeff=0.05, max_steer_rad=1.0, rear_axle_offset_ratio=0.25, front_axle_offset_ratio=0.2):
        super().__init__()
        self.register_buffer("drag_coeff", torch.tensor(drag_coeff))
        self.register_buffer("max_steer_rad", torch.tensor(max_steer_rad))
        self.register_buffer("rear_axle_offset_ratio", torch.tensor(rear_axle_offset_ratio))  # Rear axle from rear edge
        self.register_buffer("front_axle_offset_ratio", torch.tensor(front_axle_offset_ratio))  # Front axle from front edge

    def dynamics(self, states, controls):
        # States: [x, y, speed, heading, length_m, width_m]
        # Controls: [steering_angle, throttle]
        x, y, speed, heading, length_m, width_m = states.unbind(-1)
        steer, throttle = controls.unbind(-1)
        steer = torch.clamp(steer, -self.max_steer_rad, self.max_steer_rad)

        # Compute rear axle offset and wheelbase per batch element
        rear_axle_offset = self.rear_axle_offset_ratio * length_m  # Center to rear axle
        front_axle_offset = self.front_axle_offset_ratio * length_m  # Center to front axle
        wheelbase_m = length_m - (rear_axle_offset + front_axle_offset)  # Rear to front axle

        # Angular velocity: dheading/dt = (speed / wheelbase) * tan(steer)
        omega = (speed / wheelbase_m) * torch.tan(steer)

        # Velocity of car center
        dx = speed * torch.cos(heading) + omega * rear_axle_offset * torch.sin(heading)
        dy = speed * torch.sin(heading) - omega * rear_axle_offset * torch.cos(heading)

        # Speed derivative
        drag = self.drag_coeff * speed * speed.abs()
        dspeed = throttle - drag

        # Heading derivative
        dheading = omega

        # Length and width derivatives
        dlength_m = torch.zeros_like(length_m)
        dwidth_m = torch.zeros_like(width_m)

        return torch.stack([dx, dy, dspeed, dheading, dlength_m, dwidth_m], dim=-1)

    def step(self, states, controls, dt):
        # RK4 integration
        k1 = self.dynamics(states, controls)
        k2 = self.dynamics(states + 0.5 * dt * k1, controls)
        k3 = self.dynamics(states + 0.5 * dt * k2, controls)
        k4 = self.dynamics(states + dt * k3, controls)
        return states + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

class EqMPolicy(nn.Module):
    def __init__(self, hidden_dim, num_steps, total_time):
        super().__init__()
        self.num_steps = num_steps
        self.total_time = total_time
        self.dt = total_time / num_steps
        input_dim = 7
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
        self.max_accel = 10.0
        self.max_steer = 1.0

    def _relative_pose(self, states, target_trajectories):
        # states: (B, T, 6), target_trajectories: (B, T, 6)
        rel_pos_global = target_trajectories[:, :, :2] - states[:, :, :2]
        rel_speed = target_trajectories[:, :, 2] - states[:, :, 2]
        rel_heading = target_trajectories[:, :, 3] - states[:, :, 3]
        cos_h, sin_h = torch.cos(states[:, :, 3]), torch.sin(states[:, :, 3])
        rel_x = rel_pos_global[:, :, 0] * cos_h + rel_pos_global[:, :, 1] * sin_h
        rel_y = -rel_pos_global[:, :, 0] * sin_h + rel_pos_global[:, :, 1] * cos_h
        rel_heading = torch.atan2(torch.sin(rel_heading), torch.cos(rel_heading))
        rel_theta_trig = torch.stack([torch.cos(rel_heading), torch.sin(rel_heading)], dim=-1)
        return torch.cat([rel_x.unsqueeze(-1), rel_y.unsqueeze(-1), rel_speed.unsqueeze(-1), rel_theta_trig], dim=-1)  # (B, T, 5)

    def forward(self, states, target_trajectories, actions):
        logger.debug(
            "Policy forward – states: {s}, target_trajectories: {g}, actions: {a}",
            s=states.shape, g=target_trajectories.shape, a=actions.shape
        )
        rel_pose = self._relative_pose(states, target_trajectories)
        x = torch.cat([rel_pose, actions], dim=-1)  # (B, T, 8)
        x_flat = x.view(-1, x.shape[-1])

        grad_flat = self.net(x_flat)
        grad = grad_flat.view(states.shape[0], states.shape[1], 2)
        grad = torch.clamp(grad, -torch.tensor([self.max_accel, self.max_steer], device=grad.device),
                           torch.tensor([self.max_accel, self.max_steer], device=grad.device))
        return grad

    def simulate_trajectory(self, model, init_states, target_trajectories, init_actions, steps, step_size):
        actions = init_actions.clone()
        states = init_states.unsqueeze(1)
        for t in range(self.num_steps):
            next_state = model.step(states[:, -1], actions[:, t], dt=self.dt)
            states = torch.cat([states, next_state.unsqueeze(1)], dim=1)
        for _ in range(steps):
            grad = self.forward(states[:, 1:, :], target_trajectories, actions)  # Use states[:, 1:, :] to match target_trajectories
            actions = actions - step_size * grad
            actions = torch.clamp(
                actions,
                -torch.tensor([self.max_accel, self.max_steer], device=actions.device),
                torch.tensor([self.max_accel, self.max_steer], device=actions.device)
            )
            states = init_states.unsqueeze(1)
            for t in range(self.num_steps):
                next_state = model.step(states[:, -1], actions[:, t], dt=self.dt)
                states = torch.cat([states, next_state.unsqueeze(1)], dim=1)
        return actions, states[:, 1:]