import torch
from torch import nn
import math
from loguru import logger
import sys
import os

class BicycleModel(nn.Module):
    def __init__(self, wheelbase_m=1.0, drag_coeff=0.05, max_steer_rad=1.0):
        super().__init__()
        self.register_buffer("wheelbase_m", torch.tensor(wheelbase_m))
        self.register_buffer("drag_coeff", torch.tensor(drag_coeff))
        self.register_buffer("max_steer_rad", torch.tensor(max_steer_rad))

    def f_and_g(self, states):
        px, py, heading, speed, _ = states.unbind(-1)
        drift_px = speed * torch.cos(heading)
        drift_py = speed * torch.sin(heading)
        drift_heading = torch.zeros_like(speed)
        drift_speed = -self.drag_coeff * speed * speed.abs()
        drift_steer = torch.zeros_like(speed)
        f = torch.stack([drift_px, drift_py, drift_heading, drift_speed, drift_steer], dim=-1)
        B = states.shape[0]
        g = torch.zeros((B, 5, 2), dtype=states.dtype, device=states.device)
        g[:, 3, 0] = 1.0
        g[:, 4, 1] = 1.0
        g[:, 2, 1] = speed / self.wheelbase_m
        return f, g

    def dynamics(self, states, controls):
        f, g = self.f_and_g(states)
        accel, steer = controls.unbind(-1)
        steer = torch.clamp(steer, -self.max_steer_rad, self.max_steer_rad)
        f[:, 2] = (states[:, 3] / self.wheelbase_m) * torch.tan(steer)
        gu = torch.einsum("bik,bk->bi", g, controls)
        return f + gu

    def step(self, states, controls, dt, method="rk4"):
        if method == "euler":
            return states + dt * self.dynamics(states, controls)
        elif method == "rk4":
            k1 = self.dynamics(states, controls)
            k2 = self.dynamics(states + 0.5 * dt * k1, controls)
            k3 = self.dynamics(states + 0.5 * dt * k2, controls)
            k4 = self.dynamics(states + dt * k3, controls)
            return states + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        else:
            raise ValueError("method must be 'euler' or 'rk4'")

class EqMPolicy(nn.Module):
    def __init__(self, hidden_dim, num_steps, total_time):
        super().__init__()
        self.num_steps = num_steps
        self.total_time = total_time
        self.dt = total_time / num_steps
        input_dim = 2 + 4 + 2  # speed/steer + rel_pose + action
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
        # states: (B, T, 5), target_trajectories: (B, T, 5)
        rel_pos_global = target_trajectories[:, :, :2] - states[:, :, :2]
        rel_heading = target_trajectories[:, :, 2] - states[:, :, 2]
        cos_h, sin_h = torch.cos(states[:, :, 2]), torch.sin(states[:, :, 2])
        rel_x = rel_pos_global[:, :, 0] * cos_h + rel_pos_global[:, :, 1] * sin_h
        rel_y = -rel_pos_global[:, :, 0] * sin_h + rel_pos_global[:, :, 1] * cos_h
        rel_heading = torch.atan2(torch.sin(rel_heading), torch.cos(rel_heading))
        rel_theta_trig = torch.stack([torch.cos(rel_heading), torch.sin(rel_heading)], dim=-1)
        return torch.cat([rel_x.unsqueeze(-1), rel_y.unsqueeze(-1), rel_theta_trig], dim=-1)  # (B, T, 4)

    def forward(self, states, target_trajectories, actions):
        logger.debug(
            "Policy forward – states: {s}, target_trajectories: {g}, actions: {a}",
            s=states.shape, g=target_trajectories.shape, a=actions.shape
        )
        rel_pose = self._relative_pose(states, target_trajectories)
        states_part = states[:, :, 3:]  # (B, T, 2) speed, steer
        x = torch.cat([states_part, rel_pose, actions], dim=-1)  # (B, T, 8)
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

def state_error(states, trajectories):
    # Position error
    pos_error = torch.norm(states[:, :, :2] - trajectories[:, :, :2], dim=-1).mean(dim=1)

    # Heading error (angular difference using cosine-sine method)
    theta = states[:, :, 2]
    theta_target = trajectories[:, :, 2]
    cos_diff_theta = torch.cos(theta) * torch.cos(theta_target) + torch.sin(theta) * torch.sin(theta_target)
    sin_diff_theta = torch.cos(theta) * torch.sin(theta_target) - torch.sin(theta) * torch.cos(theta_target)
    heading_loss = ((1 - cos_diff_theta).pow(2) + sin_diff_theta.pow(2)).mean(dim=1)

    # Steer error (angular difference using cosine-sine method)
    steer = states[:, :, 4]
    steer_target = trajectories[:, :, 4]
    cos_diff_steer = torch.cos(steer) * torch.cos(steer_target) + torch.sin(steer) * torch.sin(steer_target)
    sin_diff_steer = torch.cos(steer) * torch.sin(steer_target) - torch.sin(steer) * torch.cos(steer_target)
    steer_loss = ((1 - cos_diff_steer).pow(2) + sin_diff_steer.pow(2)).mean(dim=1)

    # Speed error (numerical difference)
    speed_error = torch.abs(states[:, :, 3] - trajectories[:, :, 3]).mean(dim=1)

    # Combine all errors with weights
    return pos_error + 0.5 * heading_loss + 0.5 * steer_loss + speed_error

def sample_maneuver_batch(batch_size, num_steps, total_time, device, mode):
    dt = total_time / num_steps
    init = torch.zeros(batch_size, 5, device=device)
    actions = torch.zeros(batch_size, num_steps, 2, device=device)
    model = BicycleModel(max_steer_rad=1.0).to(device)

    init[:, 3] = torch.rand(batch_size, device=device) * 0.5 + 0.5
    init[:, 4] = torch.rand(batch_size, device=device) * 0.4 - 0.2

    t = torch.arange(num_steps, device=device).float() / num_steps

    num_cycles_accel = (torch.rand((batch_size,), device=device).float() * 0.5) + 0.5
    amp_accel = (torch.rand(batch_size, device=device) * 5.) + 5.0
    phase_accel = torch.zeros(batch_size, device=device) * 2 * math.pi
    accel = amp_accel.unsqueeze(1) * torch.sin(2 * math.pi * num_cycles_accel.unsqueeze(1) * t + phase_accel.unsqueeze(1))

    num_cycles_steer = (torch.rand((batch_size,), device=device).float() * 0.5) + 0.5
    amp_steer = (torch.rand(batch_size, device=device) * 0.5) + 0.5
    phase_steer = torch.rand(batch_size, device=device) * 2 * math.pi
    steer = amp_steer.unsqueeze(1) * torch.cos((2 * math.pi * t * num_cycles_steer.unsqueeze(1)) + phase_steer.unsqueeze(1))

    actions[:, :, 0] = accel
    actions[:, :, 1] = steer

    actions = torch.clamp(actions, -torch.tensor([10.0, 1.0], device=device), torch.tensor([10.0, 1.0], device=device))
    actions += torch.randn_like(actions) * 0.005

    states = init.clone().unsqueeze(1)
    for s in range(num_steps):
        nxt = model.step(states[:, -1], actions[:, s], dt=dt)
        states = torch.cat([states, nxt.unsqueeze(1)], dim=1)

    # Generate full target trajectories based on initial states and actions
    target_trajectories = states.clone()[:, 1:, :]  # (B, T, 5)

    logger.debug("Average total heading change: {thc:.3f} rad ({deg:.1f} deg)",
                 thc=(states[:, -1, 2] - states[:, 0, 2]).abs().mean(),
                 deg=(states[:, -1, 2] - states[:, 0, 2]).abs().mean() * 180 / math.pi)

    if mode is not None:
        logger.warning("Mode ignored; using sinusoidal generation for all.")

    return init, target_trajectories, actions, states

def train_controller_eqm(batch_size, num_steps, total_time, epochs, sim_steps, step_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")

    dt = total_time / num_steps
    model = BicycleModel(max_steer_rad=1.0).to(device)
    policy = EqMPolicy(hidden_dim=128, num_steps=num_steps, total_time=total_time).to(device)

    model_path="bicycle_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    policy_path="eqm_policy.pth"
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=device))

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        init_states, target_trajectories, target_actions, full_states = sample_maneuver_batch(batch_size, num_steps, total_time, device, mode=None)

        gamma = torch.rand(batch_size, 1, device=device)
        noise = torch.randn_like(target_actions) * 0.1
        u_gamma = gamma.unsqueeze(1) * target_actions + (1 - gamma.unsqueeze(1)) * noise
        c_gamma = 1.0 * (1 - gamma)
        target_grad = (noise - target_actions) * c_gamma.unsqueeze(1)

        # Simulate states with u_gamma
        states = init_states.clone().unsqueeze(1)
        for s in range(num_steps):
            next_state = model.step(states[:, -1], u_gamma[:, s], dt=dt)
            states = torch.cat([states, next_state.unsqueeze(1)], dim=1)

        # Compute grad using the same time steps as target_trajectories
        grad = policy(states[:, 1:, :], target_trajectories, u_gamma)
        eqm_loss = ((grad - target_grad) ** 2).mean()

        actions, pred_states = policy.simulate_trajectory(model, init_states, target_trajectories, u_gamma, sim_steps, step_size)
        pose_loss = state_error(pred_states, target_trajectories).mean()

        loss = eqm_loss + pose_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0 or ep <= 5:
            logger.info(
                "ep {ep:04d} | loss {l:.4f} | eqm {el:.4f} | pose {pl:.4f}",
                ep=ep, l=loss.item(), el=eqm_loss.item(), pl=pose_loss.item()
            )
            torch.save(model.state_dict(), "bicycle_model.pth")
            torch.save(policy.state_dict(), "eqm_policy.pth")

    return model, policy

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    torch.manual_seed(0)
    model, policy = train_controller_eqm(
        batch_size=256,
        num_steps=75,
        total_time=1.0,
        epochs=10000,
        sim_steps=20,
        step_size=0.1
    )