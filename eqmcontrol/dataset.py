import torch
import math
from eqmcontrol.model import BicycleModel
from loguru import logger


def sample_maneuver_batch(batch_size, num_steps, total_time, device, mode):

    dt = total_time / num_steps
    actions = torch.zeros(batch_size, num_steps, 2, device=device)
    model = BicycleModel(max_steer_rad=1.0).to(device)

    init = torch.zeros(batch_size, 6, device=device)
    init[:, 0] = torch.rand(batch_size, device=device) * 100.0 # X
    init[:, 1] = torch.rand(batch_size, device=device) * 100.0 # Y
    init[:, 2] = (torch.rand(batch_size, device=device) * 5) - 2.5 # Speed
    init[:, 3] = torch.rand(batch_size, device=device) * 2. * math.pi # Heading
    init[:, 4] = 4. + (torch.rand(batch_size, device=device) * 1.5) # dlength_m
    init[:, 5] = 2. + (torch.rand(batch_size, device=device) * 0.5) # dwidth_m

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