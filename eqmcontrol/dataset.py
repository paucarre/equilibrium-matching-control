import torch
import math
import pickle
from loguru import logger
from eqmcontrol.model import BicycleModel, EqMPolicy
from eqmcontrol.loss import state_error

def sample_parking_batch(batch_size, num_steps, device):
    model = BicycleModel(max_steer_rad=1.0).to(device)
    policy = EqMPolicy(hidden_dim=128, num_steps=1, total_time=0.04).to(device)
    model_path = "checkpoints/next/bicycle_model.pth"
    policy_path = "checkpoints/next/eqm_policy.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    model.eval()
    policy.eval()

    num_steps += 1
    agent_to_sequence = None
    with open("dataset/agent_to_sequence.pkl", "rb") as f:
        agent_to_sequence = pickle.load(f)

    max_steps = len(next(iter(agent_to_sequence.values())))
    start_idx = torch.randint(0, max_steps - 1 - num_steps, (1,)).item()

    invalid_tokens = set()
    for agent_token in agent_to_sequence:
        values = agent_to_sequence[agent_token][start_idx:start_idx+num_steps]
        for t, value in enumerate(values):
            if value is None or value['type'] != 'Car':
                invalid_tokens.add(agent_token)

    batch_size = min(batch_size, len(set(agent_to_sequence.keys())) - len(invalid_tokens))

    action_traj = torch.zeros(batch_size, num_steps - 1, 2, device=device)
    state_traj = torch.zeros(batch_size, num_steps, 6, device=device)
    idx = 0
    optim_steps = 20
    for agent_token in agent_to_sequence:
        if agent_token not in invalid_tokens and idx < batch_size:
            values = agent_to_sequence[agent_token][start_idx:start_idx+num_steps]
            for t, value in enumerate(values):
                #print(value['timestamp'], value['type']
                state_traj[idx, t, 0] = value['coords'][0] # X
                state_traj[idx, t, 1] = value['coords'][1] # Y
                state_traj[idx, t, 2] = value['speed']   # Speed
                state_traj[idx, t, 3] = value['heading'] # Heading
                state_traj[idx, t, 4] = value['size'][0] # Length_m
                state_traj[idx, t, 5] = value['size'][1] # Width_m
                assert state_traj[idx, t, 4] > 3.0
                assert state_traj[idx, t, 5] > 1.5
            init_actions = 2. * (torch.rand(batch_size, 1, 2, device=device) - 0.5)
            for t in range(len(values) - 1):
                with torch.no_grad():
                    actions, states = policy.simulate_trajectory(
                        model,
                        state_traj[idx:idx+1, t, :],
                        state_traj[idx:idx+1, t+1:t+2, :],
                        init_actions[idx:idx+1, ...],
                        steps=optim_steps,
                        step_size=0.1,
                        enabled_trajectories_mask=torch.ones_like(state_traj[idx:idx+1, t:t+1, 0]== 1.)
                    )
                    #assert state_error(states, state_traj[idx:idx+1, t+1:t+2, :]).item() < 1e-2
                    action_traj[idx, t, :] = actions[0, 0, :]
            idx += 1

    init = state_traj[:, 0, :]
    target_trajectories = state_traj[:, 1:, :]
    init, target_trajectories, action_traj, state_traj
    assert not torch.isnan(init).any().item()
    assert not torch.isnan(target_trajectories).any().item()
    assert not torch.isnan(action_traj).any().item()
    assert not torch.isnan(state_traj).any().item()
    return init, target_trajectories, action_traj, state_traj

if __name__ == "__main__":
    sample_parking_batch(batch_size=20, num_steps=50, dt=0.04, device="cuda:0")


def sample_maneuver_batch(batch_size, num_steps, total_time, device):
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
    return init, target_trajectories, actions, states