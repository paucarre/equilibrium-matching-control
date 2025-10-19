import click
import torch
from loguru import logger
import sys
import os
from eqmcontrol.dataset import sample_maneuver_batch, sample_parking_batch
from eqmcontrol.model import BicycleModel, EqMPolicy
from eqmcontrol.loss import state_error


def train_controller_eqm(batch_size, num_steps, mode, dt, epochs, sim_steps, step_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")

    total_time  = dt * num_steps
    model = BicycleModel(max_steer_rad=1.0).to(device)
    policy = EqMPolicy(hidden_dim=128, num_steps=num_steps, total_time=total_time).to(device)

    model_path=f"checkpoints/{mode}/bicycle_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    policy_path=f"checkpoints/{mode}/eqm_policy.pth"
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=device))

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-8)
    assert dt == 0.04
    for ep in range(1, epochs + 1):
        if mode == "next":
            init_states, target_trajectories, target_actions, _ = sample_maneuver_batch(batch_size, num_steps, total_time, device)
        elif mode == "parking":
            init_states, target_trajectories, target_actions, _ = sample_parking_batch(batch_size, num_steps, device=device)

        enabled_trajectories_mask = target_trajectories[:, :, 4] > 3.

        gamma = torch.rand(batch_size, 1, device=device)
        noise = torch.randn_like(target_actions) * 0.1
        u_gamma = gamma.unsqueeze(1) * target_actions + (1 - gamma.unsqueeze(1)) * noise
        c_gamma = 1.0 * (1 - gamma)
        target_grad = (noise - target_actions) * c_gamma.unsqueeze(1)

        # Simulate states with u_gamma
        states = init_states.clone().unsqueeze(1)
        for s in range(num_steps):
            next_state = model.step(states[:, -1], u_gamma[:, s], dt=dt,
                                    enabled_trajectories_mask=enabled_trajectories_mask[:, s])
            states = torch.cat([states, next_state.unsqueeze(1)], dim=1)

        # Compute grad using the same time steps as target_trajectories
        grad = policy(states[:, 1:, :], target_trajectories, u_gamma)
        eqm_loss = ((grad - target_grad) ** 2).mean()

        actions, pred_states = policy.simulate_trajectory(model, init_states, target_trajectories, u_gamma, sim_steps, step_size,
                                                          enabled_trajectories_mask)

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
            torch.save(model.state_dict(), f"checkpoints/{mode}/bicycle_model.pth")
            torch.save(policy.state_dict(), f"checkpoints/{mode}/eqm_policy.pth")

    return model, policy

@click.command()
@click.option('--batch-size', default=256, type=int, help='Batch size for training')
@click.option('--num-steps', default=50, type=int, help='Number of steps')
@click.option('--mode', default="parking", type=str, help='Training mode')
@click.option('--dt', default=0.04, type=float, help='Total time for simulation')
@click.option('--epochs', default=10000, type=int, help='Number of training epochs')
@click.option('--sim-steps', default=20, type=int, help='Number of simulation steps')
@click.option('--step-size', default=0.1, type=float, help='Step size for simulation')
def train(batch_size, num_steps, mode, dt, epochs, sim_steps, step_size):
    """Train the controller model with specified parameters."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    #torch.manual_seed(0)
    model, policy = train_controller_eqm(
        batch_size=batch_size,
        num_steps=num_steps,
        mode=mode,
        dt=dt,
        epochs=epochs,
        sim_steps=sim_steps,
        step_size=step_size
    )
    return model, policy

if __name__ == "__main__":
    train()