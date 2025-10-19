import click
import torch
from torch import nn
from loguru import logger
import sys
from eqmcontrol.train import state_error
from eqmcontrol.model import BicycleModel, EqMPolicy
from eqmcontrol.dataset import sample_maneuver_batch

def run_inference(
    batch_size, num_steps, total_time, optimization_steps, step_size, device, model_path, policy_path, output_path
):
    """
    Run inference to simulate a trajectory for a single initial state and target trajectory.
    Save a dictionary containing the state trajectory, action trajectory, initial state,
    target trajectory, state errors, initial actions, and action update norms for each timestamp.

    Args:
        batch_size (int): Number of trajectories (set to 1 for single case).
        num_steps (int): Number of time steps in the trajectory.
        total_time (float): Total simulation time in seconds.
        optimization_steps (int): Number of optimization steps for the policy.
        step_size (float): Step size for gradient-based action optimization.
        device (str or torch.device): Device to run the simulation on (defaults to None, uses CUDA if available).
        model_path (str): Path to the saved BicycleModel weights.
        policy_path (str): Path to the saved EqMPolicy weights.
        output_path (str): Path to save the trajectory data dictionary.

    Returns:
        init_state (torch.Tensor): Initial state [1, 5].
        target_trajectory (torch.Tensor): Target trajectory [1, num_steps, 5].
        actions (torch.Tensor): Optimized actions [1, num_steps, 2].
        states (torch.Tensor): Simulated states [1, num_steps, 5].
        errors (torch.Tensor): State errors for each timestamp [1, num_steps + 1].
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running inference on {device}")

    # Initialize models
    model = BicycleModel(max_steer_rad=1.0).to(device)
    policy = EqMPolicy(hidden_dim=128, num_steps=num_steps, total_time=total_time).to(device)

    # Load pre-trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        logger.info(f"Loaded model weights from {model_path} and {policy_path}")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    model.eval()
    policy.eval()

    # Generate a single initial state, target trajectory, and initial actions
    init_states, target_trajectories, init_actions, gt_states = sample_maneuver_batch(
        batch_size, num_steps, total_time, device, mode=None
    )
    logger.info(f"init_states shape: {init_states.shape}, target_trajectories shape: {target_trajectories.shape}, init_actions shape: {init_actions.shape}, gt_states shape: {gt_states.shape}")

    # Use the first sample (since batch_size=1)
    init_state = init_states[0:1]  # [1, 5]: [px, py, heading, speed, steer]
    target_trajectory = target_trajectories[0:1]  # [1, num_steps, 5]: [px, py, heading, speed, steer]
    init_actions = init_actions[0:1]  # [1, num_steps, 2]: [accel, steer]
    logger.info(f"init_state shape: {init_state.shape}, target_trajectory shape: {target_trajectory.shape}, init_actions shape: {init_actions.shape}")

    logger.info(
        "Initial state: {s}, Target trajectory shape: {g}, Initial actions shape: {a}",
        s=init_state, g=target_trajectory.shape, a=init_actions.shape
    )

    # Simulate trajectory
    dt = total_time / num_steps
    with torch.no_grad():
        actions, states = policy.simulate_trajectory(
            model, init_state, target_trajectory, init_actions, optimization_steps, step_size
        )
        logger.info(f"actions shape: {actions.shape}, states shape: {states.shape}")
        # Log action changes
        logger.info(f"Action change norm: {torch.norm(actions - init_actions).item():.6f}")

    # Compute state errors for each timestamp
    target_trajectory_expanded = torch.zeros(1, num_steps + 1, 5, device=device)
    target_trajectory_expanded[:, 1:, :] = target_trajectory  # Fixed slicing to include all dimensions
    logger.info(f"target_trajectory_expanded shape: {target_trajectory_expanded.shape}")
    logger.debug(f"States shape: {states.shape}, Target trajectory expanded shape: {target_trajectory_expanded.shape}")
    errors = torch.zeros(1, num_steps + 1, device=device)
    for t in range(num_steps):  # Changed to num_steps to match states shape [1, num_steps, 5]
        state_t = states[:, t, :].unsqueeze(1)  # [1, 1, 5] to match state_error expectation
        goal_t = target_trajectory_expanded[:, t + 1, :].unsqueeze(1)  # Shift by 1 to align with simulated steps
        errors[:, t + 1] = state_error(state_t, goal_t).squeeze()  # [1]
    # Set initial state error (t=0) to 0 or compute separately if needed
    errors[:, 0] = state_error(init_state.unsqueeze(1), target_trajectory_expanded[:, 0, :].unsqueeze(1)).squeeze()

    logger.info(f"errors shape: {errors.shape}")

    # Log error components for the final timestamp
    pos_error = torch.norm(states[:, :, :2] - target_trajectory_expanded[:, 1:, :2], dim=2)  # Adjust for num_steps
    theta = states[:, :, 2]
    theta_target = target_trajectory_expanded[:, 1:, 2]
    cos_diff = torch.cos(theta) * torch.cos(theta_target) + torch.sin(theta) * torch.sin(theta_target)
    sin_diff = torch.cos(theta) * torch.sin(theta_target) - torch.sin(theta) * torch.cos(theta_target)
    heading_loss = (1 - cos_diff).pow(2) + sin_diff.pow(2)
    logger.info(f"Final position error: {pos_error[0, -1].item():.6f}, Final heading loss: {0.5 * heading_loss[0, -1].item():.6f}")

    # Create dictionary to save
    trajectory_data = {
        "gt_states": gt_states,
        "init_state": init_state,  # [1, 5]
        "target_trajectory": target_trajectory,  # [1, num_steps, 5] (derived from gt_states[:, 1:, :5] but kept for convenience)
        "states": states,  # [1, num_steps, 5]
        "actions": actions,  # [1, num_steps, 2]
        "errors": errors,  # [1, num_steps + 1]
        "init_actions": init_actions,  # [1, num_steps, 2]
    }

    # Save dictionary to disk
    try:
        torch.save(trajectory_data, output_path)
        logger.info(f"Saved trajectory data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save trajectory data: {e}")
        raise

    logger.info(
        "Simulation complete. Actions shape: {a}, States shape: {s}, Errors shape: {e}",
        a=actions.shape, s=states.shape, e=errors.shape
    )

    return init_state, target_trajectory, actions, states, errors

@click.command()
@click.option('--batch-size', default=1, type=int, help='Batch size for inference')
@click.option('--num-steps', default=75, type=int, help='Number of steps')
@click.option('--total-time', default=1.0, type=float, help='Total time for simulation')
@click.option('--optimization-steps', default=20, type=int, help='Number of optimization steps')
@click.option('--step-size', default=0.1, type=float, help='Step size for simulation')
@click.option('--device', default=None, type=str, help='Device to run inference on (e.g., cpu, cuda)')
@click.option('--model-path', default="bicycle_model.pth", type=str, help='Path to the model file')
@click.option('--policy-path', default="eqm_policy.pth", type=str, help='Path to the policy file')
@click.option('--output-path', default="trajectory_data.pt", type=str, help='Path to save trajectory data')
def run_inference_cmd(batch_size, num_steps, total_time, optimization_steps, step_size, device, model_path, policy_path, output_path):
    """Run inference with the specified parameters and log results."""
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Run inference
    init_state, target_trajectory, actions, states, errors = run_inference(
        batch_size=batch_size,
        num_steps=num_steps,
        total_time=total_time,
        optimization_steps=optimization_steps,
        step_size=step_size,
        device=device,
        model_path=model_path,
        policy_path=policy_path,
        output_path=output_path
    )

    # Log results
    logger.info(f"Initial State [px, py, heading, speed, steer]: {init_state.cpu().numpy()}")
    logger.info(f"Target Trajectory shape [px, py, heading]: {target_trajectory.shape}")
    logger.info(f"Actions shape [batch, time, (accel, steer)]: {actions.shape}")
    logger.info(f"States shape [batch, time, (px, py, heading, speed, steer)]: {states.shape}")
    logger.info(f"Errors shape [batch, time]: {errors.shape}")
    logger.info(f"Final Simulated State: {states[0, -1].cpu().numpy()}")
    logger.info(f"Final Pose Error: {errors[0, -1].cpu().numpy()}")

if __name__ == "__main__":
    run_inference_cmd()