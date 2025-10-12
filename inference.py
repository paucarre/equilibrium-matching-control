import torch
from torch import nn
from loguru import logger
import sys
from control import BicycleModel, EqMPolicy, sample_maneuver_batch, pose_error_cos_sin

def run_inference(
    batch_size, num_steps, total_time, optimization_steps, step_size, device, model_path, policy_path, output_path
):
    """
    Run inference to simulate a trajectory for a single initial state and target goal.
    Save a dictionary containing the state trajectory, action trajectory, initial state,
    target state, pose errors, initial actions, and action update norms for each timestamp.

    Args:
        batch_size (int): Number of trajectories (set to 1 for single case).
        num_steps (int): Number of time steps in the trajectory.
        total_time (float): Total simulation time in seconds.
        optimization_steps (int): Number of optimization steps for the policy.
        step_size (float): Step size for gradient-based action optimization.
        device (str or torch.device): Device to run the simulation on.
        model_path (str): Path to the saved BicycleModel weights.
        policy_path (str): Path to the saved EqMPolicy weights.
        output_path (str): Path to save the trajectory data dictionary.

    Returns:
        init_state (torch.Tensor): Initial state [1, 5].
        target_goal (torch.Tensor): Target goal [1, 3].
        actions (torch.Tensor): Optimized actions [1, num_steps, 2].
        states (torch.Tensor): Simulated states [1, num_steps, 5].
        errors (torch.Tensor): Pose errors for each timestamp [1, num_steps].
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

    # Generate a single initial state, final state, and initial actions
    init_states, final_states, init_actions, gt_states = sample_maneuver_batch(
        batch_size, num_steps, total_time, device, mode=None
    )

    # Use the first sample (since batch_size=1)
    init_state = init_states[0:1]  # [1, 5]: [px, py, heading, speed, steer]
    target_goal = final_states[0:1, :3]  # [1, 3]: [px, py, heading]
    init_actions = init_actions[0:1]  # [1, num_steps, 2]: [accel, steer]

    logger.info(
        "Initial state: {s}, Target goal: {g}, Initial actions shape: {a}",
        s=init_state, g=target_goal, a=init_actions.shape
    )

    # Simulate trajectory
    dt = total_time / num_steps
    with torch.no_grad():
        actions, states = policy.simulate_trajectory(
            model, init_state, target_goal, init_actions, optimization_steps, step_size
        )
        # Log action changes
        logger.info(f"Action change norm: {torch.norm(actions - init_actions).item():.6f}")

    # Compute pose errors for each timestamp
    target_goal_expanded = torch.zeros(1, num_steps, 5, device=device)
    target_goal_expanded[:, :, :3] = target_goal.unsqueeze(1).expand(-1, num_steps, -1)
    target_goal_expanded[:, :, 3:] = 0.0
    logger.debug(f"States shape: {states.shape}, Target goal expanded shape: {target_goal_expanded.shape}")
    errors = torch.zeros(1, num_steps, device=device)
    for t in range(num_steps):
        state_t = states[:, t, :]  # [1, 5]
        goal_t = target_goal_expanded[:, t, :]  # [1, 5]
        errors[:, t] = pose_error_cos_sin(state_t, goal_t)  # [1]
    # Log error components for the final timestamp
    pos_error = torch.norm(states[:, :, :2] - target_goal_expanded[:, :, :2], dim=2)  # [1, num_steps]
    theta = states[:, :, 2]
    theta_target = target_goal_expanded[:, :, 2]
    cos_diff = torch.cos(theta) * torch.cos(theta_target) + torch.sin(theta) * torch.sin(theta_target)
    sin_diff = torch.cos(theta) * torch.sin(theta_target) - torch.sin(theta) * torch.cos(theta_target)
    heading_loss = (1 - cos_diff).pow(2) + sin_diff.pow(2)
    logger.info(f"Final position error: {pos_error[0, -1].item():.6f}, Final heading loss: {0.5 * heading_loss[0, -1].item():.6f}")

    # Create dictionary to save
    trajectory_data = {
        "gt_states": gt_states,
        "init_state": init_state,  # [1, 5]
        "target_goal": target_goal,  # [1, 3]
        "states": states,  # [1, num_steps, 5]
        "actions": actions,  # [1, num_steps, 2]
        "errors": errors,  # [1, num_steps]
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

    return init_state, target_goal, actions, states, errors

if __name__ == "__main__":
    # Set up logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Run inference
    init_state, target_goal, actions, states, errors = run_inference(
        batch_size=1,
        num_steps=75,
        total_time=1.,
        optimization_steps=20,
        step_size=0.1,
        device=None,
        model_path="bicycle_model.pth",
        policy_path="eqm_policy.pth",
        output_path="trajectory_data.pt"
    )

    logger.info(f"Initial State [px, py, heading, speed, steer]: {init_state.cpu().numpy()}")
    logger.info(f"Target Goal [px, py, heading]: {target_goal.cpu().numpy()}")
    logger.info(f"Actions shape [batch, time, (accel, steer)]: {actions.shape}")
    logger.info(f"States shape [batch, time, (px, py, heading, speed, steer)]: {states.shape}")
    logger.info(f"Errors shape [batch, time]: {errors.shape}")
    logger.info(f"Final Simulated State: {states[0, -1].cpu().numpy()}")
    logger.info(f"Final Pose Error: {errors[0, -1].cpu().numpy()}")