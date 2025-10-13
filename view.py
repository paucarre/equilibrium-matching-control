"""
Visualize trajectory data from trajectory_data.pt by saving to .rrd for Rerun.
Displays the car moving with a blue box for current pose, transparent green box for target pose,
and charts for state error, acceleration, and steering rate.
"""

import click
import numpy as np
import torch
import rerun as rr
import sys
import time

@click.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--rerun-addr", default="127.0.0.1:9876", help="Address of running Rerun server (ip:port, ignored in 0.25.1)")
@click.option("--save-only", is_flag=True, help="Save to .rrd file without attempting to connect")
def main(data_path, rerun_addr, save_only):
    """
    Visualize trajectory data by saving to .rrd for viewing in Rerun.

    Args:
        data_path (str): Path to trajectory_data.pt
        rerun_addr (str): Address of Rerun server (ignored in 0.25.1)
        save_only (bool): If set, saves to .rrd without attempting connection
    """
    # Initialize Rerun
    print("Note: rerun-sdk 0.25.1 does not support direct remote connections.")
    print(f"Saving visualization to trajectory.rerun{' for transfer to ' + rerun_addr if not save_only else ''}.")
    rr.init("bicycle_trajectory", spawn=False)

    # Load the trajectory data
    try:
        data = torch.load(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Run inference.py first.")
        sys.exit(1)

    init_state = data["init_state"]  # [1, 5]
    states = data["states"]  # [1, T, 5]
    actions = data["actions"]  # [1, T, 2]
    errors = data["errors"]  # [1, T]
    gt_states = data["gt_states"]

    target_goal = data["target_trajectory"][:, -1, :]  # [1, 3]

    # Convert to numpy for Rerun
    gt_states_np = gt_states[0].cpu().numpy()  # [T, 5]
    states_np = states[0].cpu().numpy()  # [T, 5]
    actions_np = actions[0].cpu().numpy()  # [T, 2]
    errors_np = errors[0].cpu().numpy()  # [T]
    target_goal_np = target_goal[0, :3].cpu().numpy()  # [3]

    T = len(states_np)
    print(f"Loaded trajectory with {T} time steps.")

    # Log static entities (no timeline)
    # Log the world origin
    rr.log("world/", rr.Transform3D())


    # Log target pose as a transparent green box (static)
    target_pos = target_goal_np[:2]
    target_heading = target_goal_np[2]
    # Calculate direction vector from heading angle
    arrow_length = 0.5  # Adjust the arrow length as needed
    direction = np.array([np.cos(target_heading), np.sin(target_heading)]) * arrow_length
    rr.log(
        "world/target_pose",
        rr.Arrows2D(
            origins=[target_pos],    # position should be [x, y]
            vectors=[direction],   # direction vector for heading
            colors=[[0, 255, 0]],  # blue arrow for current pose
            radii=[0.01],
            labels=["target"]
        )
    )
    # Log full trajectory line (static)
    gt_trajectory_pos = gt_states_np[:, :2]
    rr.log("world/gt_trajectory", rr.LineStrips2D([gt_trajectory_pos]))

    trajectory_pos = states_np[:, :2]
    rr.log("world/trajectory", rr.LineStrips2D([trajectory_pos]))

    # Animate: log current pose and charts per step
    for t in range(T):
        rr.set_time("step", sequence=t)
        time.sleep(0.1)
        arrow_length = .5  # Adjust the arrow length as needed

        # Current pose as blue box
        current_pos = states_np[t, :2]
        current_heading = states_np[t, 2]
        direction = np.array([np.cos(current_heading), np.sin(current_heading)]) * arrow_length
        rr.log(
            "world/path",
            rr.Arrows2D(
                origins=[current_pos],    # position should be [x, y]
                vectors=[direction],   # direction vector for heading
                colors=[[0, 0, 255]],  # blue arrow for current pose
                #radii=[0.1],
                labels=["path"]
            )
        )

        # Current pose as blue box
        gt_current_pos = gt_states_np[t+1, :2]
        gt_current_heading = gt_states_np[t+1, 2]
        direction = np.array([np.cos(gt_current_heading), np.sin(gt_current_heading)]) * arrow_length
        rr.log(
            "world/target_path",
            rr.Arrows2D(
                origins=[gt_current_pos],    # position should be [x, y]
                vectors=[direction],   # direction vector for heading
                colors=[[0, 0, 255]],  # blue arrow for current pose
                #radii=[0.1],
                #labels=["target_path"]
            )
        )


        # Log state error chart (up to current t)
        error_values = errors_np[:t + 1]
        rr.log("charts/error", rr.SeriesLines(colors=[255, 0, 0, 255], names=["State Error"]))

        # Log actions charts: acceleration and steering
        accel_values = actions_np[:t + 1, 0]
        steer_values = actions_np[:t + 1, 1]
        rr.log("charts/accel", rr.SeriesLines(colors=[0, 255, 0, 255], names=["Acceleration"]))
        rr.log("charts/steer", rr.SeriesLines(colors=[128, 0, 128, 255], names=["Steering Rate"]))

    # Save to .rrd file
    rr.save("trajectory.rrd")
    print("Visualization saved to trajectory.rrd.")
    print(f"Copy trajectory.rerun to the remote machine ({rerun_addr}) and run 'rerun trajectory.rrd' to view.")
    print("Play the 'step' sequence to see the animation.")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("Error: Install required packages: pip install rerun-sdk==0.25.1 torch numpy click")
        sys.exit(1)