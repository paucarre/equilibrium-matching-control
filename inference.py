"""Inference / analysis for the transformer EqM policy.

Runs gradient-descent sampling on the learned landscape and records, per
optimisation step, the gradient norm, the action update norm and the pose
error.  The gradient norm is the interesting one: EqM's whole claim is that
ground-truth actions are stationary points, so it should decay towards zero.
"""

import argparse
import sys

import torch
from loguru import logger

from control import BicycleModel, EqMPolicy, sample_maneuver_batch, state_error


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def per_step_error(states, targets):
    """state_error, but kept per timestep: (B, T, 5) x (B, T, 5) -> (B, T)."""
    B, T, D = states.shape
    return state_error(states.reshape(B * T, 1, D), targets.reshape(B * T, 1, D)).view(B, T)


def build_initial_actions(mode, target_actions, policy, noise_std, gamma):
    """Where the optimisation starts.

    noise : pure prior sample, the honest test of the landscape
    gamma : partially corrupted, u = gamma*u* + (1-gamma)*eps  (paper Fig. 10)
    zeros : no control at all
    gt    : the ground-truth actions -- a sanity check only.  If the landscape
            is correct the policy should leave these almost untouched.
    """
    noise = torch.randn_like(target_actions) * policy.action_scale * noise_std
    if mode == "noise":
        return noise
    if mode == "gamma":
        return gamma * target_actions + (1.0 - gamma) * noise
    if mode == "zeros":
        return torch.zeros_like(target_actions)
    if mode == "gt":
        return target_actions.clone()
    raise ValueError(f"unknown init mode: {mode}")


def policy_from_checkpoint(path, num_steps, total_time, n_heads, device):
    """Rebuild the policy with the architecture stored in the checkpoint.

    d_model / n_layers / mlp_ratio are read off the weights so the script
    cannot silently disagree with training.  n_heads only affects how the
    attention tensor is reshaped, not the parameter shapes, so it has to be
    supplied and must match what was trained.
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)

    d_model = state_dict["tokenizer.proj.weight"].shape[0]
    n_layers = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("blocks."))
    mlp_ratio = state_dict["blocks.0.mlp.0.weight"].shape[0] // d_model

    policy = EqMPolicy(
        num_steps=num_steps,
        total_time=total_time,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        mlp_ratio=mlp_ratio,
    ).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()

    logger.info(
        "Loaded policy from {p} (d_model={d}, layers={l}, heads={h}, mlp_ratio={m})",
        p=path, d=d_model, l=n_layers, h=n_heads, m=mlp_ratio,
    )
    return policy


# --------------------------------------------------------------------------- #
# Sampling with instrumentation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def optimize_actions(
    policy,
    model,
    init_states,
    target_trajectories,
    actions,
    steps,
    step_size,
    momentum=0.0,
    grad_tol=None,
    track_error=True,
):
    """Same descent as EqMPolicy.simulate_trajectory, with a per-step trace.

    Returns (actions, states, trace) where each trace entry is (B, n_steps).
    """
    actions = policy.clamp_actions(actions.clone())
    prev_actions = actions
    trace = {"grad_norm": [], "update_norm": [], "pose_error": []}

    for _ in range(steps):
        if momentum > 0.0:
            probe = policy.clamp_actions(actions + momentum * (actions - prev_actions))
        else:
            probe = actions

        states = policy.rollout(model, init_states, probe)
        grad = policy(states, target_trajectories, probe)

        prev_actions = actions
        new_actions = policy.clamp_actions(actions - step_size * grad)

        trace["grad_norm"].append(grad.flatten(1).norm(dim=1))
        trace["update_norm"].append((new_actions - actions).flatten(1).norm(dim=1))
        actions = new_actions

        if track_error:
            rolled = policy.rollout(model, init_states, actions)
            trace["pose_error"].append(per_step_error(rolled, target_trajectories).mean(dim=1))

        # Adaptive compute (paper 3.3): stop once the whole batch is at rest.
        if grad_tol is not None and trace["grad_norm"][-1].max() < grad_tol:
            logger.info("Gradient below {t} after {n} steps; stopping early.",
                        t=grad_tol, n=len(trace["grad_norm"]))
            break

    states = policy.rollout(model, init_states, actions)
    trace = {k: torch.stack(v, dim=1) for k, v in trace.items() if v}
    return actions, states, trace


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_inference(
    batch_size=1,
    num_steps=75,
    total_time=1.0,
    optimization_steps=20,
    step_size=0.1,
    momentum=0.0,
    grad_tol=None,
    init_mode="noise",
    noise_std=0.1,
    gamma=0.5,
    n_heads=8,
    policy_path="eqm_policy.pth",
    output_path="trajectory_data.pt",
    seed=None,
    device=None,
):
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Running inference on {d}", d=device)
    if seed is not None:
        torch.manual_seed(seed)
    # PyTorch seeds its default generator from OS entropy, so leaving seed as
    # None gives a different maneuver every run (the original script's
    # behaviour).  Log whatever we ended up with so any run can be replayed
    # afterwards with --seed.
    logger.info("Torch seed: {s}{k}", s=torch.initial_seed(),
                k=" (explicit)" if seed is not None else " (random; pass --seed to fix)")

    # BicycleModel holds only buffers (no learnable parameters), so there is
    # nothing to load -- the old bicycle_model.pth checkpoint was empty.
    model = BicycleModel(max_steer_rad=1.0).to(device).eval()
    policy = policy_from_checkpoint(policy_path, num_steps, total_time, n_heads, device)

    init_states, target_trajectories, target_actions = sample_maneuver_batch(
        batch_size, num_steps, total_time, device
    )
    # states and targets are aligned: index t is the state *after* action t.
    # The full ground-truth rollout including t=0 is init + targets.
    gt_states = torch.cat([init_states.unsqueeze(1), target_trajectories], dim=1)

    init_actions = build_initial_actions(init_mode, target_actions, policy, noise_std, gamma)
    init_states_rolled = policy.rollout(model, init_states, init_actions)
    init_error = per_step_error(init_states_rolled, target_trajectories).mean(dim=1)

    actions, states, trace = optimize_actions(
        policy, model, init_states, target_trajectories, init_actions,
        optimization_steps, step_size, momentum=momentum, grad_tol=grad_tol,
    )

    errors = per_step_error(states, target_trajectories)  # (B, T)
    action_mae = (actions - target_actions).abs().mean(dim=(1, 2))

    logger.info("init mode {m!r} | error before {a:.4f} -> after {b:.4f} (mean over batch)",
                m=init_mode, a=init_error.mean().item(), b=errors.mean().item())
    logger.info("grad norm {g0:.4f} -> {g1:.4f} | action MAE vs ground truth {e:.4f}",
                g0=trace["grad_norm"][:, 0].mean().item(),
                g1=trace["grad_norm"][:, -1].mean().item(),
                e=action_mae.mean().item())
    logger.info("final position error {p:.4f} | final heading error {h:.4f} rad",
                p=(states[:, -1, :2] - target_trajectories[:, -1, :2]).norm(dim=-1).mean().item(),
                h=(states[:, -1, 2] - target_trajectories[:, -1, 2]).abs().mean().item())

    trajectory_data = {
        "init_state": init_states,                  # (B, 5)
        "gt_states": gt_states,                     # (B, T + 1, 5)
        "target_trajectory": target_trajectories,   # (B, T, 5)
        "target_actions": target_actions,           # (B, T, 2)
        "init_actions": init_actions,               # (B, T, 2)
        "states": states,                           # (B, T, 5)
        "actions": actions,                         # (B, T, 2)
        "errors": errors,                           # (B, T)
        "init_error": init_error,                   # (B,)
        "action_mae": action_mae,                   # (B,)
        "trace": trace,                             # each (B, n_opt_steps)
        "config": {
            "num_steps": num_steps, "total_time": total_time,
            "optimization_steps": optimization_steps, "step_size": step_size,
            "momentum": momentum, "init_mode": init_mode, "noise_std": noise_std,
            "gamma": gamma, "seed": seed, "torch_seed": torch.initial_seed(),
        },
    }
    torch.save(trajectory_data, output_path)
    logger.info("Saved trajectory data to {p}", p=output_path)

    return trajectory_data


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=75)
    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--optimization-steps", type=int, default=20)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.0, help="NAG look-ahead factor")
    p.add_argument("--grad-tol", type=float, default=None, help="adaptive-compute stop threshold")
    p.add_argument("--init-mode", choices=["noise", "gamma", "zeros", "gt"], default="noise")
    p.add_argument("--noise-std", type=float, default=0.1, help="must match training")
    p.add_argument("--gamma", type=float, default=0.5, help="only used with --init-mode gamma")
    p.add_argument("--n-heads", type=int, default=8, help="must match training")
    p.add_argument("--policy-path", default="eqm_policy.pth")
    p.add_argument("--output-path", default="trajectory_data.pt")
    p.add_argument("--seed", type=int, default=None,
                   help="omit for a different maneuver each run")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = parse_args()
    run_inference(**vars(args))