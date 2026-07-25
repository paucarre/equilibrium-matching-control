import contextlib
import math
import os
import sys

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn


# --------------------------------------------------------------------------- #
# Dynamics
# --------------------------------------------------------------------------- #
class BicycleModel(nn.Module):
    def __init__(self, wheelbase_m=1.0, drag_coeff=0.05, max_steer_rad=1.0,
                 heading_mode="exact"):
        """heading_mode:
          "exact"         d(heading)/dt = (v / L) * tan(clamp(delta))
          "legacy_double" reproduces the original code, where g[:, 2, 1] added a
                          *second* small-angle heading term from the same
                          control (using the un-clamped value), so heading was
                          integrated roughly twice.  Physically wrong, but it
                          is what the original dataset was generated with:
                          switching to "exact" roughly halves every turn
                          (79 deg -> 43 deg mean heading change at the default
                          sampler settings).  If you want that curvature back
                          under correct physics, shorten wheelbase_m rather
                          than re-enabling this.
        """
        super().__init__()
        if heading_mode not in ("exact", "legacy_double"):
            raise ValueError("heading_mode must be 'exact' or 'legacy_double'")
        self.heading_mode = heading_mode
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
        g[:, 2, 1] = speed / self.wheelbase_m  # small-angle heading rate
        return f, g

    def dynamics(self, states, controls):
        f, g = self.f_and_g(states)
        accel, steer = controls.unbind(-1)
        steer = torch.clamp(steer, -self.max_steer_rad, self.max_steer_rad)

        f = f.clone()
        f[:, 2] = (states[:, 3] / self.wheelbase_m) * torch.tan(steer)

        if self.heading_mode == "exact":
            # Drop the duplicate small-angle heading term; see __init__.
            g = g.clone()
            g[:, 2, 1] = 0.0

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


# --------------------------------------------------------------------------- #
# Rotary position embeddings over the time axis
# --------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Standard RoPE, applied to the time index of the trajectory."""

    def __init__(self, head_dim, base=10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(device))  # (T, head_dim // 2)
        return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim // 2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.flatten(-2)


class RoPESelfAttention(nn.Module):
    """Bidirectional self-attention with rotary positions.

    We do not use nn.TransformerEncoderLayer here because RoPE has to be
    applied to q and k *after* the input projection, which that module does
    not expose.  Everything else (LayerNorm, Linear, GELU, the fused SDPA
    kernel) is stock PyTorch.
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, T, hd)

        cos, sin = self.rope(T, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        attn_mask = None
        if key_padding_mask is not None:  # (B, T) bool, True = keep
            attn_mask = key_padding_mask[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm block: attention over time, then a position-wise MLP."""

    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Linear(mlp_ratio * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


# --------------------------------------------------------------------------- #
# State tokenizer
# --------------------------------------------------------------------------- #
class StateTokenizer(nn.Module):
    """Maps (state_t, target_t, action_t) to one d_model token per timestep.

    Features are ego-relative and roughly unit-scaled, so the network never
    sees raw world coordinates or raw +/-10 accelerations.
    """

    FEATURE_DIM = 10

    def __init__(self, d_model, max_accel, max_steer, pos_scale=1.0, speed_scale=1.0):
        super().__init__()
        self.register_buffer("action_scale", torch.tensor([max_accel, max_steer]))
        self.register_buffer("pos_scale", torch.tensor(float(pos_scale)))
        self.register_buffer("speed_scale", torch.tensor(float(speed_scale)))
        self.proj = nn.Linear(self.FEATURE_DIM, d_model)
        self.norm = nn.LayerNorm(d_model)

    def features(self, states, targets, actions):
        px, py, heading, speed, steer = states.unbind(-1)
        t_px, t_py, t_heading, t_speed, t_steer = targets.unbind(-1)

        dx, dy = t_px - px, t_py - py
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        rel_x = (dx * cos_h + dy * sin_h) / self.pos_scale
        rel_y = (-dx * sin_h + dy * cos_h) / self.pos_scale
        d_heading = t_heading - heading

        feats = torch.stack(
            [
                speed / self.speed_scale,
                steer / self.action_scale[1],
                rel_x,
                rel_y,
                torch.cos(d_heading),
                torch.sin(d_heading),
                (t_speed - speed) / self.speed_scale,
                (t_steer - steer) / self.action_scale[1],
            ],
            dim=-1,
        )
        return torch.cat([feats, actions / self.action_scale], dim=-1)  # (B, T, 10)

    def forward(self, states, targets, actions):
        return self.norm(self.proj(self.features(states, targets, actions)))


# --------------------------------------------------------------------------- #
# Equilibrium-matching policy
# --------------------------------------------------------------------------- #
class EqMPolicy(nn.Module):
    """Predicts the EqM gradient for a whole action sequence at once.

    forward(states, target_trajectories, actions) -> (B, T, 2)

    No gamma / timestep conditioning: the landscape is time-invariant in the
    EqM sense.  RoPE encodes the *trajectory* index t, which is a different
    axis from the corruption level and is legitimately observable.
    """

    def __init__(
        self,
        num_steps,
        total_time,
        d_model=256,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.0,
        max_accel=10.0,
        max_steer=1.0,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.total_time = total_time
        self.dt = total_time / num_steps
        self.max_accel = max_accel
        self.max_steer = max_steer

        self.tokenizer = StateTokenizer(d_model, max_accel, max_steer)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

        self.register_buffer("action_scale", torch.tensor([max_accel, max_steer]))
        self.apply(self._init_weights)
        # Zero-init the head so the model starts as the identity landscape
        # (predicted gradient == 0) instead of injecting noise into sampling.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, states, target_trajectories, actions, key_padding_mask=None):
        logger.debug(
            "Policy forward - states: {s}, targets: {g}, actions: {a}",
            s=tuple(states.shape), g=tuple(target_trajectories.shape), a=tuple(actions.shape),
        )
        h = self.tokenizer(states, target_trajectories, actions)
        for block in self.blocks:
            h = block(h, key_padding_mask)
        h = self.final_norm(h)
        # Head predicts a normalised gradient; rescale to action units so the
        # output is directly comparable to (eps - u*) * c(gamma).
        return self.head(h) * self.action_scale

    # ------------------------------------------------------------------ #
    # Rollout / sampling
    # ------------------------------------------------------------------ #
    def rollout(self, model, init_states, actions):
        """(B, 5), (B, T, 2) -> (B, T, 5), excluding the initial state."""
        state = init_states
        states = []
        for t in range(actions.shape[1]):
            state = model.step(state, actions[:, t], dt=self.dt)
            states.append(state)
        return torch.stack(states, dim=1)

    def clamp_actions(self, actions):
        lo = -self.action_scale
        return torch.clamp(actions, lo, self.action_scale)

    def simulate_trajectory(
        self,
        model,
        init_states,
        target_trajectories,
        init_actions,
        steps,
        step_size,
        momentum=0.0,
        backprop_last=None,
    ):
        """Gradient-descent sampling on the learned landscape.

        momentum > 0 gives Nesterov-accelerated GD (paper Eq. 9).
        backprop_last=k keeps only the final k descent steps in the autograd
        graph; the rest run under no_grad.  With None, everything is kept
        (the original, very expensive, behaviour).
        """
        actions = self.clamp_actions(init_actions.clone())
        prev_actions = actions
        no_grad_steps = 0 if backprop_last is None else max(0, steps - backprop_last)

        for i in range(steps):
            # nullcontext (not enable_grad) so that an outer torch.no_grad()
            # at inference time is respected rather than overridden.
            ctx = torch.no_grad() if i < no_grad_steps else contextlib.nullcontext()
            with ctx:
                if momentum > 0.0:
                    probe = actions + momentum * (actions - prev_actions)
                    probe = self.clamp_actions(probe)
                else:
                    probe = actions
                states = self.rollout(model, init_states, probe)
                grad = self.forward(states, target_trajectories, probe)
                prev_actions = actions
                actions = self.clamp_actions(actions - step_size * grad)

        states = self.rollout(model, init_states, actions)
        return actions, states


# --------------------------------------------------------------------------- #
# Losses / data
# --------------------------------------------------------------------------- #
def c_truncated(gamma, a=0.8, lam=1.0):
    """Truncated decay from the paper (Eq. 5), the best-performing c(gamma).

    Constant lam for gamma <= a, then linearly to 0 at gamma = 1.
    """
    return lam * torch.where(gamma <= a, torch.ones_like(gamma), (1 - gamma) / (1 - a))


def state_error(states, trajectories):
    pos_error = torch.norm(states[:, :, :2] - trajectories[:, :, :2], dim=-1).mean(dim=1)

    theta, theta_t = states[:, :, 2], trajectories[:, :, 2]
    heading_loss = (2.0 - 2.0 * torch.cos(theta - theta_t)).mean(dim=1)

    # Steer is a bounded angle in [-max_steer, max_steer]; no wrap-around, so
    # plain squared error is the right metric here.
    steer_loss = (states[:, :, 4] - trajectories[:, :, 4]).pow(2).mean(dim=1)

    speed_error = torch.abs(states[:, :, 3] - trajectories[:, :, 3]).mean(dim=1)

    return pos_error + 0.5 * heading_loss + 0.5 * steer_loss + speed_error


def sample_maneuver_batch(batch_size, num_steps, total_time, device, mode=None,
                          model=None, randomize_accel_phase=False):
    """model: pass an existing BicycleModel so the sampler's physics matches
    the one used for rollouts.  Defaults to a fresh model with heading_mode
    "exact".

    randomize_accel_phase: the original code had `torch.zeros(B) * 2 * pi`,
    i.e. phase 0, so every vehicle accelerates first and builds speed.  That
    looks like a typo but it is load-bearing: with a random phase, ~64% of the
    batch brakes from an initial speed of 0.5-1.0 with amplitude 5-10, goes
    into reverse, and mean path length drops ~40%.  Left off by default.
    """
    dt = total_time / num_steps
    init = torch.zeros(batch_size, 5, device=device)
    if model is None:
        model = BicycleModel(max_steer_rad=1.0).to(device)

    init[:, 3] = torch.rand(batch_size, device=device) * 0.5 + 0.5
    init[:, 4] = torch.rand(batch_size, device=device) * 0.4 - 0.2

    t = torch.arange(num_steps, device=device).float() / num_steps

    num_cycles_accel = torch.rand(batch_size, device=device) * 0.5 + 0.5
    amp_accel = torch.rand(batch_size, device=device) * 5.0 + 5.0
    if randomize_accel_phase:
        phase_accel = torch.rand(batch_size, device=device) * 2 * math.pi
    else:
        phase_accel = torch.zeros(batch_size, device=device)
    accel = amp_accel[:, None] * torch.sin(
        2 * math.pi * num_cycles_accel[:, None] * t + phase_accel[:, None]
    )

    num_cycles_steer = torch.rand(batch_size, device=device) * 0.5 + 0.5
    amp_steer = torch.rand(batch_size, device=device) * 0.5 + 0.5
    phase_steer = torch.rand(batch_size, device=device) * 2 * math.pi
    steer = amp_steer[:, None] * torch.cos(
        2 * math.pi * num_cycles_steer[:, None] * t + phase_steer[:, None]
    )

    actions = torch.stack([accel, steer], dim=-1)
    bounds = torch.tensor([10.0, 1.0], device=device)
    actions = torch.clamp(actions, -bounds, bounds)
    actions = actions + torch.randn_like(actions) * 0.005

    state = init
    states = []
    for s in range(num_steps):
        state = model.step(state, actions[:, s], dt=dt)
        states.append(state)
    target_trajectories = torch.stack(states, dim=1)  # (B, T, 5)

    d_heading = (target_trajectories[:, -1, 2] - init[:, 2]).abs().mean()
    logger.debug(
        "Average total heading change: {r:.3f} rad ({d:.1f} deg) | mean path {p:.2f}",
        r=d_heading, d=d_heading * 180 / math.pi,
        p=(target_trajectories[:, 1:, :2] - target_trajectories[:, :-1, :2]).norm(dim=-1).sum(1).mean(),
    )

    if mode is not None:
        logger.warning("Mode ignored; using sinusoidal generation for all.")

    return init, target_trajectories, actions


def train_controller_eqm(
    batch_size,
    num_steps,
    total_time,
    epochs,
    sim_steps,
    step_size,
    noise_std=0.1,
    pose_weight=1.0,
    backprop_last=1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")

    dt = total_time / num_steps
    model = BicycleModel(max_steer_rad=1.0).to(device)
    policy = EqMPolicy(num_steps=num_steps, total_time=total_time, d_model=256,
                       n_heads=8, n_layers=6).to(device)
    logger.info("Policy params: {n:,}", n=sum(p.numel() for p in policy.parameters()))

    policy_path = "eqm_policy.pth"
    if os.path.exists(policy_path):
        try:
            policy.load_state_dict(torch.load(policy_path, map_location=device))
            logger.info("Resumed policy from {p}", p=policy_path)
        except RuntimeError as e:
            logger.warning("Ignoring incompatible checkpoint {p}: {e}", p=policy_path, e=e)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=0.0)
    scale = policy.action_scale

    for ep in range(1, epochs + 1):
        init_states, target_trajectories, target_actions = sample_maneuver_batch(
            batch_size, num_steps, total_time, device
        )

        # ---- EqM corruption -------------------------------------------------
        gamma = torch.rand(batch_size, 1, device=device)
        # Noise lives in action units so that eps and u* share a scale.
        noise = torch.randn_like(target_actions) * scale * noise_std
        u_gamma = gamma[:, :, None] * target_actions + (1 - gamma[:, :, None]) * noise
        c_gamma = c_truncated(gamma)
        target_grad = (noise - target_actions) * c_gamma[:, :, None]

        states = policy.rollout(model, init_states, u_gamma)
        grad = policy(states, target_trajectories, u_gamma)
        # Normalised so accel (~10) and steer (~1) contribute comparably.
        eqm_loss = (((grad - target_grad) / scale) ** 2).mean()

        # ---- Optional rollout / pose loss -----------------------------------
        if pose_weight > 0.0:
            actions, pred_states = policy.simulate_trajectory(
                model, init_states, target_trajectories, u_gamma,
                sim_steps, step_size, backprop_last=backprop_last,
            )
            pose_loss = state_error(pred_states, target_trajectories).mean()
        else:
            pose_loss = torch.zeros((), device=device)

        loss = eqm_loss + pose_weight * pose_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        if ep % 10 == 0 or ep <= 5:
            logger.info(
                "ep {ep:04d} | loss {l:.4f} | eqm {el:.4f} | pose {pl:.4f}",
                ep=ep, l=loss.item(), el=eqm_loss.item(), pl=pose_loss.item(),
            )
            torch.save(policy.state_dict(), policy_path)

    return model, policy


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    torch.manual_seed(0)
    model, policy = train_controller_eqm(
        batch_size=64,
        num_steps=75,
        total_time=1.0,
        epochs=10000,
        sim_steps=20,
        step_size=0.1,
    )