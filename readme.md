## Introduction

This project is a simple bicycle model neural controller using a diffusion model inspired by [DynaFlow](https://arxiv.org/pdf/2509.19804) differential simulator
and [Equilibrium Matching](https://arxiv.org/pdf/2510.02300) gradient-based denoising.

The project manages to generate non-trivial trajectories reaching targets for the simple bicycle controller, but bear in mind, it's simulator-generated trajectories and the system is quite simple.


## Setup

Install `uv` if not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync `uv` project:
```bash
uv venv
source .venv/bin/activate
uv sync
```

To train use:
```bash
python -m eqmcontrol.train
```

Inference:
```bash
python -m eqmcontrol.inference
```

Generate data for visualization:
```bash
python -m eqmcontrol.view trajectory_data.pt
```

Visualize data:
```bash
rerun trajectory.rrd
```
