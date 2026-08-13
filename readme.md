Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Installation (modernized fork)
------------------------------

This fork has been modernized to run on current Python and libraries. Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in `pyproject.toml` / `uv.lock`.

```bash
# install uv (see https://docs.astral.sh/uv/ for other methods)
curl -LsSf https://astral.sh/uv/install.sh | sh

# create the environment and install spinup + dependencies
uv sync

# run an algorithm (PyTorch backend)
uv run python -m spinup.run ppo --env CartPole-v1 --exp_name ppo-cartpole

# plot results / watch a trained agent
uv run python -m spinup.run plot data/ppo-cartpole
uv run python -m spinup.run test_policy data/ppo-cartpole
```

Optional extras (`uv sync --extra <name>`):

- `mpi` — multi-process training (`num_cpu > 1`). Also needs a system MPI runtime: `brew install open-mpi` (macOS) or `apt install libopenmpi-dev` (Debian/Ubuntu). Single-process training works without it.
- `mujoco` — MuJoCo environments (e.g. `HalfCheetah-v4`), via `gymnasium[mujoco]`.
- `box2d` — Box2D environments (e.g. `LunarLander-v3`); requires SWIG on the system.
- `atari` — Atari environments via `ale-py`.

### What changed in this fork

- **TRPO (PyTorch)**: Added TRPO pytorch implementation and works awesome :). Tried it on LunarLander v3
- **Dependencies**: migrated to `uv`; `torch`, `numpy`, `gymnasium`, `pandas`, `matplotlib`, `seaborn`, `scipy` bumped to current stable releases.
- **Gym → Gymnasium**: the code now targets [Gymnasium](https://gymnasium.farama.org/), including the 5-tuple `step()` (`terminated`/`truncated`) and tuple `reset()` APIs. Default environment IDs were updated (e.g. `HalfCheetah-v2` → `-v4`, `CartPole-v0` → `-v1`).
- **TensorFlow**: the legacy TF1 backend is not installed by default (TF1 is incompatible with modern Python). The TF1 source is kept but dormant; the **PyTorch backend is the default** for every algorithm. Requesting a `_tf1` variant raises a clear error.
- **MPI is optional**: without `mpi4py`/a system MPI, training runs single-process; `num_cpu > 1` raises a clear error telling you what to install.

Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```