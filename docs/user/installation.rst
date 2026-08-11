============
Installation
============


.. contents:: Table of Contents

Spinning Up requires Python (3.10--3.12) and `Gymnasium`_. This modernized fork
manages its dependencies with `uv`_ and defaults to the **PyTorch** backend.
OpenMPI is optional and only needed to run with more than one process.

Spinning Up is currently only supported on Linux and OSX. It may be possible to
install on Windows, though this hasn't been extensively tested. [#]_

.. admonition:: You Should Know

    Many examples and benchmarks in Spinning Up refer to RL environments that use the `MuJoCo`_ physics engine. MuJoCo used to be proprietary software that required a paid license, but it was open-sourced by DeepMind in 2021 and is now free to use. Installing it is optional, but because of its importance to the research community---it is the de facto standard for benchmarking deep RL algorithms in continuous control---it is preferred.

    Don't worry if you decide not to install MuJoCo, though. You can definitely get started in RL by running RL algorithms on the `Classic Control`_ and `Box2d`_ environments in Gymnasium, which are totally free to use.

.. [#] It looks like at least one person has figured out `a workaround for running on Windows`_. If you try another way and succeed, please let us know how you did it!

.. _`Classic Control`: https://gymnasium.farama.org/environments/classic_control/
.. _`Box2d`: https://gymnasium.farama.org/environments/box2d/
.. _`MuJoCo`: https://gymnasium.farama.org/environments/mujoco/
.. _`Gymnasium`: https://gymnasium.farama.org/
.. _`uv`: https://docs.astral.sh/uv/
.. _`a workaround for running on Windows`: https://github.com/openai/spinningup/issues/23

Installing uv
=============

This fork uses `uv`_ to manage the Python interpreter, the virtual environment,
and all dependencies (pinned in ``pyproject.toml`` and ``uv.lock``). You do not
need to install Python or create an environment yourself---uv does both.

Install uv with the official standalone installer:

.. parsed-literal::

    curl -LsSf https://astral.sh/uv/install.sh | sh

See the `uv installation docs`_ for alternatives (Homebrew, pipx, etc.).

.. admonition:: You Should Know

    If you're new to Python environments and package management, the short
    version is: uv creates an isolated ``.venv`` for this project so its
    dependencies don't collide with anything else on your system. You run
    project commands by prefixing them with ``uv run`` (e.g. ``uv run python
    ...``), which automatically uses that environment.

.. _`uv installation docs`: https://docs.astral.sh/uv/getting-started/installation/


Installing Spinning Up
======================

.. parsed-literal::

    git clone https://github.com/openai/spinningup.git
    cd spinningup
    uv sync

``uv sync`` creates the virtual environment and installs Spinning Up together
with its core dependencies (PyTorch, NumPy, Gymnasium, and the plotting stack).

.. admonition:: You Should Know

    ``uv sync`` installs Gymnasium's base environments, which include the
    `Classic Control`_ suite (such as ``CartPole-v1``). The `Box2d`_ and
    `MuJoCo`_ environment families, as well as multi-process training with MPI,
    are optional and installed as extras (see below).


Optional Extras
===============

Install optional dependency groups with ``uv sync --extra <name>`` (extras can
be combined, e.g. ``uv sync --extra mujoco --extra mpi``):

- ``mujoco`` --- MuJoCo environments (e.g. ``HalfCheetah-v4``, ``Walker2d-v4``), via ``gymnasium[mujoco]``.
- ``box2d`` --- Box2D environments (e.g. ``LunarLander-v3``). Requires SWIG on the system (``brew install swig`` / ``apt install swig``).
- ``atari`` --- Atari environments via ``ale-py``.
- ``mpi`` --- Multi-process training (``num_cpu > 1``). Also needs a system MPI runtime (see below).


Installing OpenMPI (Optional)
=============================

MPI is only required to train with more than one process (``num_cpu > 1``).
Single-process training works without it. To enable multi-process training,
install a system MPI runtime and the ``mpi`` extra.

Ubuntu
------

.. parsed-literal::

    sudo apt-get update && sudo apt-get install libopenmpi-dev
    uv sync --extra mpi

Mac OS X
--------
Installation of system packages on Mac requires Homebrew_. With Homebrew installed, run the following:

.. parsed-literal::

    brew install open-mpi
    uv sync --extra mpi

.. _Homebrew: https://brew.sh


Check Your Install
==================

To see if you've successfully installed Spinning Up, try running PPO in the
``CartPole-v1`` environment with

.. parsed-literal::

    uv run python -m spinup.run ppo --hid "[32,32]" --env CartPole-v1 --exp_name installtest --gamma 0.999

This won't train the agent to completion, but will run it for long enough that
you can see *some* learning progress when the results come in.

After it finishes training, watch a video of the trained policy with

.. parsed-literal::

    uv run python -m spinup.run test_policy data/installtest/installtest_s0

And plot the results with

.. parsed-literal::

    uv run python -m spinup.run plot data/installtest/installtest_s0


Using MuJoCo (Optional)
=======================

MuJoCo is now free and open-source, and Gymnasium ships bindings for it---there
is no license to obtain and no separate ``mujoco-py`` install. Just add the
``mujoco`` extra:

.. parsed-literal::

    uv sync --extra mujoco

Then check that things are working by running PPO in the ``Walker2d-v4`` environment with

.. parsed-literal::

    uv run python -m spinup.run ppo --hid "[32,32]" --env Walker2d-v4 --exp_name mujocotest


.. admonition:: You Should Know

    This fork targets `Gymnasium`_ (the maintained successor to the original
    OpenAI Gym), so environment IDs use current versions---for example
    ``HalfCheetah-v4`` rather than ``HalfCheetah-v2``, and ``CartPole-v1``
    rather than ``CartPole-v0``. The legacy TensorFlow 1.x backend is not
    installed; every algorithm runs on PyTorch by default.
