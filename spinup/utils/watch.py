"""
Watch a trained policy play in a live render window.

Why this exists instead of just using test_policy: spinup pickles the training
env into vars.pkl, but Gymnasium fixes the render mode when the env is built,
so the saved env comes back with render_mode=None and calling render() on it
raises. This takes only the policy from the save and builds a fresh, renderable
env around it.

Usage:
    python -m spinup.utils.watch data/trpo_lunar/trpo_lunar_s0
    python -m spinup.utils.watch data/trpo_cartpole/trpo_cartpole_s0 --env CartPole-v1
    python -m spinup.utils.watch <path> -n 10 -l 500 --norender
"""
import argparse

import gymnasium as gym

from spinup.utils.test_policy import load_policy_and_env, run_policy


def watch(fpath, env_id='LunarLander-v3', episodes=5, max_ep_len=1000,
          itr='last', deterministic=False, render=True):
    """
    Load a saved policy and run it in a freshly built env.

    Args:
        fpath (str): Path to an experiment output dir, e.g.
            ``data/trpo_lunar/trpo_lunar_s0``.

        env_id (str): Gymnasium id of the env to build. This cannot be read
            back from the save (the pickled env loses its ``spec``), so it has
            to be passed in.

        episodes (int): How many episodes to play.

        max_ep_len (int): Cut an episode off after this many steps.

        itr (int or 'last'): Which saved checkpoint to load.

        deterministic (bool): Use the deterministic action op. Only meaningful
            for SAC policies; ignored elsewhere.

        render (bool): Open a live window. False just prints returns.
    """
    # the saved env is deliberately discarded, see module docstring
    _, get_action = load_policy_and_env(fpath, itr, deterministic)

    env = gym.make(env_id, render_mode='human' if render else None)

    # Fail early and clearly if env_id doesn't match what the policy was
    # trained on, rather than deep inside the rollout with a shape error.
    try:
        get_action(env.observation_space.sample())
    except Exception as e:
        env.close()
        raise SystemExit(
            f"Policy in {fpath} does not accept observations from '{env_id}' "
            f"({env.observation_space}).\nPass the env it was trained on with "
            f"--env.\nUnderlying error: {type(e).__name__}: {e}"
        )

    try:
        run_policy(env, get_action, max_ep_len, episodes, render)
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('fpath', type=str,
                        help='experiment dir, e.g. data/trpo_lunar/trpo_lunar_s0')
    parser.add_argument('--env', '-e', type=str, default='LunarLander-v3',
                        help='Gymnasium env id to build (default: %(default)s)')
    parser.add_argument('--episodes', '-n', type=int, default=5)
    parser.add_argument('--len', '-l', type=int, default=1000,
                        help='max episode length')
    parser.add_argument('--itr', '-i', type=int, default=-1,
                        help='which checkpoint to load; -1 means latest')
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--norender', '-nr', action='store_true',
                        help='no window, just print episode returns')
    args = parser.parse_args()

    watch(args.fpath,
          env_id=args.env,
          episodes=args.episodes,
          max_ep_len=args.len,
          itr=args.itr if args.itr >= 0 else 'last',
          deterministic=args.deterministic,
          render=not args.norender)
