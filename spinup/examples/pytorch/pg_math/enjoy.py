import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
import gymnasium as gym


# same mlp builder as the training script, so the loaded weights fit exactly
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def enjoy(env_name='CartPole-v1', episodes=5, render=True, deterministic=True):
    # human render_mode opens a live window (pass --no-render to run headless)
    env = gym.make(env_name, render_mode='human' if render else None)

    # load the checkpoint written by 1_simple_pg.py
    ckpt_path = '%s_policy.pt' % env_name.replace('/', '_')
    ckpt = torch.load(ckpt_path, weights_only=False)

    # rebuild the exact architecture that was trained, then load the weights
    sizes = [ckpt['obs_dim']] + ckpt['hidden_sizes'] + [ckpt['n_acts']]
    logits_net = mlp(sizes=sizes)
    logits_net.load_state_dict(ckpt['state_dict'])
    logits_net.eval()  # inference mode
    print('Loaded policy from %s (architecture: %s)' % (ckpt_path, sizes))

    for episode in range(episodes):
        obs, _ = env.reset()
        done, ep_ret, ep_len = False, 0.0, 0
        while not done:
            with torch.no_grad():  # no gradients needed at inference
                logits = logits_net(torch.as_tensor(np.array(obs), dtype=torch.float32))
                if deterministic:
                    action = torch.argmax(logits).item()      # greedy: policy's best guess
                else:
                    action = Categorical(logits=logits).sample().item()  # sample (as in training)
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += rew
            ep_len += 1
        print('episode %2d \t return: %.1f \t ep_len: %d' % (episode, ep_ret, ep_len))

    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--no-render', dest='render', action='store_false',
                        help='run headless (no window)')
    parser.add_argument('--stochastic', dest='deterministic', action='store_false',
                        help='sample actions instead of taking argmax')
    args = parser.parse_args()
    enjoy(env_name=args.env_name, episodes=args.episodes,
          render=args.render, deterministic=args.deterministic)
