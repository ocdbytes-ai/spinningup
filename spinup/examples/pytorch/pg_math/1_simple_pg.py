import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# creates mlp 
# suppose if sizes = [5 (input), 32 (hidden), 3 (num actions)]
# iter : len(sizes) - 1 : j = 0, 1, 2
# if j < 2 --> activation
# else --> no activation
# return neural net
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    # render_mode must be set at creation time in gymnasium; "human" opens a live window
    env = gym.make(env_name, render_mode="human" if render else None)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    # each observation : vec of current state ig ? 
    # observation dimension = n inputs x m variables
    obs_dim = env.observation_space.shape[0] # n inputs
    n_acts = env.action_space.n # number of actions

    # make core of policy network
    # makes a mlp of architecture
    # input layer (observation) --> hidden layers ---> output layer (softmax over actions ??)
    # hidden layers can be any size and multiple (default : one layer of size 32)
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    # Does one forward pass to get the logits for the output
    # wraps those logits into Categorical distribution
    # Categorical holds the raw logits and computes the softmax or sampling when required !!!
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    # here we fetch the policy from the neural net forward pass 
    # now when we call sample here then the probabilities are calculated using softmax 
    # and then we get the item based on the probabilities calculated
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    # here we get the policy fro mneural net forward pass and then we 
    # call log_prob on the action 
    # action here is the index in the Categorical distribution
    # logp = logit_i - logsumexp(logits) (this is somewhat same as log(softmax) but this is more numerically stable)
    # we will do element wise multiplication on the two vectors :
    # weight and logp and then calculated the mean value in resultant vector
    # negative sign is for the direction of loss
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()    # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            observation_tensor = torch.as_tensor(np.array(obs), dtype=torch.float32)
            act = get_action(observation_tensor)
            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset()[0], False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
                                  act=torch.as_tensor(np.array(batch_acts), dtype=torch.int32),
                                  weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

    # save the trained policy weights so they can be replayed with enjoy.py
    # we also stash hidden_sizes so enjoy.py can rebuild the exact architecture
    save_path = '%s_policy.pt' % env_name.replace('/', '_')
    torch.save({'state_dict': logits_net.state_dict(),
                'hidden_sizes': hidden_sizes,
                'obs_dim': obs_dim,
                'n_acts': n_acts}, save_path)
    print('\nSaved trained policy to %s' % save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)