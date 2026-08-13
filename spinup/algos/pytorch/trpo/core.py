import numpy as np
import torch
import torch.nn as nn
import scipy.signal

# default policy builder depends on action space
# Box :
# ========= 
# in Box environment we choose a set of actions in a range
# for eg : 
# our actions can be a set of two continous values :
# (0.8475, -1.520)
# a robotic arm perhaps so we need some x,y direction (in above case)
# and then we 
# 
# Discrete :
# ==========
# In discrete environment we choose from a set of actions 
# there is only one output that's why we use categorical
# policy here
from gymnasium.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))

def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]

# Copied from exercise 1.1 solution
def gaussian_likelihood(x, mu, log_std):
    """
    Args:
        x: Tensor with shape [batch, dim]
        mu: Tensor with shape [batch, dim]
        log_std: Tensor with shape [batch, dim] or [dim]

    Returns:
        Tensor with shape [batch]
    """
    # Solution
    # Based on the diagonal gaussian policy
    # log likelyhood
    eq_1 = x - mu
    eq_2 = eq_1 / torch.exp(log_std)
    eq_2 = eq_2 ** 2
    eq_3 = eq_2 + 2 * log_std
    result = -0.5 * (eq_3 + np.log(2 * np.pi))
    return result.sum(axis = -1)

class MLP(nn.Module):
    """
    Common MLP module
    """
    def __init__(
        self,
        input_dim,
        hidden_sizes=(64, 64),
        activation=nn.Tanh,
        output_activation=None
    ):
        super().__init__()

        layers = []
        dims = [input_dim] + list(hidden_sizes)

        for i in range(len(dims) - 1):
            layers.append(
                nn.Linear(dims[i], dims[i+1])
            )
            if i < len(dims) - 2:
                layers.append(activation())

        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CategoricalActor(nn.Module):
    """
    Categorical Actor Module

    This module is for categorical policy for action
    (used in Discrete env)
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.act_dim = act_dim

        self.net = MLP(
            obs_dim,
            (*hidden_sizes, act_dim)
        )

    @property
    def info_shapes(self):
        """
        Per-timestep shapes of the policy parameters that TRPO has to remember
        from *before* the update, so the KL constraint can be evaluated against
        the old policy. Used to size the info buffers in GAEBuffer.
        """
        return {'logp_all': [self.act_dim]}

    def forward(self, x, a=None):
        """
        Categorical Actor inference function

        x: observation input vector
        a: taken action

        returns :

        pi: sampled action from policy
        logp: getting the probability of the input action that was already taken from sample
        log_pi: getting the probability of the sampled action by inference of net
        info: dict of the policy parameters (log probs over every action)
        """
        logits = self.net(x)
        # distribution of actions
        dist = torch.distributions.Categorical(logits=logits)
        # sampled value (action from the policy)
        pi = dist.sample()

        logp = None
        if a is not None:
            # now getting the action by multiplying the one hot vector
            logp = dist.log_prob(a)

        # log probability of sampled action
        log_pi = dist.log_prob(pi)

        # normalised log probs over the whole action set, this is what the
        # categorical KL needs on both sides
        logp_all = torch.log_softmax(logits, dim=-1)

        return (pi, logp, log_pi, {'logp_all': logp_all})

    def kl(self, x, old_info):
        """
        Mean KL(old || new) between the stored old policy and the current one.

        Kept separate from forward() so the CG solve and the backtracking line
        search can re-evaluate the constraint without sampling actions they
        would only throw away.
        """
        logp_all = torch.log_softmax(self.net(x), dim=-1)
        return categorical_kl(logp_all, old_info['logp_all'])

    def log_prob(self, x, a):
        """
        log pi(a|x) for actions that were already taken. Same reason as kl():
        the line search only wants this, not a fresh sample.
        """
        return torch.distributions.Categorical(logits=self.net(x)).log_prob(a)

class GaussianActor(nn.Module):
    """
    Guassian Actor Module

    This module is for guassian action policy
    (used in Box env)
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.act_dim = act_dim

        self.mu_net = MLP(
            obs_dim,
            (*hidden_sizes, act_dim)
        )

        self.log_std = nn.Parameter(
            torch.full((act_dim,), -0.5)
        )

    @property
    def info_shapes(self):
        """
        Per-timestep shapes of the policy parameters that TRPO has to remember
        from *before* the update, so the KL constraint can be evaluated against
        the old policy. Used to size the info buffers in GAEBuffer.
        """
        return {'mu': [self.act_dim], 'log_std': [self.act_dim]}

    def forward(self, x, a=None):
        """
        Gaussian Actor inference function

        x: observation input vector
        a: taken action

        returns :

        pi: sampled action vector from policy
        logp: guassian likelihood of the input action
        log_pi: guassian likelihood of the sampled action
        info: dict of the policy parameters (mean and log std)
        """
        mu = self.mu_net(x)
        std = torch.exp(self.log_std)

        # sampled value (action from policy)
        normal_dist = torch.randn_like(mu)
        pi = mu + normal_dist * std

        # calculating the guassian likelihood for input action
        logp = None
        if a is not None:
            logp = gaussian_likelihood(
                a,
                mu,
                self.log_std
            )

        # calculating guassian likelihood for sampled action
        log_pi = gaussian_likelihood(pi, mu, self.log_std)

        # log_std is state independent, but the info buffer stores one row per
        # timestep, so broadcast it out to match mu
        info = {'mu': mu, 'log_std': self.log_std.expand_as(mu)}

        return (pi, logp, log_pi, info)

    def kl(self, x, old_info):
        """
        Mean KL(old || new) between the stored old policy and the current one.

        Kept separate from forward() so the CG solve and the backtracking line
        search can re-evaluate the constraint without drawing samples they
        would only throw away.
        """
        mu = self.mu_net(x)
        log_std = self.log_std.expand_as(mu)
        return diagonal_gaussian_kl(
            mu, log_std,
            old_info['mu'], old_info['log_std']
        )

    def log_prob(self, x, a):
        """
        log pi(a|x) for actions that were already taken. Same reason as kl():
        the line search only wants this, not a fresh sample.
        """
        return gaussian_likelihood(a, self.mu_net(x), self.log_std)

class MLPCritic(nn.Module):
    """
    Value function module, shares the MLP body shape with the actor
    """
    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()

        self.v_net = MLP(
            obs_dim,
            (*hidden_sizes, 1)
        )

    def forward(self, x):
        # squeeze so v has shape [batch] rather than [batch, 1]
        return torch.squeeze(self.v_net(x), -1)

class MLPActorCritic(nn.Module):
    """
    Actor critic module

    Picks the policy that matches the action space, and pairs it with a
    value function. This is what gets passed to trpo() as actor_critic.
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(64, 64)):
        super().__init__()

        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.pi = GaussianActor(obs_dim, action_space.shape[0], hidden_sizes)
        elif isinstance(action_space, Discrete):
            self.pi = CategoricalActor(obs_dim, action_space.n, hidden_sizes)
        else:
            raise NotImplementedError(
                f"unsupported action space: {action_space}"
            )

        self.v = MLPCritic(obs_dim, hidden_sizes)

    @property
    def info_shapes(self):
        """
        Forwarded from the policy so trpo() can size the GAEBuffer info buffers
        without caring which policy got built.
        """
        return self.pi.info_shapes

    def forward(self, obs, act=None):
        pi, logp, logp_pi, info = self.pi(obs, act)
        v = self.v(obs)
        return pi, logp, logp_pi, info, v

    def step(self, obs):
        with torch.no_grad():
            pi, _, logp_pi, info, v = self(obs)
        # sorted list, to match how GAEBuffer.store indexes info.
        # detach because the gaussian log_std entry is a view straight onto the
        # nn.Parameter, so it still requires grad even under no_grad
        info = [info[k].detach().numpy() for k in keys_as_sorted_list(info)]
        return pi.numpy(), v.numpy(), logp_pi.numpy(), info

    def act(self, obs):
        """
        Just the action. This is what spinup.utils.test_policy calls when
        replaying a trained policy.
        """
        return self.step(obs)[0]

def diagonal_gaussian_kl(mu0, log_std0, mu1, log_std1, eps=1e-8):
    var0 = torch.exp(2 * log_std0)
    var1 = torch.exp(2 * log_std1)
    pre_sum = (
        0.5 * (
            ((mu1 - mu0) ** 2 + var0) / (var1 + eps)
            - 1
        )
        + log_std1
        - log_std0
    )
    all_kls = torch.sum(pre_sum, dim=1)
    return torch.mean(all_kls)

def categorical_kl(logp0, logp1):
    all_kls = torch.sum(
        torch.exp(logp1) * (logp1 - logp0),
        dim=1
    )
    return torch.mean(all_kls)

def flat_concat(xs):
    """
    Flatten a list of tensors into one 1-D vector.

    TRPO does all of its linear algebra (CG, the step direction, the line
    search) on the policy parameters as a single flat vector, so we need to be
    able to move back and forth between that view and the per-module tensors.
    """
    return torch.cat([x.reshape(-1) for x in xs])

def flat_grad(f, params, retain_graph=False, create_graph=False):
    """
    Gradient of scalar f w.r.t. params, returned as one flat vector.

    create_graph=True keeps the gradient itself differentiable, which is what
    makes the Hessian-vector product possible.
    """
    if create_graph:
        retain_graph = True
    # materialise, params is usually a generator and we index it more than once
    params = list(params)
    grads = torch.autograd.grad(
        f, params,
        retain_graph=retain_graph,
        create_graph=create_graph
    )
    return flat_concat(grads)

def get_flat_params_from(module):
    """
    Read the module's parameters out as one flat vector (torch analogue of
    tf's flat_concat(get_vars('pi'))).
    """
    return flat_concat([p.data for p in module.parameters()])

def set_flat_params_to(module, flat_params):
    """
    Write a flat vector back into the module's parameters, in place.

    This is the torch analogue of assign_params_from_flat. The line search
    calls it repeatedly to try candidate steps, which is exactly why the old
    policy's distribution has to be kept in the buffer: after the first call
    the old parameters are gone.
    """
    idx = 0
    with torch.no_grad():
        for p in module.parameters():
            n = p.numel()
            p.copy_(flat_params[idx:idx + n].view_as(p))
            idx += n

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
