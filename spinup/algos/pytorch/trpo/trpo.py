import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
import time
from spinup.algos.pytorch.trpo import core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

EPS = 1e-8


class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # One row per timestep of the *old* policy's distribution parameters.
        # TRPO needs these to hold the KL constraint's left hand side fixed
        # while the line search moves the policy around.
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k,v in info_shapes.items()}
        self.sorted_info_keys = core.keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[i]
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.

        Returns a dict of tensors. The old policy's parameters come back under
        their own keys ('logp_all', or 'mu' and 'log_std'), which is what gets
        handed to the actor's kl().
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        data = dict(obs=self.obs_buf, act=self.act_buf, adv=self.adv_buf,
                    ret=self.ret_buf, logp_old=self.logp_buf)
        data.update(self.info_bufs)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def trpo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='trpo'):
    """
    Trust Region Policy Optimization

    (with support for Natural Policy Gradient)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, a ``pi`` module, and a ``v`` module, plus an
            ``info_shapes`` property. The ``step`` method should accept a batch
            of observations and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ``info``     N/A               | List of numpy arrays holding the
                                           | policy's distribution parameters,
                                           | ordered by sorted key.
            ===========  ================  ======================================

            The ``pi`` module must additionally provide:

            ============  =====================================================
            Method        Description
            ============  =====================================================
            ``log_prob``  | (obs, act) -> log probs, without sampling.
            ``kl``        | (obs, old_info) -> mean KL(old || current).
            ============  =====================================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update.
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be
            smallish. Adjusts Hessian-vector product calculation:

            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient.
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.
            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the
            backtracking line search. Since the line search usually doesn't
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        algo: Either 'trpo' or 'npg': this code supports both, since they are
            almost the same.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Experience buffer. The policy declares the shapes of its own distribution
    # parameters, so we don't have to reverse engineer them from the graph.
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch,
                    ac.info_shapes, gamma, lam)

    def mpi_avg_torch(t):
        """Average a flat torch vector across MPI processes."""
        if num_procs() == 1:
            return t
        return torch.as_tensor(mpi_avg(t.detach().numpy()), dtype=torch.float32)

    # Set up function for computing TRPO policy loss. This has to be a function
    # rather than a value, because the line search re-evaluates it at every
    # candidate set of parameters.
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp_old']
        logp = ac.pi.log_prob(obs, act)
        ratio = torch.exp(logp - logp_old)      # pi(a|s) / pi_old(a|s)
        return -(ratio * adv).mean()

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Optimizer for the value function only. The policy is moved by the natural
    # gradient step, so it has no optimizer.
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = torch.zeros_like(b)
        r = b.clone() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.clone()
        r_dot_old = torch.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (torch.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = torch.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        data = buf.get()
        obs = data['obs']

        # The old policy's distribution, frozen at the values collected during
        # the rollout. These tensors carry no gradient, so the KL only ever
        # differentiates through the current policy.
        old_info = {k: data[k] for k in ac.info_shapes}

        pi_params = list(ac.pi.parameters())

        # Gradient of the surrogate loss, i.e. the vanilla policy gradient g
        loss_pi_old = compute_loss_pi(data)
        g = mpi_avg_torch(core.flat_grad(loss_pi_old, pi_params))
        pi_l_old = mpi_avg(loss_pi_old.item())
        v_l_old = compute_loss_v(data).item()

        # Build the KL's first-order graph once, then reuse it for every
        # Hessian-vector product. d_kl is 0 at the current params and so is its
        # gradient, but its *second* derivative is the Fisher information
        # matrix, which is exactly the metric TRPO needs.
        d_kl = ac.pi.kl(obs, old_info)
        kl_grad = core.flat_grad(d_kl, pi_params, create_graph=True)

        def Hx(x):
            hvp = core.flat_grad((kl_grad * x).sum(), pi_params, retain_graph=True)
            hvp = mpi_avg_torch(hvp)
            if damping_coeff > 0:
                hvp = hvp + damping_coeff * x
            return hvp

        # Core calculations for TRPO or NPG.
        # x solves Hx = g, then alpha scales it to sit exactly on the KL
        # boundary: the quadratic approximation of the KL at a step alpha*x is
        # 0.5 * alpha^2 * x'Hx, and setting that equal to delta gives alpha.
        # Every Hx call must happen before the line search, since set_and_eval
        # mutates the parameters that kl_grad's graph refers to.
        x = cg(Hx, g)
        alpha = torch.sqrt(2 * delta / (torch.dot(x, Hx(x)) + EPS))
        old_params = core.get_flat_params_from(ac.pi)

        def set_and_eval(step):
            # NOTE: this overwrites the policy in place, which is why old_info
            # had to be saved during the rollout: after this call the old policy
            # is no longer recoverable from the network.
            core.set_flat_params_to(ac.pi, old_params - alpha * x * step)
            with torch.no_grad():
                kl = ac.pi.kl(obs, old_info).item()
                loss_pi = compute_loss_pi(data).item()
            return mpi_avg(kl), mpi_avg(loss_pi)

        if algo == 'npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif algo == 'trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff**j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    logger.log('Accepting new params at step %d of line search.'%j)
                    logger.store(BacktrackIters=j)
                    break

                if j == backtrack_iters - 1:
                    logger.log('Line search failed! Keeping old params.')
                    logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

        # Value function learning
        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        v_l_new = compute_loss_v(data).item()

        # Log changes from update
        logger.store(LossPi=pi_l_old, LossV=v_l_old, KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset()[0], 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp, info = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, terminated, truncated, _ = env.step(a)
            d = terminated
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp, info)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o

            timeout = truncated or (ep_len == max_ep_len)
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset()[0], 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform TRPO or NPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('KL', average_only=True)
        if algo == 'trpo':
            logger.log_tabular('BacktrackIters', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v3')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
         logger_kwargs=logger_kwargs)
