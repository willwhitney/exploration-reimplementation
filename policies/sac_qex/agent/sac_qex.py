import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# import ipdb; ipdb.set_trace()
from .. import sac_utils as utils
from . import Agent
from . import actor
from . import critic


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_args,
                 actor_args, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, num_seed_steps,
                 bonus_scale=1):
        super().__init__()

        self.action_dim = action_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.num_seed_steps = num_seed_steps
        self.bonus_scale = bonus_scale
        self.step = 0

        self.critic = critic.DoubleQCritic(**critic_args).to(self.device)
        self.critic_target = critic.DoubleQCritic(**critic_args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.exploration_critic = critic.DoubleQCritic(**critic_args).to(self.device)
        self.exploration_critic_target = critic.DoubleQCritic(**critic_args).to(self.device)
        self.exploration_critic_target.load_state_dict(self.exploration_critic.state_dict())

        self.actor = actor.DiagGaussianActor(**actor_args).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.exploration_critic_optimizer = torch.optim.Adam(
            self.exploration_critic.parameters(),
            lr=critic_lr, betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def act_samples(self, obs, n=1, sample=False):
        if self.step <= self.num_seed_steps:
            actions = np.random.uniform(
                size=(obs.shape[0], n, self.action_dim,),
                low=self.action_range[0],
                high=self.action_range[1])
            return actions, 0
        obs = torch.FloatTensor(obs).to(self.device)
        # obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        if sample:
            actions = dist.sample((n,))
            entropy = - dist.log_prob(actions).sum(axis=-1).mean().item()
        else:
            actions = dist.mean.unsqueeze(0)
            actions = actions.expand((n, *actions.shape[1:]))
            entropy = 0
        actions = actions.transpose(0, 1)
        actions = actions.clamp(*self.action_range)

        assert actions.ndim == 3
        return utils.to_np(actions), entropy

    def update_critic(self, obs, action, reward, next_obs):
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        # logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_exploration_critic(self, obs, action, bonus, next_obs):
        obs = torch.FloatTensor(obs).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        bonus = torch.FloatTensor(bonus).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.exploration_critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = bonus + (self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.exploration_critic(obs, action)
        exploration_critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        # logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.exploration_critic_optimizer.zero_grad()
        exploration_critic_loss.backward()
        self.exploration_critic_optimizer.step()

        # self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_exploration_Q1, actor_exploration_Q2 = self.exploration_critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_exploration_Q = torch.min(actor_exploration_Q1, actor_exploration_Q2)
        actor_sum_Q = actor_Q + self.bonus_scale * actor_exploration_Q
        actor_loss = (self.alpha.detach() * log_prob - actor_sum_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, transitions, bonus_transitions):
        self.step += 1
        if self.step >= self.num_seed_steps:
            obs, action, next_obs, bonus = bonus_transitions
            self.update_exploration_critic(obs, action, bonus, next_obs)

            obs, action, next_obs, reward = transitions
            self.update_critic(obs, action, reward, next_obs)

            if self.step % self.actor_update_frequency == 0:
                self.update_actor_and_alpha(obs)

            if self.step % self.critic_target_update_frequency == 0:
                utils.soft_update_params(self.critic, self.critic_target,
                                        self.critic_tau)
                utils.soft_update_params(self.exploration_critic,
                                         self.exploration_critic_target,
                                         self.critic_tau)
