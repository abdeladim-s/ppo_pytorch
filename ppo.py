#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file contains the PPO algorithm
Copyright (C) <2021>  <Abdeladim S.>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
"""
import math
import os
from typing import Callable, Union
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import device, discounted_cumulative_sum, build_neural_net, standardize, numpy_to_torch_dtype_dict, \
    build_neural_net_general

from hyper_parameters import config

logger = logging.getLogger(__name__)


class Buffer:

    def __init__(self, state_shape: tuple, action_shape: tuple, buffer_size: int, gamma: float, gae_lambda: float):
        self.states = np.zeros(shape=(buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(shape=(buffer_size, *action_shape), dtype=np.float32)
        self.advantages = np.zeros(shape=buffer_size, dtype=np.float32)
        self.rewards = np.zeros(shape=buffer_size, dtype=np.float32)
        self.returns = np.zeros(shape=buffer_size, dtype=np.float32)
        self.state_values = np.zeros(shape=buffer_size, dtype=np.float32)
        self.log_probabilities = np.zeros(shape=buffer_size, dtype=np.float32)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size

        self.buffer_pointer = 0
        self.save_buffer_pointer = 0

    def append(self, state, action, reward, state_value, log_probability):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.buffer_pointer < self.buffer_size  # buffer has to have room so you can store

        self.states[self.buffer_pointer] = state
        self.actions[self.buffer_pointer] = action
        self.rewards[self.buffer_pointer] = reward
        self.state_values[self.buffer_pointer] = state_value
        self.log_probabilities[self.buffer_pointer] = log_probability
        self.buffer_pointer += 1

    def estimate_advantage(self, last_state_value=0):
        path_slice = slice(self.save_buffer_pointer, self.buffer_pointer)
        rewards = np.append(self.rewards[path_slice], last_state_value)
        state_values = np.append(self.state_values[path_slice], last_state_value)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * state_values[1:] - state_values[:-1]
        self.advantages[path_slice] = discounted_cumulative_sum(deltas, self.gamma * self.gae_lambda)

        # the next line computes rewards-to-go, to be targets for the value function
        self.returns[path_slice] = discounted_cumulative_sum(rewards, self.gamma)[:-1]

        self.save_buffer_pointer = self.buffer_pointer

    def get_buffer(self):
        # assert self.buffer_pointer == self.buffer_size  # buffer has to be full before you can get

        self.buffer_pointer = 0
        self.save_buffer_pointer = 0

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.advantages), np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (
                adv_std + 1e-8)  # 1e-8 : RuntimeWarning: invalid value encountered in true_divide self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(states=self.states, actions=self.actions, returns=self.returns,
                    advantages=self.advantages, log_probabilities=self.log_probabilities)
        return {k: torch.from_numpy(v).to(device) for k, v in data.items()}


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

    def distribution(self, state):
        raise NotImplementedError

    def log_probability_from_distribution(self, policy, actions):
        raise NotImplementedError

    def forward(self, state, actions=None):
        policy = self.distribution(state)
        log_probability = None
        if actions is not None:
            log_probability = self.log_probability_from_distribution(policy, actions)
        return policy, log_probability


class DiscreteActor(Actor):
    def __init__(self, state_size, action_size, agent_state_layer, agent_action_layer, agent_hidden_layers):
        super(DiscreteActor, self).__init__()
        self.net = build_neural_net_general(state_size, agent_state_layer, action_size, agent_action_layer,
                                            agent_hidden_layers)

    def distribution(self, state):
        logits = self.net(state)
        return Categorical(logits=logits)

    def log_probability_from_distribution(self, policy, actions):
        """

        :type policy: Categorical
        """
        return policy.log_prob(actions)


class ContinuousActor(Actor):
    def __init__(self, state_size, action_size, agent_state_layer, agent_action_layer, agent_hidden_layers, std):
        super(ContinuousActor, self).__init__()

        self.net = build_neural_net_general(state_size, agent_state_layer, action_size, agent_action_layer,
                                            agent_hidden_layers)
        self.log_std = torch.nn.Parameter(torch.as_tensor(-std * np.ones(action_size, dtype=np.float32), device=device))

    def distribution(self, state):
        mu = self.net(state)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_probability_from_distribution(self, policy, actions):
        return policy.log_prob(actions).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class Critic(nn.Module):
    def __init__(self, state_size, agent_state_layer, agent_action_layer, agent_hidden_layers):
        super(Critic, self).__init__()
        self.net = build_neural_net_general(state_size=state_size, agent_state_layer=agent_state_layer, action_size=1,
                                            agent_action_layer=agent_action_layer,
                                            agent_hidden_layers=agent_hidden_layers)

    def forward(self, state):
        return torch.squeeze(self.net(state), -1)  # Critical to ensure v has right shape.


class Agent:
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, agent_state_layer,
                 agent_action_layer, agent_hidden_layers,
                 actor_learning_rate: float, critic_learning_rate: float, std: float, checkpoint_dir: str):

        state_size = observation_space.shape[len(observation_space.shape) - 1]
        # state_size = math.prod(observation_space.shape)
        if isinstance(action_space, Box):
            action_size = action_space.shape[len(action_space.shape) - 1]
            self.actor = ContinuousActor(state_size=state_size,
                                         action_size=action_size, agent_action_layer=agent_action_layer,
                                         agent_state_layer=agent_state_layer, agent_hidden_layers=agent_hidden_layers,
                                         std=std)
        elif isinstance(action_space, Discrete):
            n = action_space.n
            self.actor = DiscreteActor(state_size=state_size,
                                       action_size=n, agent_action_layer=agent_action_layer,
                                       agent_state_layer=agent_state_layer, agent_hidden_layers=agent_hidden_layers,)

        self.critic = Critic(state_size=state_size, agent_state_layer=agent_state_layer,
                             agent_action_layer=agent_action_layer, agent_hidden_layers=agent_hidden_layers)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.actor_checkpoint_file = os.path.join(self.checkpoint_dir, 'actor.pt')
        self.critic_checkpoint_file = os.path.join(self.checkpoint_dir, 'critic.pt')
        self.training_info_checkpoint_file = os.path.join(self.checkpoint_dir, 'training_info.pt')

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        # load models if they exist
        self.model_id = None
        self.epoch = None
        self.steps_per_epoch = None
        self.total_episodes = None
        self.best_episode_return = None
        self.best_average_epoch_returns = None

        if os.path.exists(self.actor_checkpoint_file) and os.path.exists(self.critic_checkpoint_file):
            self.load_models()

    def step(self, state):
        with torch.no_grad():
            policy = self.actor.distribution(state)
            action = policy.sample()
            log_probability = self.actor.log_probability_from_distribution(policy, action)
            value = self.critic(state)
        return action.cpu().numpy(), value.cpu().numpy(), log_probability.cpu().numpy()

    def action(self, state):
        return self.step(state)[0]

    def save_training_info(self, env_name, model_id, epoch, steps_per_epoch, total_episodes, best_episode_return,
                           best_average_epoch_returns):
        torch.save({
            'env_name': env_name,
            'model_id': model_id,
            'epoch': epoch,
            'steps_per_epoch': steps_per_epoch,
            'total_episodes': total_episodes,
            'best_episode_return': best_episode_return,
            'best_average_epoch_returns': best_average_epoch_returns,
        }, self.training_info_checkpoint_file)

    def save_models(self):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, self.actor_checkpoint_file)

        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, self.critic_checkpoint_file)

    def load_models(self):
        #  load training info
        checkpoint = torch.load(self.training_info_checkpoint_file)
        self.model_id = checkpoint['model_id']
        self.epoch = checkpoint['epoch']
        self.steps_per_epoch = checkpoint['steps_per_epoch']
        self.total_episodes = checkpoint['total_episodes']
        self.best_episode_return = checkpoint['best_episode_return']
        self.best_average_epoch_returns = checkpoint['best_average_epoch_returns']

        # load actor
        checkpoint = torch.load(self.actor_checkpoint_file, map_location=torch.device(device))
        self.actor.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # load critic
        checkpoint = torch.load(self.critic_checkpoint_file, map_location=torch.device(device))
        self.critic.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def __str__(self):
        return f"""-- Agent Information -- 
        - Model id: {self.model_id}
        - Actor architecture: {self.actor}
        - Critic architecture : {self.critic}
        - Epoch: {self.epoch}
        - Steps per epoch: {self.steps_per_epoch}
        - Total number of episodes: {self.total_episodes}
        - Best episode return: {self.best_episode_return}
        - Best Average epoch returns: {self.best_average_epoch_returns}
        ----------------------------------"""


class Policy:
    # TODO: Default params
    # TODO: Generalize to more than one dim tensor

    def __init__(self,
                 env: Union[gym.Env, str, object],
                 model_id='defaults',
                 tensorboard=True,
                 debug=True,
                 render=False,
                 render_mode=None,
                 seed=config['defaults']['seed'],
                 epochs=config['defaults']['epochs'],
                 steps_per_epoch=config['defaults']['steps_per_epoch'],
                 max_episode_steps=config['defaults']['max_episode_steps'],
                 gradient_actor_loss_max_steps=config['defaults']['gradient_actor_loss_max_steps'],
                 gradient_critic_loss_max_steps=config['defaults']['gradient_critic_loss_max_steps'],
                 agent_state_layer=config['defaults']['agent_state_layer'],
                 agent_action_layer=config['defaults']['agent_action_layer'],
                 agent_hidden_layers=config['defaults']['agent_hidden_layers'],
                 agent_actor_learning_rate=config['defaults']['agent_actor_learning_rate'],
                 agent_critic_learning_rate=config['defaults']['agent_critic_learning_rate'],
                 agent_std=config['defaults']['agent_std'],
                 gamma=config['defaults']['gamma'],
                 gae_lambda=config['defaults']['gae_lambda'],
                 clip_ratio=config['defaults']['clip_ratio'],
                 break_training_if_solved=config['defaults']['break_training_if_solved'],
                 reward_threshold=config['defaults']['reward_threshold']
                 ):

        self.models_dir = config["general_configs"]["models_dir"]
        self.tensorboard_dir = config["general_configs"]["tensorboard_dir"]
        self.debug_dir = config["general_configs"]["debug_dir"]

        self.env = env if isinstance(env, gym.Env) else gym.make(env, render_mode=render_mode)
        self.model_id = model_id
        try:
            self.env_name = self.env.unwrapped.spec.id
        except:
            self.env_name = env if type(env) == str else 'Unknown'

        try:  # check if a custom model is defined in the hyper_parameters file
            self.params = config[self.env_name][model_id]
        except KeyError:
            self.params = {}

        self.render = render
        self.debug = debug
        if self.debug:
            self.setup_logger(log_file_name='debug')

        self.tensorboard = tensorboard

        self.seed = self.params['seed'] if 'seed' in self.params else seed
        self.epochs = self.params['epochs'] if 'epochs' in self.params else epochs
        self.steps_per_epoch = self.params['steps_per_epoch'] if 'steps_per_epoch' in self.params else steps_per_epoch
        try:
            # take max steps from the env itself
            self.max_episode_steps = self.env.unwrapped.spec.max_episode_steps
        except:  # probably does not exist
            self.max_episode_steps = self.params['max_episode_steps'] if 'max_episode_steps' in self.params \
                else max_episode_steps

        self.gradient_actor_loss_max_steps = self.params['gradient_actor_loss_max_steps'] if \
            'gradient_actor_loss_max_steps' in self.params else gradient_actor_loss_max_steps
        self.gradient_critic_loss_max_steps = self.params['gradient_critic_loss_max_steps'] if \
            'gradient_critic_loss_max_steps' in self.params else gradient_critic_loss_max_steps
        self.agent_hidden_layers = self.params['agent_hidden_layers'] if \
            'agent_hidden_layers' in self.params else agent_hidden_layers
        self.agent_state_layer = self.params['agent_state_layer'] if \
            'agent_state_layer' in self.params else agent_state_layer
        self.agent_action_layer = self.params['agent_action_layer'] if \
            'agent_action_layer' in self.params else agent_action_layer

        self.agent_actor_learning_rate = self.params['agent_actor_learning_rate'] if \
            'agent_actor_learning_rate' in self.params else agent_actor_learning_rate

        self.agent_critic_learning_rate = self.params['agent_critic_learning_rate'] if \
            'agent_critic_learning_rate' in self.params else agent_critic_learning_rate

        self.agent_std = self.params['agent_std'] if \
            'agent_std' in self.params else agent_std

        self.gamma = self.params['gamma'] if \
            'gamma' in self.params else gamma

        self.gae_lambda = self.params['gae_lambda'] if \
            'gae_lambda' in self.params else gae_lambda
        self.clip_ratio = self.params['clip_ratio'] if \
            'clip_ratio' in self.params else clip_ratio

        self.break_training_if_solved = self.params['break_training_if_solved'] if \
            'break_training_if_solved' in self.params else break_training_if_solved

        # reward_threshold
        # we will consider env solved when average epoch return is greater than or equal to the reward_threshold
        try:
            self.reward_threshold = self.env.unwrapped.spec.reward_threshold
            self.logger.debug(f"**Environment is considered solved if reward = {self.reward_threshold}**")
        except:
            self.reward_threshold = self.params['reward_threshold'] if 'reward_threshold' in self.params \
                else reward_threshold

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Create agent
        agent_dir = os.path.join(self.models_dir, self.env_name, model_id)
        os.makedirs(agent_dir, exist_ok=True)

        self.agent = Agent(observation_space=self.env.observation_space, action_space=self.env.action_space,
                           agent_state_layer=self.agent_state_layer, agent_action_layer=self.agent_action_layer,
                           agent_hidden_layers=self.agent_hidden_layers,
                           actor_learning_rate=self.agent_actor_learning_rate,
                           critic_learning_rate=self.agent_critic_learning_rate, std=self.agent_std,
                           checkpoint_dir=agent_dir)

        self.buffer = Buffer(self.env.observation_space.shape, self.env.action_space.shape, self.steps_per_epoch,
                             self.gamma, self.gae_lambda)

        self.logger.debug(self)

    def setup_logger(self, log_file_name=None):
        self.logger = logger
        os.makedirs((os.path.join(self.debug_dir, self.env_name, self.model_id)), exist_ok=True)
        if log_file_name is not None:
            handlers = [
                logging.FileHandler(os.path.join(self.debug_dir, self.env_name, self.model_id,
                                                 log_file_name + '-' + datetime.now().strftime(
                                                     "%d-%m-%Y_%H:%M:%S") + '.log')),
                logging.StreamHandler()
            ]
        else:
            handlers = [logging.StreamHandler()]

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=handlers
        )

    def setup_tensorboard(self):
        tb_log_dir = os.path.join(self.tensorboard_dir, self.env_name, self.model_id)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

    def actor_loss(self, data):
        states, actions, advantages, log_probabilities_old = data['states'], data['actions'], data['advantages'], data[
            'log_probabilities']
        # states, actions, advantages, log_probabilities_old =  data['states'], data['actions'], data['advantages'], data['log_probabilities']

        policy, log_probabilities = self.agent.actor(states, actions)
        ratio = torch.exp(log_probabilities - log_probabilities_old)
        clip = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        loss = -(torch.min(ratio * advantages, clip)).mean()

        return loss

    def critic_loss(self, data):
        states, returns = data['states'], data['returns']
        return ((self.agent.critic(states) - returns) ** 2).mean()

    def update(self):
        data = self.buffer.get_buffer()
        for i in range(self.gradient_actor_loss_max_steps):
            self.agent.actor_optimizer.zero_grad()
            policy_loss = self.actor_loss(data)
            policy_loss.backward()
            self.agent.actor_optimizer.step()

        for i in range(self.gradient_critic_loss_max_steps):
            self.agent.critic_optimizer.zero_grad()
            value_loss = self.critic_loss(data)
            value_loss.backward()
            self.agent.critic_optimizer.step()

        return policy_loss, value_loss

    def train(self):
        # setup logger
        if self.tensorboard:
            self.setup_tensorboard()
        # print the device we are running the learning on
        self.logger.debug("-- device: {}".format(device))

        # log agent info
        self.logger.debug(str(self.agent))

        state, info = self.env.reset()
        episode_return = 0  # sum of rewards over 1 episode
        episode_len = 0  # number of steps over 1 episode
        # episode_returns = []  #

        # the best return found so far
        best_episode_return = self.agent.best_episode_return if self.agent.best_episode_return is not None else - np.Inf

        # the best average returns per epoch found so far
        best_average_epoch_returns = self.agent.best_average_epoch_returns if self.agent.best_average_epoch_returns is \
                                                                              not None else - np.Inf

        starting_epoch = self.agent.epoch if self.agent.epoch is not None else 0

        # number of all episodes experienced so far
        total_episodes = self.agent.total_episodes if self.agent.total_episodes is not None else 0

        for epoch in range(starting_epoch + 1, self.epochs + 1):
            sum_returns = 0  # sum of returns over 1 epoch
            best_return = -np.Inf  # best return over 1 epoch
            n_episodes = 0  # number of episodes

            self.logger.debug('Epoch: {}/{}'.format(epoch, self.epochs))
            for t in range(self.steps_per_epoch):
                action, value, log_probability = self.agent.step(
                    torch.as_tensor(state, dtype=numpy_to_torch_dtype_dict[state.dtype.name], device=device))
                    # torch.as_tensor(state, dtype=torch.float32, device=device))

                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_return += reward
                episode_len += 1

                self.buffer.append(state, action, reward, value, log_probability)
                state = next_state

                episode_timeout = episode_len == self.max_episode_steps
                epoch_ended = t == self.steps_per_epoch - 1
                terminal_state = terminated or truncated or episode_timeout or epoch_ended

                if terminal_state:
                    if not terminated:
                        _, last_value, _ = self.agent.step(
                            torch.as_tensor(state, dtype=numpy_to_torch_dtype_dict[state.dtype.name], device=device))
                    else:
                        last_value = 0

                    self.buffer.estimate_advantage(last_value)

                    # if terminated or episode_timeout:
                    n_episodes += 1
                    sum_returns += episode_return
                    if episode_return > best_return:
                        best_return = episode_return

                    if self.tensorboard:
                        episode_number = total_episodes + n_episodes
                        self.writer.add_scalar("Return - episode", episode_return, episode_number)
                        self.writer.add_scalar("Average reward - episode", episode_return / episode_len, episode_number)

                    episode_return = 0
                    episode_len = 0

                    state, info = self.env.reset()

            actor_loss, critic_loss = self.update()
            average_epoch_returns = round(sum_returns / n_episodes, 2)

            self.logger.debug(
                f"==> Epoch Statistics -- Average epoch returns: {average_epoch_returns} | "
                f"Best episode return: {best_return}")

            if self.tensorboard:
                self.writer.add_scalar("Actor loss - epoch", actor_loss, epoch)
                self.writer.add_scalar("Critic loss - epoch", critic_loss, epoch)
                self.writer.add_scalar('Average returns - epoch', average_epoch_returns, epoch)
                self.writer.flush()

            # update best average epoch returns
            if average_epoch_returns > best_average_epoch_returns:
                best_average_epoch_returns = average_epoch_returns

            self.logger.debug("Saving agent ... ")
            self.agent.save_models()

            # save training info
            total_episodes += n_episodes
            best_episode_return = best_return if best_return > best_episode_return else best_episode_return

            self.agent.save_training_info(self.env_name, self.model_id, epoch, self.steps_per_epoch,
                                          total_episodes, best_episode_return, best_average_epoch_returns)

            if self.break_training_if_solved and self.reward_threshold is not None:
                if average_epoch_returns >= self.reward_threshold:
                    self.logger.debug('************ Environment solved in {} Epochs **************'.format(epoch))
                    return

        self.env.close()
        if self.tensorboard:
            self.writer.close()

    def evaluate(self, n_episodes):
        self.setup_logger(log_file_name=None)
        self.logger.debug('Testing agent performance ...')
        self.logger.debug(self.agent)
        # if render:  # workaround for pybullet envs to work
        #     try:
        #         self.env.render(mode="human")
        #     except:
        #         pass
        state, info = self.env.reset()
        returns = []

        for i in range(int(n_episodes)):
            episode_return = 0
            episode_length = 0
            while True:
                action, value, log_probability = self.agent.step(
                    torch.as_tensor(state, dtype=numpy_to_torch_dtype_dict[state.dtype.name], device=device))
                next_state, reward, terminated, truncated, info = self.env.step(action)
                # pybullet.resetDebugVisualizerCamera(0, 0, 0, 0)
                episode_return += reward
                episode_length += 1
                state = next_state
                if terminated or episode_length == self.max_episode_steps:
                    returns.append(episode_return)
                    state, info = self.env.reset()
                    break
            self.logger.debug(
                "Episode {}/{}: Return: {} | Length: {}".format(i + 1, n_episodes, episode_return, episode_length))

        self.logger.debug("=> Average return over {} episode(s): {}".format(n_episodes, np.mean(returns)))

    def __str__(self):
        return f"""
         seed={self.seed},
         epochs={self.epochs},
         steps_per_epoch={self.steps_per_epoch},
         max_episode_steps={self.max_episode_steps},
         gradient_actor_loss_max_steps={self.gradient_actor_loss_max_steps},
         gradient_critic_loss_max_steps={self.gradient_critic_loss_max_steps},
         agent_state_layer={self.agent_state_layer},
         agent_action_layer={self.agent_action_layer},
         agent_hidden_layers={self.agent_hidden_layers},
         agent_actor_learning_rate={self.agent_actor_learning_rate},
         agent_critic_learning_rate={self.agent_critic_learning_rate},
         agent_std={self.agent_std},
         gamma={self.gamma},
         gae_lambda={self.gae_lambda},
         clip_ratio={self.clip_ratio},
         break_training_if_solved={self.break_training_if_solved},
         reward_threshold={self.reward_threshold}
         ---
         render = {self.render}
         debug = {self.debug}
         tensorboard = {self.tensorboard}
        """
