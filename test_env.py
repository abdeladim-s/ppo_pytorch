#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_env

Longer description of this module.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import time

import gymnasium as gym


def play_episode(env_name: str, render_mode: str):
    env = gym.make(env_name, render_mode=render_mode)
    observation, info = env.reset()

    while True:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'observation: {observation}')
        print(f'reward: {reward}')
        print(f'terminated: {terminated}')
        print(f'truncated: {truncated}')
        print(f'info: {info}')
        print('---------------------------------')
        time.sleep(5)
        if terminated or truncated:
            # env.reset()
            break

    env.close()


if __name__ == '__main__':
    env_name = 'CartPole-v0'
    render_mode = 'human'
    # render_mode = None
    play_episode(env_name, render_mode)
