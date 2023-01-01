#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple implementation of the Proximal Policy Optimization (PPO) algorithm using Pytorch.
Credits to: https://spinningup.openai.com/en/latest/

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

__author__ = "Abdeladim S."
__copyright__ = "Copyright 2022, "
__deprecated__ = False
__license__ = "GNUv3"
__maintainer__ = __author__
__version__ = "1.0.0"


from ppo import Policy


def main():

    env_name = 'CartPole-v0'
    model_id = 'defaults'

    # train = True  # for training
    train = False  # for evaluation
    if train:
        policy = Policy(env_name, model_id=model_id, render_mode=None)
        policy.train()
    else:
        policy = Policy(env_name, model_id=model_id, render_mode='human')
        policy.evaluate(1)


if __name__ == '__main__':
    main()
