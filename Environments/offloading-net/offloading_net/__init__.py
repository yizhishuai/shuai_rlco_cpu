# -*- coding: utf-8 -*-
"""
Created on Sat Nov  13 16:39:00 2021

@author: Mieszko Ferens
"""

from gym.envs.registration import register

register(
    id='offload-planning-v0',
    entry_point='offloading_net.envs:offload_planning_v0_netEnv',
)

register(
    id='offload-noplanning-v0',
    entry_point='offloading_net.envs:offload_noplanning_v0_netEnv',
)

register(
    id='offload-planning-v1',
    entry_point='offloading_net.envs:offload_planning_v1_netEnv',
)

register(
    id='offload-noplanning-v1',
    entry_point='offloading_net.envs:offload_noplanning_v1_netEnv',
)
