# -*- coding: utf-8 -*-
"""
Created on Saturday Feb 12 11:08:57 2022

@author: Mieszko Ferens
"""

from random import choice

### Heuristic algorithms for load distributions

"""
These classes implement the heuristic algorithms for comparison with RL agents.
They are designed to work in the environments that tested RL agents are going
to be working in.
"""

class local_processing_agent:
    
    """
    This agent always selects the last action, because in our environment that
    is always the local vehicle (processes the applications locally).
    """
    # 不管怎么样，action都取3（本地车辆）
    # 这一块只是返回一个动作
    def __init__(self, n_actions):
        self.action = n_actions-1
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        return self.action
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        return self.action

class cloud_processing_agent:
    
    """
    This agent always selects the first action, because in our environment that
    is always the cloud node (processes the applications at cloud).
    """
    def __init__(self):
        self.action = 0
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        return self.action
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        return self.action

class uniform_distribution_agent:
    
    """
    This agent selects any of the possible action with uniform probability,
    distributing its decisions and the load equally.
    """
    # 随机取动作【0，1，2，3】
    def __init__(self, env):
        self.last_action = env.action_space.n - 1
        self.env = env
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        # Find the corresponding RSU and MEC node actions (in case there is
        # multiple) by looking at the path to the cloud node
        path = self.env.get_path(0)
        MEC = path[1] - 1
        # Select action randomly
        return choice([0, MEC, self.last_action])
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        # Find the corresponding RSU and MEC node actions (in case there is
        # multiple) by looking at the path to the cloud node
        path = self.env.get_path(0)
        MEC = path[1] - 1
        # Select action randomly
        return choice([0, MEC, self.last_action])

class max_distance_agent:
    
    """
    This agent always selects the lowest value action possible given the
    total delay estimation to distribute the load in the network while
    attempting to reduce delays (the agent basically checks what node can
    process the application in theory going by the order of the furthest nodes
    first to the closest, i.e. cloud -> MEC -> RSU -> local vehicle).
    It doesn't take into account the current load of the nodes.
    It will not consider MECs or RSUs from other branches than the one the
    vehicle is in.
    """
    # 
    # 从最远的一端依次进行延迟的匹配，云->MEC->RSU->LOACL VEHICLE
    def __init__(self, env):
        self.n_actions = env.action_space.n
        self.env = env
    
    def act_and_train(self, obs, reward):
        # Input parameters are ignored since this is not a RL agent
        # Find the corresponding RSU and MEC node actions (in case there is
        # multiple) by looking at the path to the cloud node
        path = self.env.get_path(0)
        MEC = path[1] - 1
        # Check starting from the furthest nodes (lowest actions) whether the
        # estimated delay is under the max tolerable delay (consider only
        # current network branch)
        for i in [0, MEC, self.n_actions-1]:
            path = self.env.get_path(i) # Get path
            # Check if delay is acceptable
            if(sum(self.env.calc_delays(i, path)) <= self.env.app_max_delay):
                break
        # Return first acceptable action
        return i
    
    def act(self, obs):
        # Input parameter is ignored since this is not a RL agent
        # Find the corresponding RSU and MEC nodes (in case there is multiple)
        # by looking at the path to the cloud node
        path = self.env.get_path(0)
        MEC = path[1]
        # Check starting from the furthest nodes (lowest actions) whether the
        # estimated delay is under the max tolerable delay (consider only
        # current network branch)
        for i in [0, MEC, self.n_actions-1]:
            path = self.env.get_path(i) # Get path
            # Check if delay is acceptable
            if(sum(self.env.calc_delays(i, path)) <= self.env.app_max_delay):
                break
        # Return first acceptable action
        return i

# Funtion that instances shortest path algorithms imitating RL agents
def make_heuristic_agents(env):
    
    agent_local_processing = local_processing_agent(env.action_space.n)
    local_processing_info = 'Always local processing'
    
    agent_cloud_processing = cloud_processing_agent()
    cloud_processing_info = 'Always cloud processing'
    
    agent_uniform_distribution = uniform_distribution_agent(env)
    uniform_distribution_info = 'Uniform distribution of load'
    
    agent_max_distance = max_distance_agent(env)
    max_distance_info = 'Max distance processing'
    
    return [[agent_uniform_distribution, uniform_distribution_info],
            # [agent_local_processing, local_processing_info],
            [agent_cloud_processing, cloud_processing_info]]
            # [agent_max_distance, max_distance_info]]

