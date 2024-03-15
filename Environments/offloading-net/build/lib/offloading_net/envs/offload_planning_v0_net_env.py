# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:44:00 2021

@author: Mieszko Ferens
"""

### Environment for computation offloading

## Nodes use planning (reserve at the earliest available slot in their queues)  节点使用规划（在其队列中最早可用的时隙预留）
## Reward is calculated based on degree of success to comply with the latency   奖励是根据遵从潜伏期的成功程度来计算的

import numpy as np
import gym
from gym import spaces

from .traffic_generator import traffic_generator
from .core_manager import core_manager
from .parameters import (links, links_rate, links_delay, node_type, node_clock,
                         node_cores, n_nodes, net_nodes, vehicle_nodes,
                         n_vehicles, all_paths, node_comb, apps, app_cost,
                         app_data_in, app_data_out, app_max_delay, app_rate,
                         app_info, estimation_err_var, upper_var_limit,
                         lower_var_limit, node_buffer, reserv_limit,
                         topology_label)
                        # 链路，链路速率，链路时延，节点类型，节点clock？
                        # 节点核心，节点数量,网络节点，车辆节点
                        # 车辆数量，所有的路径，？，应用，应用花费
                        # 应用数据输入，应用数据输出，应用最大时延，应用速率
                        # 应用信息，

"""
Explanation on implemented discrete event simulator:
    
    This environment simulates a network consisting of four types of nodes
    (cloud, MEC, RSU and vehicle). The cloud node is special as there can only
    be one and it has unlimited resources (no queue although its processing
    speed is still modeled after real cores). Together with the cloud node, the
    MEC and RSU nodes form a network interconnected with links. The vehicle
    nodes connect to this network via a shared wireless link at the RSU.
    Each vehicle node runs a set of defined applications with generate
    petitions that will have to be processed at some node of the network. This
    also means that unless the processing occurs at the same vehicle, the input
    data from the applications has to be sent to the corresponding node and
    once processed, the output data has to return.
    The applications from one vehicle cannot be processed by other vehicles.
    
    At each time step, one petition is processed. At this time the action
    by the interacting agent is used to calculate delays and reserve a time
    slot at a core of the selected node. The time at which each petition
    are randomly generated with an exponential distribution and a mean defined
    for each application. These time are also used at each time step to update
    the queues at all cores (time passes). Finally, an observation is returned
    to the agent containing information on the current state of the cores of
    the network and the parameters of the next petition's application.

In this environment the petitions are generated in the traffic_generator class
and passed to the environment. The module that manages the queues of the cores
is the core_manager class.

Approximations are used for modeling transmision delays and queues. The focus
is on the processing (the cores).
"""

# 实现离散事件模拟器的说明：
# 该环境模拟由四种类型的节点（云、MEC、RSU 和车辆）组成的网络。云节点的特殊之处在于它只能有一个，而且它拥有无限的资源（没有队列，尽管它的处理速度仍然是仿照真实内核的）。
# MEC和RSU节点与云节点一起形成了一个通过链路相互连接的网络。车辆节点通过 RSU 上的共享无线链路连接到该网络。每个车辆节点运行一组定义的应用程序
# 必须在网络的某个节点处理的请愿书。这也意味着除非处理发生在同一辆车上，否则来自应用程序的输入数据必须发送到相应的节点，一旦处理完毕，输出数据必须返回。
# 来自一辆车的申请不能被其他车辆处理。在每个时间步，处理一个请求。此时，交互代理的动作用于计算延迟并在所选节点的核心处保留一个时隙。每个请求以指数分布和为每个应用程序定义
# 的平均值随机生成的时间。这些时间也用于每个时间步以更新所有核心的队列（时间流逝）。最后，向代理返回一个观察结果，其中包含有关网络核心当前状态和下一个请求应用程序参数的信息。
# 在此环境中，请愿书在 traffic_generator 类中生成并传递给环境。管理核心队列的模块是核心管理器类。
# 近似值用于模拟传输延迟和队列。重点是处理（核心）。
class offload_planning_v0_netEnv(gym.Env):

    def __init__(self):
        # Precision limit (in number of decimals) for numpy.float64 variables
        # used to avoid overflow when operating with times in the simulator  用于在模拟器中与时间操作时避免溢出
        self.precision_limit = 10 # Numpy.float64 has 15 decimals
        
        # Topology name (used as information for metrics for logging)  拓扑名称（用作日志记录的度量信息）本身
        self.topology_label = topology_label
        
        # Node type list (used as information for metrics during testing) 节点类型列表（在测试期间用作度量的信息）
        self.node_type = node_type
        
        # Node combination list (used as information for heuristic algorithms) 节点组合列表（用作启发式算法的信息）
        self.node_comb = node_comb
        
        # Type of next application 下一个应用程序自身的类型
        self.app = 0
        # Origin node of next application (vehicle) 下一个应用程序的起源节点（车辆）
        self.app_origin = 0
        # Origin node shift (multiple vehicles can be defined by one node) 原始节点移位（一个节点可以定义多个交通工具）自身。
        self.app_shift = 0
        # Time until next application 直到下一个应用程序
        self.app_time = 0
        # Cost of processing the next application (clock clycles/bit) 处理下一个应用程序本身的成本（时钟clycles/bit）
        self.app_cost = 0
        # Input data quantity from the next application (kbits) 从下一个应用程序本身输入数据量(kbits)
        self.app_data_in = 0
        # Output data quantity from the next application (kbits) 从下一个应用程序自身输出数据量(kbits)
        self.app_data_out = 0
        # Maximum tolerable delay for the next application (ms) 下一个应用程序(ms)自身的最大允许延迟。
        self.app_max_delay = 0
        
        # Reference information for defined applications (maximums) 已定义应用程序的参考信息（最大值）。
        self.max_cost = max(app_cost)
        self.max_data_in = max(app_data_in)
        self.max_data_out = max(app_data_out)
        self.max_delay = max(app_max_delay)
        self.app_param_count = 4
        
        # Total estimated delay of current application  当前应用程序本身的总估计延迟。
        self.total_delay_est = 0
        # Total real delays of all processed application since last time step  自上次步骤以来所有已处理应用程序的总实际延迟。
        self.total_delays = 0
        self.app_types = 0 # Corresponding application types    对应的应用程序类型
        
        # For metrics  对于度量
        # Total number of terminated applications in this time step.  在此时间步骤中被终止的应用程序总数。
        self.app_count = 0
        # Number of succesfully processed applications in this time step  在此时间步骤内成功处理的应用程序数
        self.success_count = 0
        
        # For monitoring purposes the observation will be kept at all times  为了监测目的，将随时保存观察结果
        self.obs = 0
        
        # Number of cores in the network (except cloud) and one vehicle   网络中的核心数（云除外）和一个车辆本身
        self.n_cores = 0
        for a in range(net_nodes + 1):
            self.n_cores += node_cores[a]
        
        # Total number of vehicles in the network  网络中的车辆总数
        self.n_vehicles = n_vehicles
        # Number of vehicles per vehicle node (rounded to the nearest integer)  每个车辆节点的车辆数（四舍五入至最接近的整数）
        self.node_vehicles = 0
        
        # Discrete event simulator traffic generation initialization   离散事件模拟器流量生成初始化
        self.traffic_generator = traffic_generator(n_nodes, net_nodes, apps,
                                                   app_cost, app_data_in,
                                                   app_data_out, app_max_delay,
                                                   app_rate, app_info)
        
        # Discrete event simulator core manager initialization    离散事件模拟器核心管理器初始化
        self.core_manager = core_manager(estimation_err_var, upper_var_limit,
                                         lower_var_limit, reserv_limit)
        
        # The observation space has an element per core in the network (with   观测空间在网络中每个核心都有一个元素
        # the exception of vehicles, where only one is observable at a time)  （车辆除外，一次只能观测到一个）
        self.observation_space = spaces.Box(low=0, high=1, shape=(
            self.n_cores + self.app_param_count + n_nodes + 1, 1),
            dtype=np.float32)
        
        self.action_space = spaces.Discrete(net_nodes + 1)

    def step(self, action):
        
        # For each application, the node that processes it cannot be a vehicle  对于每个应用程序，处理它的节点不能是本地车辆以外的车辆
        # other than the local vehicle (the one that generates the petition)   （生成请愿的车辆）获取与代理操作对应的节点之间的路径
        
        # Get path between nodes corresponding to the action of the agent     获取与代理动作对应的节点之间的路径
        path = self.get_path(action)
        
        # For each action one node for processing an application is chosen,  对于每个操作，选择一个处理应用程序的节点，
        # reserving one of its cores (might queue)    保留它的一个核心（可能排队）
        
        # Calculate the associated delays (transmission and processing) without
        # accounting for queueing
        forward_delay, return_delay, proc_delay = self.calc_delays(action,path)   #计算相关的延迟（传输和处理）而不考虑排队
        
        # Choose the next as soon to be available core from the selected    
        # processing node and reserve its core for the required time
        # If the selected processing node is the cloud no reservation is done
        # as it has infinite resources
        # Due to a vehicle node defining multiple vehicles, a translation to
        # the corresponding index is required
        #从所选处理节点中选择下一个可用的核心，并在所需时间内保留其核心。
        # 如果所选处理节点是云，则不进行保留，因为它具有无限的资源，
        # 由于一个车辆节点定义了多个车辆，因此需要转换到相应的索引
        self.total_delay_est = 0
        self.total_delays = np.array([], dtype=np.float32)
        self.app_types = np.array([], dtype=np.int32)
        self.app_count = [0]*len(apps)
        if(action): # Not cloud
            if(action == net_nodes): # Process locally
                node_index = (net_nodes-1 + self.app_shift +
                              (self.app_origin-net_nodes-1)*self.node_vehicles)
            else: # Process in network
                node_index = action - 1
            self.total_delay_est = self.core_manager.reserve_with_planning(
                node_index, forward_delay, proc_delay, return_delay, self.app)
        else: # Cloud
            # Calculate the total estimated delay of the application processing   #计算应用程序处理的估计总延迟
            self.total_delay_est = forward_delay + proc_delay + return_delay
            self.total_delays = np.append(
                self.total_delays, self.total_delay_est)
            self.app_types = np.append(self.app_types, self.app)
        
        ## Reward calculation (a priori)  ##奖励计算（先验）
        if(action and self.total_delay_est < 0): # Application not queued    #应用程序未排队
            reward = -1000
            self.app_count[self.app-1] += 1
        else: # Application queued/processed   #已排队/已处理的应用程序
            reward = 0
        
        # Get next arriving petition   #收到下一份到达的请愿书
        next_petition = self.traffic_generator.gen_traffic()
        # Assign variables from petition   从请愿书中分配变量
        self.app = next_petition[0]
        self.app_origin = next_petition[1]
        self.app_shift = next_petition[2]
        self.app_time = np.around(next_petition[3], self.precision_limit)
        self.app_cost = next_petition[4]
        self.app_data_in = next_petition[5] # in kb
        self.app_data_out = next_petition[6] # in kb
        self.app_max_delay = next_petition[7]
        
        ## Observation calculation
        # Calculate the next relevant vehicle index   计算下一个相关车辆指标
        vehicle_index = (net_nodes-1 + self.app_shift +
                         (self.app_origin-net_nodes-1)*self.node_vehicles)
        
        # Core and processed applications information  Core and processed applications information
        self.obs, total_delays, app_types = (
            self.core_manager.update_and_calc_obs(
                self.app_time, self.precision_limit, vehicle_index))
        
        self.total_delays = np.append(self.total_delays, total_delays)
        self.app_types = np.append(self.app_types, app_types)
        
        ## Reward calculation (a posteriori)
        # Check for processed and unprocessed applications
        processed = np.where(self.total_delays >= 0)[0]
        failed = np.where(self.total_delays < 0)[0]
        # Find corresponding max tolerable delays
        max_delays = np.array([], dtype=np.float32)
        for i in self.app_types:
            max_delays = np.append(max_delays, app_max_delay[i-1])
            self.app_count[i-1] += 1
        # Calculate total reward for current time step
        remaining = np.clip(
            max_delays[processed] - self.total_delays[processed], None, 0)
        over_max_delay = np.where(remaining < 0)[0]
        remaining[over_max_delay] = np.subtract(remaining[over_max_delay], 100)
        reward += np.sum(remaining)
        reward -= len(self.total_delays[failed]) * 1000
        
        # For metrics (successful processing)
        self.success_count = [0]*len(apps)
        under_max_delay = np.where(remaining == 0)[0]
        success_count = np.unique(
            (self.app_types[processed])[under_max_delay], return_counts=True)
        for i, app in enumerate(success_count[0]):
            self.success_count[app-1] += success_count[1][i]
        
        # Add current petition's application type to observation
        app_type = []
        app_type.append(1 - (self.app_cost/self.max_cost))
        app_type.append(1 - (self.app_data_in/self.max_data_in))
        app_type.append(1 - (self.app_data_out/self.max_data_out))
        app_type.append(self.app_max_delay/self.max_delay)
        self.obs = np.append(self.obs, np.array(app_type, dtype=np.float32))
        
        # Add current petition's estimated delay for each possible action
        predict_delay = []
        for action in range(net_nodes + 1):
            path = self.get_path(action)
            predict_delay.append(1 - min(
                sum(self.calc_delays(action, path))/self.app_max_delay, 1))
        self.obs = np.append(
            self.obs, np.array(predict_delay, dtype=np.float32))
        
        # Add current vehicle's position to the observation
        vehicle_pos = [0]*vehicle_nodes
        vehicle_pos[self.app_origin-net_nodes-1] = 1
        self.obs = np.append(self.obs, np.array(vehicle_pos, dtype=np.float32))
        
        done = False # This environment is continuous and is never done
        
        return np.array([self.obs, reward, done, ""], dtype=object)

    def reset(self):
        
        # Define the number of vehicles per vehicle node (rounded to the
        # nearest integer) - Even distribution
        self.node_vehicles = round(self.n_vehicles/vehicle_nodes)
        
        # Reset and create all cores
        self.core_manager.reset(n_nodes, node_cores, self.node_vehicles,
                                self.node_type, node_buffer)
        
        # Generate initial petitions to get things going
        self.traffic_generator.gen_initial_traffic(self.node_vehicles)
        
        # Get first petition
        next_petition = self.traffic_generator.gen_traffic()
        # Assign variables from petition
        self.app = next_petition[0]
        self.app_origin = next_petition[1]
        self.app_shift = next_petition[2]
        self.app_time = np.around(next_petition[3], self.precision_limit)
        self.app_cost = next_petition[4]
        self.app_data_in = next_petition[5]
        self.app_data_out = next_petition[6]
        self.app_max_delay = next_petition[7]
        
        # Calculate observation
        self.obs = np.array(
            [0]*(self.n_cores + self.app_param_count + n_nodes + 1),
            dtype=np.float32)
        
        return self.obs

    def render(self, mode='human'):
        # Print current core reservation times   打印当前核心预订时间
        info = 'Core reservation time: ' + str(self.obs[0:self.n_cores])
        # Print next application to be processed  打印下一个要处理的申请
        info = info + '\n' + 'Next application: ' + str(self.app)
        # Print origin vehicle node   打印原始车辆节点
        info = info + '\n' + 'From vehicle node: ' + str(self.app_origin)
        # Print application related observation   打印应用程序相关观察
        info = (info + '\n' + 'Application info: ' +
                str(self.obs[self.n_cores:self.n_cores+self.app_param_count]))
        # Print delay prediction   打印延迟预测
        info = (info + '\n' + 'Delay prediction: ' +
                str(self.obs[-(net_nodes+1+vehicle_nodes):-vehicle_nodes]))
        
        return info
    
    def get_path(self, action):
        # Translate the action to the correct node number if the agent decides
        # to process the application locally
        if(action == net_nodes):
            action = self.app_origin - 1
            path = []
        # Prepare path between origin node and the selected processor
        else:
            current_nodes = [self.app_origin, action + 1]
            current_nodes.sort()
            path = all_paths[node_comb.index(current_nodes)]
        
        return path
    
    def calc_delays(self, action, path):
        # Calculate the transmission delay for the application's data (in ms)
        forward_delay = 0
        return_delay = 0
        if path:
            for a in range(len(path)-1):
                link = [path[a], path[a+1]]
                link.sort()
                link_index = links.index(link)
                link_rate = links_rate[link_index] # in kbit/s
                link_delay = links_delay[link_index] # in ms
                
                forward_delay += (self.app_data_in/link_rate)*1000 + link_delay
                return_delay += link_delay + (self.app_data_out/link_rate)*1000
        
        # Calculate the processing delay at the node (in ms)
        proc_delay = (self.app_data_in*self.app_cost/node_clock[action])*1000
        
        # Limit the precision of the numbers to prevent overflow
        forward_delay = np.around(forward_delay, self.precision_limit)
        return_delay = np.around(return_delay, self.precision_limit)
        proc_delay = np.around(proc_delay, self.precision_limit)
        
        return forward_delay, return_delay, proc_delay
    
    def set_total_vehicles(self, total_vehicles):
        self.n_vehicles = total_vehicles
    
    def set_error_var(self, error_var):
        self.core_manager.set_error_var(error_var)
    
    def set_upper_var_limit(self, upper_lim):
        self.core_manager.set_upper_var_limit(upper_lim)
    
    def set_lower_var_limit(self, lower_lim):
        self.core_manager.set_lower_var_limit(lower_lim)

