# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:43:15 2021

@author: Mieszko Ferens
"""

import numpy as np

### Class for traffic generation

class traffic_generator():
    
    def __init__(self, n_nodes, net_nodes, apps, app_cost, app_data_in,
                 app_data_out, app_max_delay, app_rate, app_info):
        # The petition queue stores petitions that have been precalculated  请愿队列存储已预先计算的请愿
        # NOTE: This queue does not represent pending petitions for the agent,
        #       it simply stores all petitions that have been generated
        #       randomly for the discrete event simulator
        # 此队列不表示代理的未决请求，
        # 它只是存储所有已经生成的请愿书
        # 随机离散事件模拟器

        self.petition_Q = []
        
        # Number of nodes in the network
        self.n_nodes = n_nodes
        # Number of non-vehicle nodes in the network
        self.net_nodes = net_nodes
        
        # Imported application parameters
        self.apps = apps
        self.app_cost = app_cost
        self.app_data_in = app_data_in
        self.app_data_out = app_data_out
        self.app_max_delay = app_max_delay
        self.app_info = app_info
        self.app_rate = app_rate
    
    def gen_traffic(self):
        
        """
        Every time this method is called it returns the next arriving
        application petition and generates a new one for the source that was
        the origin for that petition. This newly generated petition is inserted
        into the already generated petitions queue into the proper position
        depending on its arrival time
        """
        # 每次调用此方法时，它都会返回下一个到达的app请求，并为该请求的来源生成一个新的请求。
        # 这个新生成的应用请求被插入到已经生成的请求队列中的适当位置，这取决于它的到达时间

        
        # The next generated petition (next in arrival time) to send to the
        # controller
        current_petition = self.petition_Q.pop(0)
        # print(current_petition)  
        # Update arrival times for the rest of the petitions in the queue
        # [3]是下次到达时间，为每一个修改下次到达时间
        for i in range(len(self.petition_Q)):
            self.petition_Q[i][3] -= current_petition[3]
        
        ## Generate next petition from node that is about to be processed
        # [0]是应用序号  下标-1 是0-5
        app_index = current_petition[0] - 1
        
        # Calculate the arrival time of the petition
        # 为这个应用生成一个下次到达时间
        next_arrival_time = self.gen_distribution(
                self.app_rate[app_index], 'exponential')
        # current_petition[1]： 4 车辆节点  current_petition[2]：车辆的号码
        next_petition = [app_index + 1, current_petition[1],
                         current_petition[2], next_arrival_time,
                         self.app_cost[app_index], self.app_data_in[app_index],
                         self.app_data_out[app_index],
                         self.app_max_delay[app_index]]
        
        # Insert new petition in the corresponding queue position (according
        # to arrival time)
        i = next((x for x, f in enumerate(self.petition_Q)
                  if f[3] > next_arrival_time), -1)
        if(self.petition_Q and i >= 0):
            self.petition_Q.insert(i, next_petition)
        else:
            self.petition_Q.append(next_petition)
        
        # Return current petition to the controller
        return current_petition

    def gen_initial_traffic(self, node_vehicles):
        
        self.petition_Q.clear() # Clear the queue for initialization
        
        # Generate one petition for each application of each vehicle
        # print(self.net_nodes, self.n_nodes)   3   4
        for node in range(self.net_nodes, self.n_nodes):
            for vehicle in range(node_vehicles):
                for app_index in range(len(self.apps)):
                    # print("node:{}  vehicle:{}  app_index:{}".format(node, vehicle, app_index ))
                    # node 只有 3 vechile 0-49 appindex 0-5   1*50*6
                    # Calculate the arrival time of the petition
                    # 下次到达时间服从速度数值的一个指数分布
                    next_arrival_time = self.gen_distribution(
                            self.app_rate[app_index], 'exponential')
                    # print("self.app_rate[app_index]:{}  next_arrival_time:{}",
                    # format(self.app_rate[app_index]))
                    # print(next_arrival_time)
                    # Assign values to the queue (not sorted by arrival times)
                    # print(app_index + 1, node + 1, vehicle, next_arrival_time,
                    #      self.app_cost[app_index], self.app_data_in[app_index],
                    #      self.app_data_out[app_index],
                    #      self.app_max_delay[app_index])
                    # 下标，4 ， 50个车辆， 下次到达时间 | cost | data_in  data_out  max_delay
                    self.petition_Q.append(
                        [app_index + 1, node + 1, vehicle, next_arrival_time,
                         self.app_cost[app_index], self.app_data_in[app_index],
                         self.app_data_out[app_index],
                         self.app_max_delay[app_index]])
        
        # Sort petitions by arrival time
        self.petition_Q.sort(key=lambda x:x[3])  

    def gen_distribution(self, beta=1, dist='static'):
        if(dist in 'static'):
            return beta
        elif(dist in 'exponential'):
            return np.random.exponential(beta)
        else:
            raise KeyboardInterrupt('Unexpected type of distribution')

