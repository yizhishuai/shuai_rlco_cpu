# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:06:32 2021

@author: Mieszko Ferens
"""
import time
import numpy as np

### Class for core management

class core_manager():
    
    def __init__(self, error_var, upper_var_limit, lower_var_limit,
                 reserv_limit):
        
        # Number of network nodes which the class store information of
        self.net_nodes = 0
        
        # Processing time estimation error variation coeficient
        self.error_var = error_var
        # Processing time estimation error variation limits (percentage)
        self.upper_var_limit = upper_var_limit
        self.lower_var_limit = lower_var_limit
        
        # Lists that hold available time slots info for all cores (except
        # cloud)
        # Available time slots's duration
        self.slots_duration = []
        # Available time slots's start time
        self.slots_start = []
        
        # Lists that hold info on reservations for all cores (except cloud)
        # Current remaining number of possible reservations
        self.queue_limit = []
        # Times at with each reservation starts and ends
        self.reserv_start = []
        self.reserv_end = []
        # Estimated total delay of each reservation
        self.total_est_delay = []
        # Type of application of each reservation
        self.app_type = []
        
        # Time to add to total time for next update of each core
        self.update_time = []
        
        # Limit for relative core load calculation (in ms)
        self.time_limit = np.array([], dtype=np.float64)
        # Limit for maximum reservations in queue
        self.reserv_limit = reserv_limit
    
    def reserve_no_planning(self, action, forward_delay, proc_delay,
                            return_delay, app_type):
        
        # Look for time slots
        next_core = []
      
        for core, f in enumerate(self.slots_duration[action]):
            
            # If core queue has too many reservations ignore it
            # queue_limit[action] = 一个数组   初始化 是 1000*core的数量
            # print("queue_limit[{}]:{}".format(action,self.queue_limit[action]))
            
            if(self.queue_limit[action][core] == 0):
                continue
            # Check if the end of the queue has room for the reservation
            diff = (max(self.slots_start[action][core][-1], forward_delay) -
                    self.slots_start[action][core][-1])
            if(f[-1] >= diff + proc_delay):
                next_core.append(
                    [core, len(f)-1, self.slots_start[action][core][-1]])
    
        # Check if any slot is available
        if(len(next_core) == 0):
            return -1 # If processing was not possible return -1
        
        # Order the time slots by minimal delay
        next_core.sort(key=lambda x:x[2])
        core = next_core[0][0]
        slot = next_core[0][1]
        # Create reservation  预定
        # if(action == 0):
        #     time.sleep(5)
        total_est_delay = self.reserve(action, core, slot, forward_delay,
                                       proc_delay, return_delay, app_type)
        
        return total_est_delay
    
    def reserve_with_planning(self, action, forward_delay, proc_delay,
                              return_delay, app_type):
        
        # Look for time slots
        available = []
        next_core = []
        for core, f in enumerate(self.slots_duration[action]):
            # If core queue has too many reservations ignore it
            if(self.queue_limit[action][core] == 0):
                available.append(np.array([])) # Empty append to match indexes
                continue
            # Look for slots with sufficient durations (if fully usable)
            available.append(np.argwhere(f > proc_delay))
            # Check if no slots are available
            if(available[core].size == 0):
                continue
            # Register first available slot from core
            for i, g in enumerate(available[core]):
                if(forward_delay <= self.slots_start[action][core][g[0]]):
                    next_core.append(
                        [core, available[core][i][0],
                         self.slots_start[action][core][available[core][i][0]]])
                    break
                # Check if slots that cannot be used fully are sufficient
                elif(f[g[0]] > forward_delay -
                   self.slots_start[action][core][g[0]] + proc_delay):
                    next_core.append(
                        [core, available[core][i][0],
                         self.slots_start[action][core][available[core][i][0]]])
                    break
        
        # Check if any slot is available
        if(len(next_core) == 0):
            return -1 # If processing was not possible return -1
        
        # Order the time slots by minimal delay
        next_core.sort(key=lambda x:x[2])
        core = next_core[0][0]
        slot = next_core[0][1]
        
        # Create reservation
        total_est_delay = self.reserve(action, core, slot, forward_delay,
                                       proc_delay, return_delay, app_type)
        
        return total_est_delay
    
    def reserve(self, action, core, slot, forward_delay, proc_delay,
                return_delay, app_type):
        # print("---now is resere---")
        # print("forward_delay={}\nself.slots_start[{}][{}]={}\nself.slots_start[{}][{}][{}]={}".format(forward_delay,
        # action,core,self.slots_start[action][core],action,core,slot,self.slots_start[action][core][slot]))
        # Calculate the total estimated delay
        # if(action == 0):
        #     # print("-----------------------------------------")
        
        total_est_delay = (max(forward_delay,
                               self.slots_start[action][core][slot]) +
                           proc_delay + return_delay)
        # print(total_est_delay)
        # Register the reservation
        self.queue_limit[action][core] -= 1
        
        # Assign the reservation to the first available time slot
        # If arrival of data is prior to slot's start
        start_time = 0
        if(forward_delay <= self.slots_start[action][core][slot]):
            # Register the reservation's start time
            self.reserv_start[action][core] = np.sort(np.append(
                self.reserv_start[action][core],
                self.slots_start[action][core][slot]))
            start_time = self.slots_start[action][core][slot]
            # print("reserv_start:{}".format(self.reserv_start[action][core]))
            # Shorten the slot
            self.slots_start[action][core][slot] += proc_delay
            self.slots_duration[action][core][slot] -= proc_delay
            # Register the reservation's end time
            self.reserv_end[action][core] = np.sort(np.append(
                self.reserv_end[action][core],
                self.slots_start[action][core][slot]))
            # Delete the slot if it is null (unless it is the end of the queue)
            if(self.slots_duration[action][core][slot] == 0 and
               len(self.slots_start[action][core]) - 1 != slot):
                self.slots_start[action][core] = np.delete(
                    self.slots_start[action][core], slot)
                self.slots_duration[action][core] = np.delete(
                    self.slots_duration[action][core], slot)
            # Negative values for slot duration should not be possible
            elif(self.slots_duration[action][core][slot] < 0):
                raise KeyboardInterrupt(
                    "Trouble with precision at core queues!")
        # If arrival of data is posterior to slot's start
        else:
            # Register the reservation's start time
            self.reserv_start[action][core] = np.sort(np.append(
                self.reserv_start[action][core], forward_delay))
            start_time = forward_delay
            # Create new slot before the new reservation
            self.slots_start[action][core] = np.insert(
                self.slots_start[action][core], slot,
                self.slots_start[action][core][slot])
            self.slots_duration[action][core] = np.insert(
                self.slots_duration[action][core], slot,
                forward_delay - self.slots_start[action][core][slot])
            # Shorten the old slot
            self.slots_start[action][core][slot+1] = (
                forward_delay + proc_delay)
            self.slots_duration[action][core][slot+1] -= (
                self.slots_duration[action][core][slot] + proc_delay)
            # Register the reservation's end time
            self.reserv_end[action][core] = np.sort(np.append(
                self.reserv_end[action][core],
                self.slots_start[action][core][slot+1]))
            # Delete the slot if it is null (unless it is the end of the queue)
            if(self.slots_duration[action][core][slot+1] == 0 and
               len(self.slots_start[action][core]) - 1 != slot+1):
                self.slots_start[action][core] = np.delete(
                    self.slots_start[action][core], slot+1)
                self.slots_duration[action][core] = np.delete(
                    self.slots_duration[action][core], slot+1)
            # Negative values for slot duration should not be possible
            elif(self.slots_duration[action][core][slot+1] < 0):
                raise KeyboardInterrupt(
                    "Trouble with precision at core queues!")
        
        # Find the reservations index and register its total estimated delay
        index = np.where(self.reserv_start[action][core] == start_time)[0][0]
        self.total_est_delay[action][core] = np.insert(
            self.total_est_delay[action][core], index, total_est_delay)
        # Register the type of application that corresponds to this reservation
        self.app_type[action][core] = np.insert(
            self.app_type[action][core], index, app_type)
        
        return total_est_delay
    
    def update_and_calc_obs(self, app_time, precision_limit, obs_vehicle):
        
        ## Observation and total real delays calculation
        obs = np.array([], dtype=np.float32)
        total_delay = np.array([], dtype=np.float32)
        app_type = np.array([], dtype=np.int32)
        # Update the times which each core has to be updated by
        for node in range(len(self.update_time)):
            for core in range(len(self.update_time[node])):
                self.update_time[node][core] += app_time
                
        # Create an array with each element being a core (the self.cores
        # variable is an irregular array so a loop is necessary)
        for node in list(range(self.net_nodes)) + [obs_vehicle]:
            core_load = []
            # print(self.slots_start)
            # print("node:{}".format(node))
            # print(len(self.slots_start[node]))
            for core in range(len(self.slots_start[node])):
                
                # Check update time for this core
                app_time = np.around(
                    self.update_time[node][core], precision_limit)
                self.update_time[node][core] = 0
                
                # As time passes the core's queue advances
                self.slots_start[node][core] -= app_time
                self.slots_duration[node][core][-1] += app_time
                self.reserv_start[node][core] -= app_time
                self.reserv_end[node][core] -= app_time
                
                # Check if any reservation started processing
                started = len(np.where(self.reserv_start[node][core] < 0)[0])
                # The first reservation could have started processing before
                # the last time step and as such has already been updated
                if(started > 0 and
                   self.reserv_start[node][core][0] + app_time < 0):
                    loop = range(1, started)
                else:
                    loop = range(started)
                
                # Update queue one by one considering real values
                for i in loop:
                    # Check if in the updated queue the reservation
                    # actually started processing
                    if(self.reserv_start[node][core][i] >= 0):
                        break
                    
                    self.process_and_update_queue(i, node, core, app_time,
                                                  precision_limit)
                
                # Check if any reservation has been canceled
                canceled = len(np.where(
                    self.total_est_delay[node][core] < 0)[0])
                if(canceled):
                    total_delay = np.append(total_delay,
                        self.total_est_delay[node][core][-canceled::])
                    self.total_est_delay[node][core] = np.delete(
                        self.total_est_delay[node][core], range(-canceled, 0))
                    app_type = np.append(
                        app_type, self.app_type[node][core][-canceled::])
                    self.app_type[node][core] = np.delete(
                        self.app_type[node][core], range(-canceled, 0))
                
                # Check if any reservation has ended
                ended = len(np.where(self.reserv_end[node][core] <= 0)[0])
                if(ended):
                    self.queue_limit[node][core] += ended
                    self.reserv_start[node][core] = np.delete(
                        self.reserv_start[node][core], range(ended))
                    self.reserv_end[node][core] = np.delete(
                        self.reserv_end[node][core], range(ended))
                    total_delay = np.append(
                        total_delay, self.total_est_delay[node][core][0:ended])
                    self.total_est_delay[node][core] = np.delete(
                        self.total_est_delay[node][core], range(ended))
                    app_type = np.append(
                        app_type, self.app_type[node][core][0:ended])
                    self.app_type[node][core] = np.delete(
                        self.app_type[node][core], range(ended))
                
                # Check if any slot has ended
                if(np.amax(self.slots_start[node][core]) < 0):
                    loop = range(len(self.slots_start[node][core]))
                else:
                    loop = range(np.argmin(self.slots_start[node][core] < 0))
                for i in reversed(loop):
                    diff = np.around(self.slots_duration[node][core][i] +
                                     self.slots_start[node][core][i],
                                     precision_limit)
                    if(diff > 0): # If there is still time remaining
                        self.slots_start[node][core][i] = 0
                        self.slots_duration[node][core][i] = diff
                    else: # If the slot is over (all prior slots too)
                        self.slots_start[node][core] = np.delete(
                            self.slots_start[node][core], range(i+1))
                        self.slots_duration[node][core] = np.delete(
                            self.slots_duration[node][core], range(i+1))
                        break
                
                # Calculate load of the core
                core_load.append(np.sum(
                    self.slots_duration[node][core])/self.time_limit[node])
                if(core_load[-1] < 0 or core_load[-1] > 1): # Impresicion
                    print(
                        "-----------------\n"
                        "Warning: Impresicion detected while updating queues\n"
                        "\nCore Load = " + str(core_load[-1]) +
                        "\nSlots Start = " +
                        str(self.slots_start[node][core]) +
                        "\nSlots Duration = " +
                        str(self.slots_duration[node][core]) +
                        "\nReserv Start = " +
                        str(self.reserv_start[node][core]) +
                        "\nReserv End = " + str(self.reserv_end[node][core]) +
                        "\nTotal Est Delay = " +
                        str(self.total_est_delay[node][core]) +
                        "\n-----------------")
            # Add core load to observation
            obs = np.append(obs, np.array(core_load, dtype=np.float32))
        
        return obs, total_delay, app_type
    
    def reset(self, n_nodes, node_cores, node_vehicles, node_type,
              node_buffer):
        
        # Store the amount of non-vehicle nodes that will be managed
        self.net_nodes = n_nodes - node_type.count(3) - 1
        
        # Reset all cores
        self.slots_duration = []
        self.slots_start = []
        self.queue_limit = []
        self.reserv_start = []
        self.reserv_end = []
        self.total_est_delay = []
        self.app_type = []
        self.update_time = []
        self.time_limit = np.array([], dtype=np.float64)
        # n_nodes = 4
        # a= 0 node_type[0] = 1 是云节点 忽略
        # a= 1 时 loop = 1  所以 只有j=0    node_core[1]=8   同理 node_core[2]=4
        # 因此 self.开头的变量走只有一份，共三个内容  下标为0，1，2
        # 比如 slots_duration[duration1=[node_buffer=0*8],duration2=[node_buffer=0*4],duration3=[node_buffer=a*1]]
        for a in range(n_nodes):
            # print("n_nodes:{} and a:{}".format(n_nodes,a))
            if(node_type[a] != 1): # Ignore cloud nodes
                if(node_type[a] != 3): # If it's not a vehicle
                    loop = 1
                else: # If it's a vehicle
                    loop = node_vehicles
                for j in range(loop):
                    # print("j:{}".format(j))
                    duration = []
                    start = []
                    reserv_starts = []
                    reserv_ends = []
                    total_est_delays = []
                    app_types = []
                    limit = [self.reserv_limit]*node_cores[a]  # reserv_limit=1000
                    # print("node_cores[a]:{}".format(node_cores[a]))
                    update_times = np.zeros(node_cores[a], dtype=np.float64)
                    # 
                    for i in range(node_cores[a]):
                        duration.append(np.array([np.float64(node_buffer[a])],
                                                 dtype=np.float64))
                        start.append(np.array([0], dtype=np.float64))
                        reserv_starts.append(np.array([], dtype=np.float64))
                        reserv_ends.append(np.array([], dtype=np.float64))
                        total_est_delays.append(np.array([], dtype=np.float64))
                        app_types.append(np.array([], dtype=np.int32))
                    self.slots_duration.append(duration)
                    self.slots_start.append(start)
                    self.queue_limit.append(limit)
                    self.reserv_start.append(reserv_starts)
                    self.reserv_end.append(reserv_ends)
                    self.total_est_delay.append(total_est_delays)
                    self.app_type.append(app_types)
                    self.update_time.append(update_times)
                    self.time_limit = np.append(
                        self.time_limit, np.float64(node_buffer[a]))
                # print(self.slots_duration,self.slots_start,self.queue_limit,self.reserv_start,
                # self.reserv_end,self.total_est_delay,self.app_type,self.update_time,self.time_limit)
                    # print(self.update_time,self.time_limit)
    def process_and_update_queue(self, i, node, core, app_time,
                                 precision_limit):
        """
        Update the queue taking into account the random time variation in the
        processing of applications
        """
        
        # Generate the random variation using a normal distribution with limits
        var = (self.error_var * self.total_est_delay[node][core][i] *
               np.random.randn())
        var = np.around(min(
            var, (self.reserv_end[node][core][i] -
                  self.reserv_start[node][core][i]) * self.upper_var_limit),
            precision_limit)
        var = np.around(max(
            var, - (self.reserv_end[node][core][i] -
                    self.reserv_start[node][core][i]) * self.lower_var_limit),
            precision_limit)
        
        # Next adjacent time slot (if it exists)
        slot = np.where(self.slots_start[node][core] ==
                        self.reserv_end[node][core][i])
        # Next time slot (must exist)
        next_slot = np.where(self.slots_start[node][core] >=
                             self.reserv_end[node][core][i])
        
        # Update reservation
        self.reserv_end[node][core][i] += var
        self.total_est_delay[node][core][i] += var
        
        # If the processing took shorter than expected
        if(var < 0):
            # Update the adjacent time slot if it exists
            if(len(slot[0])):
                self.slots_start[node][core][slot[0][0]] += var
                self.slots_duration[node][core][slot[0][0]] -= var
            else:
                self.slots_start[node][core] = np.insert(
                    self.slots_start[node][core], next_slot[0][0],
                    self.reserv_end[node][core][i])
                self.slots_duration[node][core] = np.insert(
                    self.slots_duration[node][core], next_slot[0][0], -var)
        elif(var > 0): # If the processing took longer than expected
            while(1): # This loop has to reach a break line
                # Check that the queue is still within limits
                if(self.reserv_end[node][core][i] > self.time_limit[node]):
                    self.slots_start[node][core][-1] = (
                        self.reserv_start[node][core][i])
                    self.slots_duration[node][core][-1] = (
                        self.time_limit[node] -
                        self.reserv_start[node][core][i])
                    self.total_est_delay[node][core][i::] = -1
                    self.queue_limit[node][core] += len(
                        range(i, len(self.reserv_start[node][core])))
                    self.reserv_end[node][core] = np.delete(
                        self.reserv_end[node][core], range(
                            i, len(self.reserv_end[node][core])))
                    self.reserv_start[node][core] = np.delete(
                        self.reserv_start[node][core], range(
                            i, len(self.reserv_start[node][core])))
                    self.slots_start[node][core] = np.delete(
                        self.slots_start[node][core], range(
                            next_slot[0][0],
                            len(self.slots_start[node][core])-1))
                    self.slots_duration[node][core] = np.delete(
                        self.slots_duration[node][core], range(
                            next_slot[0][0],
                            len(self.slots_duration[node][core])-1))
                    break
                # Update the adjancent time slot if it exists
                if(len(slot[0])):
                    # Check if the slot can handle the increase in time
                    if(self.slots_duration[node][core][slot[0][0]] > var):
                        self.slots_start[node][core][slot[0][0]] += var
                        self.slots_duration[node][core][slot[0][0]] -= var
                        break
                    elif(self.slots_duration[node][core][slot[0][0]] == var):
                        self.slots_start[node][core] = np.delete(
                            self.slots_start[node][core], slot[0][0])
                        self.slots_duration[node][core] = np.delete(
                            self.slots_duration[node][core], slot[0][0])
                        break
                    else: # If not, delete and update remaining time
                        var -= self.slots_duration[node][core][slot[0][0]]
                        self.slots_start[node][core] = np.delete(
                            self.slots_start[node][core], slot[0][0])
                        self.slots_duration[node][core] = np.delete(
                            self.slots_duration[node][core], slot[0][0])
                # Move on to the next reservation
                i += 1
                # Next adjacent time slot (if it exists)
                slot = np.where(self.slots_start[node][core] ==
                                self.reserv_end[node][core][i])
                # Next time slot (must exist)
                next_slot = np.where(self.slots_start[node][core] >=
                                     self.reserv_end[node][core][i])
                # Update reservation
                self.reserv_start[node][core][i] += var
                self.reserv_end[node][core][i] += var
                self.total_est_delay[node][core][i] += var
                
                # Security check to avoid infinite loop
                if(i > self.reserv_limit):
                    raise KeyboardInterrupt(
                        "Trouble while updating a core queue!")
    
    def set_error_var(self, error_var):
        self.error_var = error_var
    
    def set_upper_var_limit(self, upper_lim):
        self.upper_var_limit = upper_lim
    
    def set_lower_var_limit(self, lower_lim):
        self.lower_var_limit = lower_lim

