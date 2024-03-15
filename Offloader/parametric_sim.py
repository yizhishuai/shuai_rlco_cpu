# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:51:16 2021

@author: Mieszko Ferens
"""

### Script for parametric simulations ###
"""
This script's use is to extract data from simulations with different parameters
in the environment so as to get metrics spanning various scenarios
"""

import os
import numpy as np
import gym
import chainerrl
import matplotlib.pyplot as plt
from datetime import date, datetime
from pathlib import Path

from offloader import train_scenario, test_scenario
from agent_creator import make_training_agents
from heuristic_algorithms import make_heuristic_agents
from graph_creator import makeFigurePlot

# Defined topologies
topologies = ["network_branchless", "network_branchless_v2",
              "network_Valladolid"]
topology_labels = ["Branchless network", "Branchless network v2",
                   "Valladolid's network"]

# Function for parametric simulation of number of vehicles (train per test)
def parametric_sim_vehicles_train_per_test(
        env, topology, n_vehicles, estimation_err_var, upper_var_limit,
        lower_var_limit, gammas=0.995, alg='DDQN', explorators='const',
        epsilons=0.2, repetitions=1):
    
    # Parameter error
    if(not isinstance(n_vehicles, list)):
        raise KeyboardInterrupt("The vehicle variation for the simulation must"
                                " be defined as a list.\nTIP: Check the passed"
                                " parameter.")
    
    ## Run simulations with varying network load, training the agents for each
    ## network scenario
    
    # Set invariable environment parameters
    env.set_error_var(estimation_err_var)
    env.set_upper_var_limit(upper_var_limit)
    env.set_lower_var_limit(lower_var_limit)
    
    # Metrics
    train_avg_total_times = []
    train_avg_agent_times = []
    test_benefit = []
    test_success_rate = []
    
    # Train and test the agents
    for i in range(len(n_vehicles)):
        # Vary network load parameters
        env.set_total_vehicles(n_vehicles[i])
        
        # Create RL agents
        agents = make_training_agents(
            env, gammas, explorators, epsilons, alg, repetitions)
        
        # Add heuristic algorithms imitating RL agents to the training for
        # benchmarks
        heuristic_agents = make_heuristic_agents(env)
        for j in range(len(heuristic_agents)):
            agents.insert(0, [heuristic_agents[j]])
        
        # Get metrics of trained and tested agents
        train_results = train_scenario(env, agents)
        test_results = test_scenario(env, agents)
        train_avg_total_times.append(train_results['train_avg_total_times'])
        train_avg_agent_times.append(train_results['train_avg_agent_times'])
        test_benefit.append(test_results['test_benefit'])
        test_success_rate.append(test_results['test_success_rate'])
        
        # Delete previous agents so new once can be created (unless finished)
        if(i < len(n_vehicles) - 1):
            del agents
    
    # Create the directory (if not created) where the data will be stored
    results_path = "Results/VehicleVar/TrainPerTest/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    ## Plot results
    
    # Reshape data to plot with makeFigurePlot function
    train_avg_total_times = reshape_data(train_avg_total_times)
    train_avg_agent_times = reshape_data(train_avg_agent_times)
    test_benefit = reshape_data(test_benefit)
    test_success_rate = reshape_data(test_success_rate)
    
    # Plot graphs
    labels = ['Vehicles in network', 'Training average total times',
              topology]
    legend = []
    for a in range(len(agents)):
        legend.append(agents[a][0][1])
    
    makeFigurePlot(
        n_vehicles, train_avg_total_times, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Training average agent processing times'
    makeFigurePlot(
        n_vehicles, train_avg_agent_times, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Testing benefit'
    makeFigurePlot(
        n_vehicles, test_benefit, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Testing success rate'
    makeFigurePlot(
        n_vehicles, test_success_rate, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    
    plt.close('all') # Close all figures
    
    ## Log the data of the experiment in a file
    
    # Open log file
    try:
        log_file = open(results_path + "TestLog_" + str(date.today()) + '.txt',
                        'wt', encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Initial information
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + topology + '\n')
    log_file.write("Training type: Train per Test\n")
    
    log_file.write("---------------------------------------------------\n\n")
    
    # Data
    log_file.write('n_vehicles = ' + str(n_vehicles) + '\n')
    for a in range(len(agents)):
        log_file.write("\n---" + agents[a][0][1] + '\n')
        log_file.write("-Training average total times:\n" +
                       str(train_avg_total_times[a]) + '\n')
        log_file.write("-Training average agent processing times:\n" +
                       str(train_avg_agent_times[a]) + '\n')
        log_file.write("-Test benefit:\n" + str(test_benefit[a]) + '\n')
        log_file.write("-Test success rate:\n" + str(test_success_rate[a]) +
                       '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    # .csv
    log_file.write("n_vehicles")
    for a in range(len(agents)):
        for key in train_results.keys():
            if(key != 'agents'):
                log_file.write(',' + key + '-' + agents[a][0][1])
        for key in test_results.keys():
            log_file.write(',' + key + '-' + agents[a][0][1])
    for i in range(len(n_vehicles)):
        log_file.write('\n' + str(n_vehicles[i]) + ',')
        for a in range(len(agents)):
            log_file.write(str(train_avg_total_times[a][i]) + ',')
            log_file.write(str(train_avg_agent_times[a][i]) + ',')
            log_file.write(str(test_benefit[a][i]) + ',')
            log_file.write(str(test_success_rate[a][i]) + ',')
    
    log_file.close() # Close log file

# Function for parametric simulation of number of vehicles (train once)
def parametric_sim_vehicles_train_once(
        env, topology, n_vehicles, train_vehicles, estimation_err_var,
        upper_var_limit, lower_var_limit, gammas=0.995, alg='DDQN',
        explorators='const', epsilons=0.2, repetitions=1):
    
    # Parameter error
    if(not isinstance(n_vehicles, list)):
        raise KeyboardInterrupt("The vehicle variation for the simulation must"
                                " be defined as a list.\nTIP: Check the passed"
                                " parameter.")
    
    ## Run simulations with varying network load, training the agents for each
    ## network scenario
    
    # Set invariable environment parameters
    env.set_error_var(estimation_err_var)
    env.set_upper_var_limit(upper_var_limit)
    env.set_lower_var_limit(lower_var_limit)
    
    # Metrics
    train_avg_total_times = []
    train_avg_agent_times = []
    test_benefit = []
    test_success_rate = []
    
    # Create RL agents
    agents = make_training_agents(
        env, gammas, explorators, epsilons, alg, repetitions)
    
    # Add heuristic algorithms imitating RL agents to the training for
    # benchmarks
    heuristic_agents = make_heuristic_agents(env)
    for i in range(len(heuristic_agents)):
        agents.insert(0, [heuristic_agents[i]])
    
    # Train the agents at selected load
    env.set_total_vehicles(train_vehicles)
    train_results = train_scenario(env, agents)
    # Get metrics of trained agents
    train_avg_total_times.append(train_results['train_avg_total_times'])
    train_avg_agent_times.append(train_results['train_avg_agent_times'])
    
    # Test the agents
    for i in range(len(n_vehicles)):
        # Vary network load parameters
        env.set_total_vehicles(n_vehicles[i])
        
        # Test the agents
        test_results = test_scenario(env, agents)
        # Get metrics of tested agents
        test_benefit.append(test_results['test_benefit'])
        test_success_rate.append(test_results['test_success_rate'])
    
    # Create the directory (if not created) where the data will be stored
    results_path = "Results/VehicleVar/TrainOnce/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    ## Plot results
    
    # Reshape data to plot with makeFigurePlot function
    train_avg_total_times = reshape_data(train_avg_total_times)
    train_avg_agent_times = reshape_data(train_avg_agent_times)
    test_benefit = reshape_data(test_benefit)
    test_success_rate = reshape_data(test_success_rate)
    
    # Plot graphs
    labels = ['Vehicles in network', 'Testing benefit', topology]
    legend = []
    for a in range(len(agents)):
        legend.append(agents[a][0][1])
    
    makeFigurePlot(
        n_vehicles, test_benefit, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Testing success rate'
    makeFigurePlot(
        n_vehicles, test_success_rate, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    
    plt.close('all') # Close all figures
    
    ## Log the data of the experiment in a file
    
    # Open log file
    try:
        log_file = open(results_path + "TestLog_" + str(date.today()) + '.txt',
                        'wt', encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Initial information
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + topology + '\n')
    log_file.write("Training type: Train once\n")
    
    log_file.write("---------------------------------------------------\n\n")
    
    # Data
    log_file.write('train_vehicles = ' + str(train_vehicles) + '\n')
    log_file.write('n_vehicles = ' + str(n_vehicles) + '\n')
    for a in range(len(agents)):
        log_file.write("\n---" + agents[a][0][1] + '\n')
        log_file.write("-Training average total times:\n" +
                       str(train_avg_total_times[a]) + '\n')
        log_file.write("-Training average agent processing times:\n" +
                       str(train_avg_agent_times[a]) + '\n')
        log_file.write("-Test benefit:\n" + str(test_benefit[a]) + '\n')
        log_file.write("-Test success rate:\n" + str(test_success_rate[a]) +
                       '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    # .csv
    log_file.write("n_vehicles")
    for a in range(len(agents)):
        for key in test_results.keys():
            log_file.write(',' + key + '-' + agents[a][0][1])
    for i in range(len(n_vehicles)):
        log_file.write('\n' + str(n_vehicles[i]) + ',')
        for a in range(len(agents)):
            log_file.write(str(test_benefit[a][i]) + ',')
            log_file.write(str(test_success_rate[a][i]) + ',')
    
    log_file.close() # Close log file

# Function for parametric simulation of error variance (train per test)
def parametric_sim_errorVar_train_per_test(
        env, topology, n_vehicles, estimation_err_var, upper_var_limit,
        lower_var_limit, gammas=0.995, alg='DDQN', explorators='const',
        epsilons=0.2, repetitions=1):
    
    # Parameter error
    if(not isinstance(estimation_err_var, list)):
        raise KeyboardInterrupt("The estimation error variance variation for "
                                "the simulation must be defined as a list.\n"
                                "TIP: Check the passed parameter.")
    
    ## Run simulations with varying estimation error variance, training the
    ## agents for each network scenario
    
    # Set invariable environment parameters
    env.set_total_vehicles(n_vehicles)
    env.set_upper_var_limit(upper_var_limit)
    env.set_lower_var_limit(lower_var_limit)
    
    # Metrics
    train_avg_total_times = []
    train_avg_agent_times = []
    test_benefit = []
    test_success_rate = []
    
    # Train and test the agents
    for i in range(len(estimation_err_var)):
        # Vary network load parameters
        env.set_error_var(estimation_err_var[i])
        
        # Create RL agents
        agents = make_training_agents(
            env, gammas, explorators, epsilons, alg, repetitions)
        
        # Add heuristic algorithms imitating RL agents to the training for
        # benchmarks
        heuristic_agents = make_heuristic_agents(env)
        for j in range(len(heuristic_agents)):
            agents.insert(0, [heuristic_agents[j]])
        
        # Get metrics of trained and tested agents
        train_results = train_scenario(env, agents)
        test_results = test_scenario(env, agents)
        train_avg_total_times.append(train_results['train_avg_total_times'])
        train_avg_agent_times.append(train_results['train_avg_agent_times'])
        test_benefit.append(test_results['test_benefit'])
        test_success_rate.append(test_results['test_success_rate'])
        
        # Delete previous agents so new once can be created (unless finished)
        if(i < len(estimation_err_var) - 1):
            del agents
    
    # Create the directory (if not created) where the data will be stored
    results_path = "Results/ErrorVar/TrainPerTest/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    ## Plot results
    
    # Reshape data to plot with makeFigurePlot function
    train_avg_total_times = reshape_data(train_avg_total_times)
    train_avg_agent_times = reshape_data(train_avg_agent_times)
    test_benefit = reshape_data(test_benefit)
    test_success_rate = reshape_data(test_success_rate)
    
    # Plot graphs
    labels = ['Variation coeficient', 'Training average total times',
              topology]
    legend = []
    for a in range(len(agents)):
        legend.append(agents[a][0][1])
    
    makeFigurePlot(estimation_err_var, train_avg_total_times, labels=labels,
                   legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Training average agent processing times'
    makeFigurePlot(estimation_err_var, train_avg_agent_times, labels=labels,
                   legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Testing benefit'
    makeFigurePlot(estimation_err_var, test_benefit, labels=labels,
                   legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Testing success rate'
    makeFigurePlot(estimation_err_var, test_success_rate, labels=labels,
                   legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    
    plt.close('all') # Close all figures
    
    ## Log the data of the experiment in a file
    
    # Open log file
    try:
        log_file = open(results_path + "TestLog_" + str(date.today()) + '.txt',
                        'wt', encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Initial information
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + topology + '\n')
    log_file.write("Training type: Train per Test\n")
    
    log_file.write("---------------------------------------------------\n\n")
    
    # Data
    log_file.write('estimation_err_var = ' + str(estimation_err_var) + '\n')
    for a in range(len(agents)):
        log_file.write("\n---" + agents[a][0][1] + '\n')
        log_file.write("-Training average total times:\n" +
                       str(train_avg_total_times[a]) + '\n')
        log_file.write("-Training average agent processing times:\n" +
                       str(train_avg_agent_times[a]) + '\n')
        log_file.write("-Test benefit:\n" + str(test_benefit[a]) + '\n')
        log_file.write("-Test success rate:\n" + str(test_success_rate[a]) +
                       '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    # .csv
    log_file.write("estimation_err_var")
    for a in range(len(agents)):
        for key in train_results.keys():
            if(key != 'agents'):
                log_file.write(',' + key + '-' + agents[a][0][1])
        for key in test_results.keys():
            log_file.write(',' + key + '-' + agents[a][0][1])
    for i in range(len(estimation_err_var)):
        log_file.write('\n' + str(estimation_err_var[i]) + ',')
        for a in range(len(agents)):
            log_file.write(str(train_avg_total_times[a][i]) + ',')
            log_file.write(str(train_avg_agent_times[a][i]) + ',')
            log_file.write(str(test_benefit[a][i]) + ',')
            log_file.write(str(test_success_rate[a][i]) + ',')
    
    log_file.close() # Close log file

# Function for parametric simulation of error variance (train once)
def parametric_sim_errorVar_train_once(
        env, topology, n_vehicles, estimation_err_var, train_est_err_var,
        upper_var_limit, lower_var_limit, gammas=0.995, alg='DDQN',
        explorators='const', epsilons=0.2, repetitions=1):
    
    # Parameter error
    if(not isinstance(estimation_err_var, list)):
        raise KeyboardInterrupt("The estimation error variance variation for "
                                "the simulation must be defined as a list.\n"
                                "TIP: Check the passed parameter.")
    
    ## Run simulations with varying estimation error variance, training the
    ## agents for each network scenario
    
    # Set invariable environment parameters
    env.set_total_vehicles(n_vehicles)
    env.set_upper_var_limit(upper_var_limit)
    env.set_lower_var_limit(lower_var_limit)
    
    # Metrics
    train_avg_total_times = []
    train_avg_agent_times = []
    test_benefit = []
    test_success_rate = []
    
    # Create RL agents
    agents = make_training_agents(
        env, gammas, explorators, epsilons, alg, repetitions)
    
    # Add heuristic algorithms imitating RL agents to the training for
    # benchmarks
    heuristic_agents = make_heuristic_agents(env)
    for i in range(len(heuristic_agents)):
        agents.insert(0, [heuristic_agents[i]])
    
    # Train the agents at selected load
    env.set_error_var(train_est_err_var)
    train_results = train_scenario(env, agents)
    # Get metrics of trained agents
    train_avg_total_times.append(train_results['train_avg_total_times'])
    train_avg_agent_times.append(train_results['train_avg_agent_times'])
    
    # Test the agents
    for i in range(len(estimation_err_var)):
        # Vary network load parameters
        env.set_error_var(estimation_err_var[i])
        
        # Test the agents
        test_results = test_scenario(env, agents)
        # Get metrics of tested agents
        test_benefit.append(test_results['test_benefit'])
        test_success_rate.append(test_results['test_success_rate'])
    
    # Create the directory (if not created) where the data will be stored
    results_path = "Results/ErrorVar/TrainOnce/"
    Path(results_path).mkdir(parents=True, exist_ok=True)
    
    ## Plot results
    
    # Reshape data to plot with makeFigurePlot function
    train_avg_total_times = reshape_data(train_avg_total_times)
    train_avg_agent_times = reshape_data(train_avg_agent_times)
    test_benefit = reshape_data(test_benefit)
    test_success_rate = reshape_data(test_success_rate)
    
    # Plot graphs
    labels = ['Vehicles in network', 'Testing benefit', topology]
    legend = []
    for a in range(len(agents)):
        legend.append(agents[a][0][1])
    
    makeFigurePlot(
        estimation_err_var, test_benefit, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    labels[1] = 'Testing success rate'
    makeFigurePlot(
        estimation_err_var, test_success_rate, labels=labels, legend=legend)
    plt.savefig(results_path + labels[1] + '.svg')
    
    plt.close('all') # Close all figures
    
    ## Log the data of the experiment in a file
    
    # Open log file
    try:
        log_file = open(results_path + "TestLog_" + str(date.today()) + '.txt',
                        'wt', encoding='utf-8')
    except:
        raise KeyboardInterrupt('Error while initializing log file...aborting')
    
    # Initial information
    log_file.write("Experiment Log - " + str(datetime.today()) + '\n\n')
    log_file.write("Network topology: " + topology + '\n')
    log_file.write("Training type: Train once\n")
    
    log_file.write("---------------------------------------------------\n\n")
    
    # Data
    log_file.write('train_est_err_var = ' + str(train_est_err_var) + '\n')
    log_file.write('estimation_err_var = ' + str(estimation_err_var) + '\n')
    for a in range(len(agents)):
        log_file.write("\n---" + agents[a][0][1] + '\n')
        log_file.write("-Training average total times:\n" +
                       str(train_avg_total_times[a]) + '\n')
        log_file.write("-Training average agent processing times:\n" +
                       str(train_avg_agent_times[a]) + '\n')
        log_file.write("-Test benefit:\n" + str(test_benefit[a]) + '\n')
        log_file.write("-Test success rate:\n" + str(test_success_rate[a]) +
                       '\n')
    
    log_file.write("---------------------------------------------------\n\n")
    # .csv
    log_file.write("estimation_err_var")
    for a in range(len(agents)):
        for key in test_results.keys():
            log_file.write(',' + key + '-' + agents[a][0][1])
    for i in range(len(estimation_err_var)):
        log_file.write('\n' + str(estimation_err_var[i]) + ',')
        for a in range(len(agents)):
            log_file.write(str(test_benefit[a][i]) + ',')
            log_file.write(str(test_success_rate[a][i]) + ',')
    
    log_file.close() # Close log file

# Funtion for reshaping the parametric simulation's results
def reshape_data(data):
    temp = np.array(data)
    reshaped_data = []
    for i in range(len(data[0])):
        reshaped_data.append(list(temp[:,i]))
    
    return reshaped_data

if(__name__ == "__main__"):
    
    path_to_env = "../Environments/offloading-net/offloading_net/envs/"
    
    ## Setup simulation state in temporal file (used for creating the
    ## appropriate environment, i.e. using the correct network topology)
    
    top_index = 0 # Pick one index of the above
    
    ## Define what is the current network topology for simulation in state file
    try:
        state_file = open(path_to_env + "net_topology", 'wt')
    except:
        raise KeyboardInterrupt(
            'Error while initializing state file...aborting')
    
    state_file.write(topologies[top_index])
    state_file.close()
    
    # Checking if the environment is already registered is necessary for
    # subsecuent executions
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for envs in env_dict:
        if 'offload' in envs:
            print('Remove {} from registry'.format(envs))
            del gym.envs.registration.registry.env_specs[envs]
    del env_dict
    
    ## Environment (using gym)
    env = gym.make('offloading_net:offload-v0')
    env = chainerrl.wrappers.CastObservationToFloat32(env)
    
    # Remove temporary file so it's not read in other simulations
    if(os.path.exists(path_to_env + "net_topology")):
        os.remove(path_to_env + "net_topology")
    
    # Simulation parameters
    train_vehicles = 150
    default_vehicles = 50
    n_vehicles = [10, 30, 50]
    
    train_est_err_var = 0
    default_est_err_var = 0
    estimation_err_var = [0, 0.5, 1]
    
    upper_var_limit = 0.5
    lower_var_limit = 0.5
    
    ## Default agents parameters
    # Discount factors
    # Note that for comparing average Q values, gamma should be equal for
    # all agents because this parameter influences their calculation.
    gammas = 0.995
    
    # Algorithms to be used
    alg = ['DDQN','TRPO']
    
    # Explorations that are to be analized (in algorithms that use them)
    explorators = 'const'
    epsilons = 0.1
    
    # Define the number of replicas
    repetitions = 1
    
    # Run simulations
    parametric_sim_vehicles_train_per_test(
        env, env.topology_label, n_vehicles, default_est_err_var,
        upper_var_limit, lower_var_limit, gammas=gammas, alg=alg,
        explorators=explorators, epsilons=epsilons, repetitions=repetitions)
    
    parametric_sim_vehicles_train_once(
        env, env.topology_label, n_vehicles, train_vehicles,
        default_est_err_var, upper_var_limit, lower_var_limit, gammas=gammas,
        alg=alg, explorators=explorators, epsilons=epsilons,
        repetitions=repetitions)
    
    parametric_sim_errorVar_train_per_test(
        env, env.topology_label, default_vehicles, estimation_err_var,
        upper_var_limit, lower_var_limit, gammas=gammas, alg=alg,
        explorators=explorators, epsilons=epsilons, repetitions=repetitions)
    
    parametric_sim_errorVar_train_once(
        env, env.topology_label, default_vehicles, estimation_err_var,
        train_est_err_var, upper_var_limit, lower_var_limit, gammas=gammas,
        alg=alg, explorators=explorators, epsilons=epsilons,
        repetitions=repetitions)

