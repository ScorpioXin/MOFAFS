from Encode import init_pop
from Data import all_hunger_data, trolley_num, coordinate_data, UW, supply_depot_num, scheduling_num, \
                 scheduling_data_to_excel, fitness_data_to_excel, available_trolley_data
from Evaluation import decode, evaluation, nondominated_sort

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import copy
import time


def init():
    global trolley_available_time, trolley_carrying_capacity, trolley_coordinate, supply_depot_occupy
    global finish_ongoing_pond_label, finish_ongoing_scheduling, scheduling_start_time
    for _ in range(1, trolley_num + 1):
        trolley_available_time[_], trolley_carrying_capacity[_] = 0, UW
        trolley_coordinate[_] = coordinate_data['parking' + str(_)]
    for _ in range(1, supply_depot_num + 1):
        supply_depot_occupy[_] = [0, 0, 0]
    finish_ongoing_pond_label = []
    finish_ongoing_scheduling = []
    scheduling_start_time = 0


def backtrack_function(now_scheduling_cout, scheduling_list):
    global trolley_available_time, trolley_carrying_capacity, trolley_coordinate, supply_depot_occupy
    global scheduling_interval, scheduling_start_time, finish_ongoing_pond_label, finish_ongoing_scheduling
    scheduling_start_time = (now_scheduling_cout-1) * scheduling_interval
    for scheduling in scheduling_list:
        if scheduling[3] <= scheduling_start_time or scheduling[2] < scheduling_start_time < scheduling[3]:
            trolley_label = scheduling[0]
            if scheduling[3] > trolley_available_time[trolley_label]:
                trolley_available_time[trolley_label] = scheduling[3]
                trolley_carrying_capacity[trolley_label] = scheduling[4]
                trolley_coordinate[trolley_label] = coordinate_data[scheduling[1]]
            if scheduling[1][:4] == 'pond':
                finish_ongoing_pond_label.append(int(scheduling[1][4:]))
            else:
                depot_label = int(scheduling[1][5:])
                if min(supply_depot_occupy[depot_label]) < scheduling[3]:
                    supply_depot_occupy[depot_label][supply_depot_occupy[depot_label].index(min(supply_depot_occupy[1]))] = scheduling[3]
            finish_ongoing_scheduling.append(scheduling)
    return finish_ongoing_pond_label, finish_ongoing_scheduling


def update_historical_optim(history_fitness_list1, history_fitness_list2, new_fitness_list1, new_fitness_list2, history_indiv_optim_RS, history_indiv_optim_CD,
                            new_RS, new_CD, ind_optim_pos_RS, ind_optim_pos_CD, new_rs_position, new_cd_position):
    alpha, beta = 0.55, 0.45
    f1 = history_fitness_list1 + new_fitness_list1
    f2 = history_fitness_list2 + new_fitness_list2
    com_RS = history_indiv_optim_RS + new_RS
    com_CD = history_indiv_optim_CD + new_CD
    fmax1, fmin1 = max(f1), min(f1)
    fmax2, fmin2 = max(f2), min(f2)
    if fmax1 != fmin1 and fmax2 != fmin2:
        normal_fit_list = [alpha * (fitness1 - fmin1) / (fmax1 - fmin1) + beta * (fitness2 - fmin2) / (fmax2 - fmin2) for fitness1, fitness2 in zip(f1, f2)]
    elif fmax1 == fmin1 and fmax2 != fmin2:
        normal_fit_list = [beta * (fitness2 - fmin2) / (fmax2 - fmin2) for fitness2 in f2]
    elif fmax1 != fmin1 and fmax2 == fmin2:
        normal_fit_list = [alpha * (fitness1 - fmin1) / (fmax1 - fmin1) for fitness1 in f1]
    length = int(len(normal_fit_list)/2)
    his_norm_fit = normal_fit_list[:length]
    new_norm_fit = normal_fit_list[-length:]
    for idx in range(length):
        if his_norm_fit[idx] > new_norm_fit[idx]:
            history_fitness_list1[idx] = new_fitness_list1[idx]
            history_fitness_list2[idx] = new_fitness_list2[idx]
            history_indiv_optim_RS[idx] = new_RS[idx]
            history_indiv_optim_CD[idx] = new_CD[idx]
            ind_optim_pos_RS[idx] = new_rs_position[idx]
            ind_optim_pos_CD[idx] = new_cd_position[idx]
    optim_idx = normal_fit_list.index(min(normal_fit_list))
    global_optim_fit1 = f1[optim_idx]
    global_optim_fit2 = f2[optim_idx]
    history_global_optim_RS = com_RS[optim_idx]
    history_global_optim_CD = com_CD[optim_idx]
    if optim_idx <= length-1:
        glo_optim_pos_RS = ind_optim_pos_RS[optim_idx]
        glo_optim_pos_CD = ind_optim_pos_CD[optim_idx]
    else:
        glo_optim_pos_RS = new_rs_position[optim_idx-length]
        glo_optim_pos_CD = new_cd_position[optim_idx-length]
    return history_fitness_list1, history_fitness_list2, history_indiv_optim_RS, history_indiv_optim_CD, history_global_optim_RS, history_global_optim_CD, \
           global_optim_fit1, global_optim_fit2, ind_optim_pos_RS, ind_optim_pos_CD, glo_optim_pos_RS, glo_optim_pos_CD


def location_coding_conversion(pond_number, rs_position_set, cd_position_set):
    new_RS = np.zeros_like(rs_position_set, dtype=int)
    new_CD = np.zeros_like(cd_position_set, dtype=int)

    for idx, rs_position in enumerate(rs_position_set):
        rank_indices = np.argsort(rs_position)
        new_RS[idx] = pond_number[rank_indices]

    for idx, cd_position in enumerate(cd_position_set):
        for pos, trolley in enumerate(cd_position):
            if int(abs(trolley)) in available_trolley_data[new_RS[idx][pos]]:
                new_CD[idx][pos] = int(abs(trolley))
            else:
                new_CD[idx][pos] = random.choice(available_trolley_data[new_RS[idx][pos]])
            # avail_trolley_num = len(available_trolley_data[new_RS[idx][pos]])
            # new_CD[idx][pos] = available_trolley_data[new_RS[idx][pos]][int(abs(trolley) % avail_trolley_num)]
    return new_RS.tolist(), new_CD.tolist()


def coding_location_conversion(RS, CD, position):
    RS = np.array(RS)
    CD = np.array(CD)
    rs_position_set = np.zeros_like(RS)
    cd_position_set = np.zeros_like(CD)

    length = len(RS[0])
    linspace = np.linspace(-position, position, length)
    for idx, rs in enumerate(RS):
        rank_indices = np.argsort(rs)
        rs_position_set[idx] = linspace[rank_indices]
        cd_position_set[idx] = CD[idx]
    return rs_position_set, cd_position_set


def iteration(now_scheduling_cout, hunger_data, w=0.3, c1=0.8, c2=1.7, velocity=3, position=10):
    iter_pond_RS, iter_pond_CD = init_pop(pop_size, hunger_data)
    pond_number = np.array(iter_pond_RS[0], dtype=int)
    rs_position_set, cd_position_set = coding_location_conversion(iter_pond_RS, iter_pond_CD, position)
    velo_RS = np.random.uniform(-velocity, velocity, (len(iter_pond_RS), len(iter_pond_RS[0])))
    velo_CD = np.random.uniform(-velocity, velocity, (len(iter_pond_CD), len(iter_pond_CD[0])))
    fitness_list1, fitness_list2 = [], []
    for rs, cd in zip(iter_pond_RS, iter_pond_CD):
        scheduling_list = decode(rs, cd, trolley_available_time, trolley_carrying_capacity, trolley_coordinate,
                                 supply_depot_occupy, hunger_data)
        fitness1, fitness2 = evaluation(scheduling_list, hunger_data, scheduling_start_time)
        fitness_list1.append(fitness1)
        fitness_list2.append(fitness2)
    history_fitness_list1, history_fitness_list2, history_indiv_optim_RS, history_indiv_optim_CD, history_global_optim_RS, history_global_optim_CD, global_optim_fit1, global_optim_fit2, ind_optim_pos_RS, ind_optim_pos_CD, glo_optim_pos_RS, glo_optim_pos_CD = \
        update_historical_optim(fitness_list1, fitness_list2, fitness_list1, fitness_list2, iter_pond_RS, iter_pond_CD, iter_pond_RS, iter_pond_CD, rs_position_set, cd_position_set, rs_position_set, cd_position_set)

    fit_iter, fit_iter1, fit_iter2 = [], [], []
    for now_iteration_num in range(1, iteration_num+1):
        print(f'\rrunning count:{seed_num} ---> scheduling count:{now_scheduling_cout} ---> total iteration{iteration_num} ---> now iteration:{now_iteration_num}', end="")

        velo_RS = w*velo_RS + c1*random.random()*(ind_optim_pos_RS-rs_position_set) + c2*random.random()*(glo_optim_pos_RS-rs_position_set)
        velo_CD = w*velo_CD + c1*random.random()*(ind_optim_pos_CD-cd_position_set) + c2*random.random()*(glo_optim_pos_CD-cd_position_set)
        velo_RS = np.where((velo_RS < -velocity) | (velo_RS > velocity), np.random.uniform(-velocity, velocity, velo_RS.shape), velo_RS)
        velo_CD = np.where((velo_CD < -velocity) | (velo_CD > velocity), np.random.uniform(-velocity, velocity, velo_CD.shape), velo_CD)
        # print(velo_RS)
        rs_position_set = rs_position_set + velo_RS
        cd_position_set = cd_position_set + velo_CD
        rs_position_set = np.where((rs_position_set < -position) | (rs_position_set > position), np.random.uniform(-position, position, rs_position_set.shape), rs_position_set)
        cd_position_set = np.where((cd_position_set < -position) | (cd_position_set > position), np.random.uniform(-position, position, cd_position_set.shape), cd_position_set)

        new_fitness_list1, new_fitness_list2 = [], []
        new_RS, new_CD = location_coding_conversion(pond_number, rs_position_set, cd_position_set)
        # print(f"\n{new_RS}")
        # print(new_CD)
        for rs, cd in zip(new_RS, new_CD):
            scheduling_list = decode(rs, cd, trolley_available_time, trolley_carrying_capacity, trolley_coordinate,
                                     supply_depot_occupy, hunger_data)
            new_fitness1, new_fitness2 = evaluation(scheduling_list, hunger_data, scheduling_start_time)
            new_fitness_list1.append(new_fitness1)
            new_fitness_list2.append(new_fitness2)

        history_fitness_list1, history_fitness_list2, history_indiv_optim_RS, history_indiv_optim_CD, history_global_optim_RS, history_global_optim_CD, global_optim_fit1, global_optim_fit2, ind_optim_pos_RS, ind_optim_pos_CD, glo_optim_pos_RS, glo_optim_pos_CD = \
            update_historical_optim(history_fitness_list1, history_fitness_list2, new_fitness_list1, new_fitness_list2, history_indiv_optim_RS, history_indiv_optim_CD, new_RS, new_CD, ind_optim_pos_RS, ind_optim_pos_CD, rs_position_set, cd_position_set)
        fit_iter1.append(global_optim_fit1)
        fit_iter2.append(global_optim_fit2)

        # if now_iteration_num == 1 or now_iteration_num == 300:
        #     print(f'\nfitness_list1 = {new_fitness_list1}\nfitness_list2 = {new_fitness_list2}')

    fit_iter.append(fit_iter1)
    fit_iter.append(fit_iter2)
    path1 = f'../FitnessData/dynamic16/pso/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
    fitness_data_to_excel(path1, fit_iter)
    return history_global_optim_RS, history_global_optim_CD


def main():
    global finish_ongoing_scheduling
    scheduling_list = []
    all_scheduling_list = []
    for now_scheduling_cout in range(1, len(all_hunger_data)+1):
        print('')
        hunger_data = copy.deepcopy(all_hunger_data[now_scheduling_cout - 1])
        if now_scheduling_cout == 1:
            init()
            optimal_rs, optimal_cd = iteration(now_scheduling_cout, hunger_data)
            scheduling_list = decode(optimal_rs, optimal_cd, trolley_available_time, trolley_carrying_capacity,
                                     trolley_coordinate, supply_depot_occupy, hunger_data)
        else:
            finish_ongoing_pond_label, finish_ongoing_scheduling = backtrack_function(now_scheduling_cout, scheduling_list)
            for pond_label in finish_ongoing_pond_label:
                hunger_data[pond_label] = 0
            optimal_rs, optimal_cd = iteration(now_scheduling_cout, hunger_data)
            scheduling_list = decode(optimal_rs, optimal_cd, trolley_available_time, trolley_carrying_capacity,
                                     trolley_coordinate, supply_depot_occupy, hunger_data)
        all_scheduling_list.append(scheduling_list)
        path1 = f'../SchedulingData/dynamic16/pso/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
        scheduling_data_to_excel(path1, scheduling_list)
    return all_scheduling_list, finish_ongoing_scheduling


trolley_available_time, trolley_carrying_capacity, trolley_coordinate = {}, {}, {}
supply_depot_occupy = {}
finish_ongoing_pond_label = []
finish_ongoing_scheduling = []


pop_size = 50
iteration_num = 300
theta = 150
pf_max = 1
pf_min = 0.2
pc = 0.9
pm = 0.2
scheduling_interval = 200
scheduling_start_time = 0


if __name__ == "__main__":
    com_scheduling_list = []
    stime = time.time()

    for seed_num in range(1, 21):
        random.seed(seed_num)
        all_scheduling_list, finish_ongoing_scheduling = main()
    print(f'\n{time.time()-stime}')

