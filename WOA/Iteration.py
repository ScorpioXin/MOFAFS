from Encode import init_pop
from Data import all_hunger_data, trolley_num, coordinate_data, UW, supply_depot_num, scheduling_num, \
                 scheduling_data_to_excel, fitness_data_to_excel, available_trolley_data
from Evaluation import decode, evaluation, nondominated_sort

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import itertools
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


def location_coding_conversion(pond_number, rs_position_set, cd_position_set):
    new_RS = np.zeros_like(rs_position_set, dtype=int)
    new_CD = np.zeros_like(cd_position_set, dtype=int)

    for idx, rs_position in enumerate(rs_position_set):
        rank_indices = np.argsort(rs_position)
        new_RS[idx][rank_indices] = pond_number

    for idx, cd_position in enumerate(cd_position_set):
        for pos, trolley in enumerate(cd_position):
            if int(abs(trolley)) in available_trolley_data[new_RS[idx][pos]]:
                new_CD[idx][pos] = int(abs(trolley))
            else:
                new_CD[idx][pos] = random.choice(available_trolley_data[new_RS[idx][pos]])
    return new_RS.tolist(), new_CD.tolist()


def coding_location_conversion(RS, CD, position):
    RS = np.array(RS)
    CD = np.array(CD)
    rs_position_set = np.zeros_like(RS, dtype=float)
    cd_position_set = np.zeros_like(CD, dtype=float)

    length = len(RS[0])
    linspace = np.linspace(-position, position, length)
    for idx, rs in enumerate(RS):
        rank_indices = np.argsort(rs)
        rs_position_set[idx][rank_indices] = linspace
        cd_position_set[idx] = CD[idx]
    return rs_position_set, cd_position_set


def update_position(now_iter_num, total_iter_num, rear_num, best_pos_rs, best_pos_cd, rs_pos_set, cd_pos_set, spiral=1):
    a = 2 - 2 * now_iter_num / total_iter_num
    new_pos_rs = np.zeros_like(rs_pos_set, dtype=float)
    new_pos_cd = np.zeros_like(cd_pos_set, dtype=float)
    for idx in range(pop_size):
        cur_pos_rs = rs_pos_set[idx]
        cur_pos_cd = cd_pos_set[idx]
        if random.random() < 0.5:
            A_rs = 2 * a * np.random.rand(rear_num) - a
            A_cd = 2 * a * np.random.rand(rear_num) - a
            C_rs = 2 * np.random.rand(rear_num)
            C_cd = 2 * np.random.rand(rear_num)
            if (np.linalg.norm(A_rs)+np.linalg.norm(A_cd))/0.5 < 1:
                D_rs = abs(C_rs * best_pos_rs - cur_pos_rs)
                D_cd = abs(C_cd * best_pos_cd - cur_pos_cd)
                pos_rs = best_pos_rs - A_rs * D_rs
                pos_cd = best_pos_cd - A_cd * D_cd
            else:
                rand_idx = random.randint(0, pop_size-1)
                D_rs = abs(C_rs * rs_pos_set[rand_idx] - cur_pos_rs)
                D_cd = abs(C_cd * cd_pos_set[rand_idx] - cur_pos_cd)
                pos_rs = rs_pos_set[rand_idx] - A_rs * D_rs
                pos_cd = cd_pos_set[rand_idx] - A_cd * D_cd
        else:
            D_rs = abs(best_pos_rs - rs_pos_set[idx])
            D_cd = abs(best_pos_cd - cd_pos_set[idx])
            L_rs = np.random.uniform(-1.0, 1.0, size=rear_num)
            L_cd = np.random.uniform(-1.0, 1.0, size=rear_num)
            pos_rs = D_rs * np.exp(spiral*L_rs) * np.cos(2*np.pi*L_rs) - best_pos_rs
            pos_cd = D_cd * np.exp(spiral*L_cd) * np.cos(2*np.pi*L_cd) - best_pos_cd
        new_pos_rs[idx] = pos_rs
        new_pos_cd[idx] = pos_cd
    return new_pos_rs, new_pos_cd


def selection(comb_fit1, comb_fit2, comb_pos_rs, comb_pos_cd, com_rs, com_cd, alpha=0.55, beta=0.45):
    fmax1, fmin1 = max(comb_fit1), min(comb_fit1)
    fmax2, fmin2 = max(comb_fit2), min(comb_fit2)
    if fmax1 != fmin1 and fmax2 != fmin2:
        normal_fit_list = [alpha * (fitness1 - fmin1) / (fmax1 - fmin1) + beta * (fitness2 - fmin2) / (fmax2 - fmin2) for fitness1, fitness2 in zip(comb_fit1, comb_fit2)]
    elif fmax1 == fmin1 and fmax2 != fmin2:
        normal_fit_list = [beta * (fitness2 - fmin2) / (fmax2 - fmin2) for fitness2 in comb_fit2]
    elif fmax1 != fmin1 and fmax2 == fmin2:
        normal_fit_list = [alpha * (fitness1 - fmin1) / (fmax1 - fmin1) for fitness1 in comb_fit1]
    else:
        normal_fit_list = [0.5 for _ in comb_fit1]
    rank_idx = np.argsort(np.array(normal_fit_list))
    pos_rs = comb_pos_rs[rank_idx[:pop_size]]
    pos_cd = comb_pos_cd[rank_idx[:pop_size]]
    iter_rs = np.array(com_rs, dtype=int)[rank_idx[:pop_size]].tolist()
    iter_cd = np.array(com_cd, dtype=int)[rank_idx[:pop_size]].tolist()
    best_pos_rs = comb_pos_rs[rank_idx[0]]
    best_pos_cd = comb_pos_cd[rank_idx[0]]
    optim_fit1, optim_fit2 = comb_fit1[rank_idx[0]], comb_fit2[rank_idx[0]]
    optim_rs, optim_cd = com_rs[rank_idx[0]], com_cd[rank_idx[0]]
    return pos_rs, pos_cd, best_pos_rs, best_pos_cd, optim_fit1, optim_fit2, optim_rs, optim_cd, iter_rs, iter_cd


def iteration(now_scheduling_cout, hunger_data, position=10):
    iter_pond_RS, iter_pond_CD = init_pop(pop_size, hunger_data)
    pond_number = np.array(sorted(iter_pond_RS[0]), dtype=int)
    rear_num = len(pond_number)
    rs_position_set, cd_position_set = coding_location_conversion(iter_pond_RS, iter_pond_CD, position)
    fitness_list1, fitness_list2 = [], []
    for rs, cd in zip(iter_pond_RS, iter_pond_CD):
        scheduling_list = decode(rs, cd, trolley_available_time, trolley_carrying_capacity, trolley_coordinate, supply_depot_occupy, hunger_data)
        fitness1, fitness2 = evaluation(scheduling_list, hunger_data, scheduling_start_time)
        fitness_list1.append(fitness1)
        fitness_list2.append(fitness2)
    w, ww, best_pos_rs, best_pos_cd, init_fit1, init_fit2, optim_rs, optim_cd, x, xx = selection(fitness_list1, fitness_list2, rs_position_set, cd_position_set, iter_pond_RS, iter_pond_CD)

    fit_iter, fit_iter1, fit_iter2 = [], [], []
    for now_iteration_num in range(1, iteration_num+1):
        print(f'\rrunning count:{seed_num} ---> scheduling count:{now_scheduling_cout} ---> total iteration{iteration_num} ---> now iteration:{now_iteration_num}', end="")

        new_pos_rs, new_pos_cd = update_position(now_iteration_num, iteration_num, rear_num, best_pos_rs, best_pos_cd,  rs_position_set, cd_position_set)
        new_pos_rs = np.where((new_pos_rs < -position) | (new_pos_rs > position), np.random.uniform(-position, position, new_pos_rs.shape), new_pos_rs)
        new_pos_cd = np.where((new_pos_cd < -position) | (new_pos_cd > position), np.random.uniform(-position, position, new_pos_cd.shape), new_pos_cd)
        comb_pos_rs = np.concatenate((rs_position_set, new_pos_rs), 0)
        comb_pos_cd = np.concatenate((cd_position_set, new_pos_cd), 0)

        comb_fitness_list1, comb_fitness_list2 = [], []
        new_RS, new_CD = location_coding_conversion(pond_number, new_pos_rs, new_pos_cd)
        comb_RS = iter_pond_RS + new_RS
        comb_CD = iter_pond_CD + new_CD
        for rs, cd in zip(comb_RS, comb_CD):
            scheduling_list = decode(rs, cd, trolley_available_time, trolley_carrying_capacity, trolley_coordinate, supply_depot_occupy, hunger_data)
            fitness1, fitness2 = evaluation(scheduling_list, hunger_data, scheduling_start_time)
            comb_fitness_list1.append(fitness1)
            comb_fitness_list2.append(fitness2)
        rs_position_set, cd_position_set, best_pos_rs, best_pos_cd, optim_fit1, optim_fit2, optim_rs, optim_cd, iter_pond_RS, iter_pond_CD =\
            selection(comb_fitness_list1, comb_fitness_list2, comb_pos_rs, comb_pos_cd, comb_RS, comb_CD)

        fit_iter1.append(optim_fit1)
        fit_iter2.append(optim_fit2)
    fit_iter.append(fit_iter1)
    fit_iter.append(fit_iter2)
    path1 = f'../FitnessData/dynamic16/woa/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
    fitness_data_to_excel(path1, fit_iter)
    return optim_rs, optim_cd, init_fit1, init_fit2


def main():
    global finish_ongoing_scheduling
    scheduling_list = []
    init_fit11, init_fit12, init_fit21, init_fit22 = 0, 0, 0, 0
    for now_scheduling_cout in range(1, len(all_hunger_data)+1):
        print('')
        hunger_data = copy.deepcopy(all_hunger_data[now_scheduling_cout - 1])
        if now_scheduling_cout == 1:
            init()
            optimal_rs, optimal_cd, init_fit11, init_fit12 = iteration(now_scheduling_cout, hunger_data)
            scheduling_list = decode(optimal_rs, optimal_cd, trolley_available_time, trolley_carrying_capacity,
                                     trolley_coordinate, supply_depot_occupy, hunger_data)
        else:
            finish_ongoing_pond_label, finish_ongoing_scheduling = backtrack_function(now_scheduling_cout, scheduling_list)
            for pond_label in finish_ongoing_pond_label:
                hunger_data[pond_label] = 0
            optimal_rs, optimal_cd, init_fit21, init_fit22 = iteration(now_scheduling_cout, hunger_data)
            scheduling_list = decode(optimal_rs, optimal_cd, trolley_available_time, trolley_carrying_capacity,
                                     trolley_coordinate, supply_depot_occupy, hunger_data)
        path1 = f'../SchedulingData/dynamic16/woa/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
        scheduling_data_to_excel(path1, scheduling_list)
    return init_fit11, init_fit12, init_fit21, init_fit22


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

    comfit1s, comfit2s = [], []
    fit11, fit12, fit21, fit22 = [], [], [], []
    for seed_num in range(1, 21):
        random.seed(seed_num)
        init_fit11, init_fit12, init_fit21, init_fit22 = main()

    #     fit11.append(init_fit11)
    #     fit12.append(init_fit12)
    #     if init_fit21 != 0 and init_fit22 != 0:
    #         fit21.append(init_fit21)
    #         fit22.append(init_fit22)
    # comfit1s.append(fit11)
    # comfit1s.append(fit12)
    # comfit2s.append(fit21)
    # comfit2s.append(fit22)
    #
    # path1 = f'../FitnessData/dynamic16/first_init_fitness.xlsx'
    # fitness_data_to_excel(path1, comfit1s)
    # if len(comfit2s[0]) != 0:
    #     path2 = f'../FitnessData/dynamic16/second_init_fitness.xlsx'
    #     fitness_data_to_excel(path2, comfit2s)

    print(f'\n{time.time()-stime}')

