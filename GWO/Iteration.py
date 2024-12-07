from Encode import init_pop
from Data import all_hunger_data, trolley_num, coordinate_data, UW, supply_depot_num, scheduling_data_to_excel, fitness_data_to_excel, available_trolley_data
from Evaluation import decode, evaluation

import numpy as np
import random
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


def update_position(now_iter_num, total_iter_num, rear_num, a_pos_rs, a_pos_cd, b_pos_rs, b_pos_cd, d_pos_rs, d_pos_cd, rs_pos_set, cd_pos_set):
    a = 2 - 2 * now_iter_num / total_iter_num
    A1_rs = 2 * a * np.random.rand(pop_size, rear_num) - a
    A1_cd = 2 * a * np.random.rand(pop_size, rear_num) - a
    A2_rs = 2 * a * np.random.rand(pop_size, rear_num) - a
    A2_cd = 2 * a * np.random.rand(pop_size, rear_num) - a
    A3_rs = 2 * a * np.random.rand(pop_size, rear_num) - a
    A3_cd = 2 * a * np.random.rand(pop_size, rear_num) - a
    C1_rs = 2 * np.random.rand(pop_size, rear_num)
    C1_cd = 2 * np.random.rand(pop_size, rear_num)
    C2_rs = 2 * np.random.rand(pop_size, rear_num)
    C2_cd = 2 * np.random.rand(pop_size, rear_num)
    C3_rs = 2 * np.random.rand(pop_size, rear_num)
    C3_cd = 2 * np.random.rand(pop_size, rear_num)
    Da_rs = abs(C1_rs * a_pos_rs - rs_pos_set)
    Da_cd = abs(C1_cd * a_pos_cd - cd_pos_set)
    Db_rs = abs(C2_rs * b_pos_rs - rs_pos_set)
    Db_cd = abs(C2_cd * b_pos_cd - cd_pos_set)
    Dd_rs = abs(C3_rs * d_pos_rs - rs_pos_set)
    Dd_cd = abs(C3_cd * d_pos_cd - cd_pos_set)
    X1_rs = a_pos_rs - A1_rs * Da_rs
    X1_cd = a_pos_cd - A1_cd * Da_cd
    X2_rs = b_pos_rs - A2_rs * Db_rs
    X2_cd = b_pos_cd - A2_cd * Db_cd
    X3_rs = d_pos_rs - A3_rs * Dd_rs
    X3_cd = d_pos_cd - A3_cd * Dd_cd
    new_pos_rs = (X1_rs + X2_rs + X3_rs) / 3
    new_pos_cd = (X1_cd + X2_cd + X3_cd) / 3
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
    alpha_rs = comb_pos_rs[rank_idx[0]]
    alpha_cd = comb_pos_cd[rank_idx[0]]
    beta_rs = comb_pos_rs[rank_idx[1]]
    beta_cd = comb_pos_cd[rank_idx[1]]
    delta_rs = comb_pos_rs[rank_idx[2]]
    delta_cd = comb_pos_cd[rank_idx[2]]
    optim_fit1, optim_fit2 = comb_fit1[rank_idx[0]], comb_fit2[rank_idx[0]]
    optim_rs, optim_cd = com_rs[rank_idx[0]], com_cd[rank_idx[0]]
    return pos_rs, pos_cd, alpha_rs, alpha_cd, beta_rs, beta_cd, delta_rs, delta_cd, optim_fit1, optim_fit2, optim_rs, optim_cd, iter_rs, iter_cd


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
    w, ww, alpha_pos_rs, alpha_pos_cd, beta_pos_rs, beta_pos_cd, delta_pos_rs, delta_pos_cd, z, zz, optim_rs, optim_cd, f, ff = \
        selection(fitness_list1, fitness_list2, rs_position_set, cd_position_set, iter_pond_RS, iter_pond_CD)

    fit_iter, fit_iter1, fit_iter2 = [], [], []
    for now_iteration_num in range(1, iteration_num+1):
        print(f'\rrunning count:{seed_num} ---> scheduling count:{now_scheduling_cout} ---> total iteration{iteration_num} ---> now iteration:{now_iteration_num}', end="")

        new_pos_rs, new_pos_cd = update_position(now_iteration_num, iteration_num, rear_num, alpha_pos_rs, alpha_pos_cd, beta_pos_rs, beta_pos_cd,
                                                 delta_pos_rs, delta_pos_cd,  rs_position_set, cd_position_set)
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
        rs_position_set, cd_position_set, alpha_pos_rs, alpha_pos_cd, beta_pos_rs, beta_pos_cd, delta_pos_rs, delta_pos_cd, optim_fit1, optim_fit2, optim_rs, optim_cd, iter_pond_RS, iter_pond_CD =\
            selection(comb_fitness_list1, comb_fitness_list2, comb_pos_rs, comb_pos_cd, comb_RS, comb_CD)

        fit_iter1.append(optim_fit1)
        fit_iter2.append(optim_fit2)
    fit_iter.append(fit_iter1)
    fit_iter.append(fit_iter2)
    path1 = f'../FitnessData/dynamic16/gwo/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
    fitness_data_to_excel(path1, fit_iter)
    return optim_rs, optim_cd


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
        path1 = f'../SchedulingData/dynamic16/gwo/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
        scheduling_data_to_excel(path1, scheduling_list)
    return all_scheduling_list, finish_ongoing_scheduling


trolley_available_time, trolley_carrying_capacity, trolley_coordinate = {}, {}, {}
supply_depot_occupy = {}
finish_ongoing_pond_label = []
finish_ongoing_scheduling = []


pop_size = 50
iteration_num = 300
scheduling_interval = 200
scheduling_start_time = 0


if __name__ == "__main__":
    com_scheduling_list = []
    stime = time.time()

    for seed_num in range(1, 21):
        random.seed(seed_num)
        all_scheduling_list, finish_ongoing_scheduling = main()
    print(f'\n{time.time()-stime}')
