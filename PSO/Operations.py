import random

from Data import available_trolley_data

import numpy as np

def update_historical_optim(history_fitness_list1, history_fitness_list2, new_fitness_list1, new_fitness_list2,
                            history_indiv_optim_RS, history_indiv_optim_CD, new_RS, new_CD):
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
    optim_idx = normal_fit_list.index(min(normal_fit_list))
    history_global_optim_RS = com_RS[optim_idx]
    history_global_optim_CD = com_CD[optim_idx]
    return history_fitness_list1, history_fitness_list2, history_indiv_optim_RS, history_indiv_optim_CD, history_global_optim_RS, history_global_optim_CD


def location_coding_conversion(pond_number, rs_position_set, cd_position_set):
    pond_number = np.array(pond_number, dtype=int)
    rs_position_set = np.array(rs_position_set)
    cd_position_set = np.array(cd_position_set)
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
    return new_RS.tolist(), new_CD.tolist()


def coding_location_conversion(RS, CD, position=5):
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

