from Encode import init_pop
from Data import all_hunger_data, trolley_num, coordinate_data, UW, supply_depot_num, scheduling_num, \
                 scheduling_data_to_excel, fitness_data_to_excel
from Operators import Operators
from Evaluation import decode, evaluation, nondominated_sort
from Gantt import gantt, line_chart

import matplotlib.pyplot as plt
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


def iteration(now_scheduling_cout, hunger_data):
    iter_pond_RS, iter_pond_CD = init_pop(pop_size, hunger_data)
    # manual_RS, manual_CD = init_pop(10, hunger_data)
    operator = Operators(pop_size, iteration_num, pf_max, pf_min, theta, now_scheduling_cout)
    fit_iter, fit_iter1, fit_iter2 = [], [], []
    for now_iteration_num in range(1, iteration_num+1):
        print(f'\rrunning count:{seed_num} ---> scheduling count:{now_scheduling_cout} ---> total iteration{iteration_num} ---> now iteration:{now_iteration_num}', end="")
        # crossover and mutation
        CRS, CCD = [], []
        while len(CRS) < pop_size:
            c_rs, c_cd = [], []
            two_individual_idx = random.sample(range(0, len(iter_pond_RS)), 2)
            p_idx = random.choice(range(0, len(iter_pond_RS)))
            p_rs, p_cd = iter_pond_RS[p_idx], iter_pond_CD[p_idx]
            if random.random() <= pc:
                p1_rs, p1_cd = iter_pond_RS[two_individual_idx[0]], iter_pond_CD[two_individual_idx[0]]
                p2_rs, p2_cd = iter_pond_RS[two_individual_idx[1]], iter_pond_CD[two_individual_idx[1]]
                c_rs, c_cd = operator.pox_crossover(p1_rs, p1_cd, p2_rs, p2_cd)
                c_rs, c_cd = operator.rpx_crossover(c_rs, c_cd, p_rs, p_cd, now_iteration_num)
            if random.random() <= pm:
                if len(c_rs) == 0:
                    c_rs, c_cd = p_rs.copy(), p_cd.copy()
                random_probability = random.uniform(0, 1)
                # if random_probability < 0.33 and len(c_rs) >= 2:
                #     c_rs, c_cd = operator.insertion_mutation(c_rs, c_cd)
                if random_probability < 0.5 and len(c_rs) >= 2:
                    c_rs, c_cd = operator.exchange_mutation(c_rs, c_cd)
                elif len(c_rs) >= 2:
                    c_rs, c_cd = operator.reverse_mutation(c_rs, c_cd)
                c_rs, c_cd = operator.cd_alone_mutation(c_rs, c_cd)
            if len(c_rs) != 0:
                CRS.append(c_rs)
                CCD.append(c_cd)
        comb_RS, comb_CD = iter_pond_RS + CRS, iter_pond_CD + CCD
        comb_fitness_list1, comb_fitness_list2 = [], []
        # greedy operator
        for indiv_idx in range(0, len(comb_RS)):
            rs, cd = comb_RS[indiv_idx], comb_CD[indiv_idx]
            c_rs, c_cd = operator.greed(rs, cd, trolley_available_time, trolley_carrying_capacity,
                                        trolley_coordinate, supply_depot_occupy, scheduling_start_time)
            comb_RS[indiv_idx], comb_CD[indiv_idx] = c_rs, c_cd
        # evaluation
        for rs, cd in zip(comb_RS, comb_CD):
            scheduling_list = decode(rs, cd, trolley_available_time, trolley_carrying_capacity, trolley_coordinate,
                                     supply_depot_occupy, hunger_data)
            fitness1, fitness2 = evaluation(scheduling_list, hunger_data, scheduling_start_time)
            comb_fitness_list1.append(fitness1)
            comb_fitness_list2.append(fitness2)
        all_nondominated_idx = nondominated_sort(comb_fitness_list1, comb_fitness_list2)
        # selection
        iter_pond_RS, iter_pond_CD = operator.selection(comb_fitness_list1, comb_fitness_list2, all_nondominated_idx,
                                                        comb_RS, comb_CD, now_iteration_num)
        fit_iter1.append(comb_fitness_list1[all_nondominated_idx[0][0]])
        fit_iter2.append(comb_fitness_list2[all_nondominated_idx[0][0]])

        # if now_iteration_num == 1 or now_iteration_num == 150 or now_iteration_num == iteration_num:
        #     comb_fitness_list = []
        #     comb_fitness_list.append(comb_fitness_list1)
        #     comb_fitness_list.append(comb_fitness_list2)
        #     path0 = f'../SolutionDistribution/dynamic15_{now_scheduling_cout}_{now_iteration_num}.xlsx'
        #     fitness_data_to_excel(path0, comb_fitness_list)

    fit_iter.append(fit_iter1)
    fit_iter.append(fit_iter2)
    # path1 = f'../FitnessData/dynamic16/ensga/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
    # fitness_data_to_excel(path1, fit_iter)
    path1 = f'../test/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
    fitness_data_to_excel(path1, fit_iter)
    return iter_pond_RS, iter_pond_CD


def main():
    global finish_ongoing_scheduling
    scheduling_list = []
    all_scheduling_list = []
    for now_scheduling_cout in range(1, len(all_hunger_data)+1):
        print('')
        hunger_data = copy.deepcopy(all_hunger_data[now_scheduling_cout - 1])
        if now_scheduling_cout == 1:
            init()
            iter_pond_RS, iter_pond_CD = iteration(now_scheduling_cout, hunger_data)
            optimal_rs, optimal_cd = iter_pond_RS[0], iter_pond_CD[0]
            scheduling_list = decode(optimal_rs, optimal_cd, trolley_available_time, trolley_carrying_capacity,
                                     trolley_coordinate, supply_depot_occupy, hunger_data)
        else:
            finish_ongoing_pond_label, finish_ongoing_scheduling = backtrack_function(now_scheduling_cout, scheduling_list)
            for pond_label in finish_ongoing_pond_label:
                hunger_data[pond_label] = 0
            iter_pond_RS, iter_pond_CD = iteration(now_scheduling_cout, hunger_data)
            optimal_rs, optimal_cd = iter_pond_RS[0], iter_pond_CD[0]
            scheduling_list = decode(optimal_rs, optimal_cd, trolley_available_time, trolley_carrying_capacity,
                                     trolley_coordinate, supply_depot_occupy, hunger_data)
        all_scheduling_list.append(scheduling_list)
        # path1 = f'../SchedulingData/dynamic16/ensga/scheduling{str(now_scheduling_cout)}_{str(seed_num)}.xlsx'
        # scheduling_data_to_excel(path1, scheduling_list)
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

    # for figure_num, scheduling_list in enumerate(all_scheduling_list):
    #     if figure_num != len(all_scheduling_list)-1:
    #         schedule_stime = (figure_num+1)*scheduling_interval
    #     else:
    #         schedule_stime = float('inf')
    #     gantt(figure_num, schedule_stime, scheduling_list)
    # # gantt(scheduling_num, float('inf'), finish_ongoing_scheduling+all_scheduling_list[-1])
    # plt.show()
