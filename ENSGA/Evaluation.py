from Data import coordinate_data, coefficient_data, trolley_num, hunger_threshold, supply_depot_num, w, l, h, UW, LW,  SP, SF, SL
from Encode import dynamic_urgency

import copy


def manhattan_distance(position1, position2):
    if position1[0] == coordinate_data['parking1'][0] or position2[0] == coordinate_data['parking1'][0]:
        flag = False
    else:
        flag = True
    if position1[1] == position2[1]:
        if flag:
            distance = abs(position1[0]-position2[0]) + 2*w + l
        else:
            distance = abs(position1[0]-position2[0]) + w + l
    elif position1[0] == position2[0]:
        if flag:
            distance = abs(position1[1]-position2[1]) + w
        else:
            distance = abs(position1[1]-position2[1]) + h
    else:
        if flag:
            distance = abs(position1[0]-position2[0]) + abs(position1[1]-position2[1]) + w
        else:
            distance = abs(position1[0]-position2[0]) + abs(position1[1]-position2[1])
    return round(distance, 1)


def decode(rs, cd, trolley_available_time, trolley_carrying_capacity, trolley_coordinate, supply_depot_occupy, hunger_data):
    def loading(trolley_label):
        to_depot_distance = []
        for depot_label in range(1, supply_depot_num+1):
            distance = manhattan_distance(trolley_coord[trolley_label], coordinate_data[f'depot{depot_label}'])
            to_depot_distance.append([depot_label, distance])
        for _, label_distance in enumerate(sorted(to_depot_distance, key=lambda x: (x[-1], x[-2]))):
            label, distance = label_distance
            if trolley_avail_time[trolley_label] + distance / SP >= min(depot_occupy[label]):
                sv = trolley_avail_time[trolley_label]
                el = round(sv + distance / SP + (UW - trolley_carry_capacity[trolley_label]) / SL, 2)
                position_coordinate = coordinate_data[f'depot{label}']
                depot_occupy[label][depot_occupy[label].index(min(depot_occupy[label]))] = el
                single_scheduling = [trolley_label, f'depot{label}', sv, el, UW]
                break
        trolley_avail_time[trolley_label] = el
        trolley_carry_capacity[trolley_label] = UW
        trolley_coord[trolley_label] = position_coordinate
        scheduling_list.append(single_scheduling)
        return trolley_avail_time, trolley_carry_capacity, trolley_coord, scheduling_list

    trolley_avail_time = copy.deepcopy(trolley_available_time)
    trolley_carry_capacity = copy.deepcopy(trolley_carrying_capacity)
    trolley_coord = copy.deepcopy(trolley_coordinate)
    depot_occupy = copy.deepcopy(supply_depot_occupy)

    scheduling_list = []
    for step in range(len(rs)):
        rearing_pond_label, trolley_label = rs[step], cd[step]
        feeding_weight = coefficient_data[rearing_pond_label] * (2+2*(hunger_data[rearing_pond_label]-hunger_threshold+1)/50)

        if trolley_carry_capacity[cd[step]] < feeding_weight:
            trolley_avail_time, trolley_carry_capacity, trolley_coord, scheduling_list = loading(trolley_label)

        destination = 'pond'+str(rs[step])
        to_pond_distance = manhattan_distance(trolley_coord[trolley_label], coordinate_data[destination])
        sm = trolley_avail_time[trolley_label]
        ef = round(sm + to_pond_distance/SP + feeding_weight/SF, 2)
        load = round(trolley_carry_capacity[trolley_label] - feeding_weight, 3)
        trolley_avail_time[trolley_label] = ef
        trolley_carry_capacity[trolley_label] = load
        trolley_coord[trolley_label] = coordinate_data[destination]
        single_scheduling = [trolley_label, destination, sm, ef, load]
        scheduling_list.append(single_scheduling)

        if trolley_carry_capacity[cd[step]] < LW:
            trolley_avail_time, trolley_carry_capacity, trolley_coord, scheduling_list = loading(trolley_label)
    return scheduling_list


def evaluation(scheduling_list, hunger_data, scheduling_start_time):
    dyn_urg = dynamic_urgency(hunger_data)
    fitness1, fitness2 = 0, 0
    trolley_load = [0 for _ in range(trolley_num)]
    for scheduling in scheduling_list:
        trolley_label = scheduling[0]
        if 'pond' in scheduling[1]:
            rearing_pond_label = int(scheduling[1][4:])
            fitness1 = fitness1 + dyn_urg[rearing_pond_label-1]*(scheduling[3]-scheduling_start_time)
        trolley_load[trolley_label - 1] = trolley_load[trolley_label - 1] + (scheduling[3] - scheduling[2])
    fitness2 = max(trolley_load)
    return round(fitness1, 2), round(fitness2, 2)


def nondominated_sort(comb_fitness_list1, comb_fitness_list2):
    fitness_list1, fitness_list2 = comb_fitness_list1.copy(), comb_fitness_list2.copy()
    all_nondominated_idx = []
    complete_flag = [float("inf") for _ in range(len(fitness_list1))]
    while fitness_list1 != complete_flag:
        nondominated_idx_list = []
        for idx_now in range(len(fitness_list1)):
            for idx_comp in range(len(fitness_list1)):
                if fitness_list1[idx_now] < fitness_list1[idx_comp] or fitness_list2[idx_now] < fitness_list2[idx_comp]\
                        or (fitness_list1[idx_now] == fitness_list1[idx_comp] and fitness_list2[idx_now] == fitness_list2[idx_comp]):
                    if idx_comp != len(fitness_list1) - 1:
                        continue
                    else:
                        nondominated_idx_list.append(idx_now)
                else:
                    break
        all_nondominated_idx.append(nondominated_idx_list)
        for nondominated_idx in nondominated_idx_list:
            fitness_list1[nondominated_idx], fitness_list2[nondominated_idx] = float("inf"), float("inf")
    return all_nondominated_idx
