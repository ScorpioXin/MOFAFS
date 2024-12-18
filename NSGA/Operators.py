from Data import all_hunger_data, available_trolley_data
from Encode import init_pop

import random
import copy


class Operators:
    def __init__(self, pop_size, iteration_num, pf_max, pf_min, now_scheduling_cout):
        self.pop_size = pop_size
        self.iteration_num = iteration_num
        self.pf_max = pf_max
        self.pf_min = pf_min
        self.hunger_data = all_hunger_data[now_scheduling_cout-1]

    def pox_crossover(self, p1_rs, p1_cd, p2_rs, p2_cd):
        r1_num = random.randint(1, len(p1_rs)-1)
        r1_num_idx = random.sample(range(0, len(p1_rs)-1), r1_num)
        c_rs = [0 for _ in range(0, len(p1_rs))]
        c_cd = [0 for _ in range(0, len(p1_cd))]
        for _ in r1_num_idx:
            c_rs[_], c_cd[_] = p1_rs[_], p1_cd[_]
        for idx, value in enumerate(p2_rs):
            if value not in c_rs:
                zero_idx = c_rs.index(0)
                c_rs[zero_idx], c_cd[zero_idx] = value, p2_cd[idx]
        return c_rs, c_cd

    def rpx_crossover(self, p1_rs, p1_cd, p2_rs, p2_cd, now_iteration_num):
        pf = self.pf_max - (self.pf_max - self.pf_min) / self.iteration_num * now_iteration_num
        R = [random.random() for _ in range(len(p1_rs))]
        rpx_value = [p1_rs[idx] for idx, value in enumerate(R) if value < pf]
        c_rs = copy.copy(p1_rs)
        c_cd = copy.copy(p1_cd)
        for value in rpx_value:
            c_cd[p1_rs.index(value)] = p2_cd[p2_rs.index(value)]
        return c_rs, c_cd

    def insertion_mutation(self, p_rs, p_cd):
        two_random_points = random.sample(range(0, len(p_rs)), 2)
        two_random_points.sort()
        c_rs, c_cd = [], []
        c_rs.extend(p_rs[:two_random_points[0]])
        c_rs.extend(p_rs[two_random_points[1]:two_random_points[1]+1])
        c_rs.extend(p_rs[two_random_points[0]:two_random_points[1]])
        c_rs.extend(p_rs[two_random_points[1]+1:])
        c_cd.extend(p_cd[:two_random_points[0]])
        c_cd.extend(p_cd[two_random_points[1]:two_random_points[1]+1])
        c_cd.extend(p_cd[two_random_points[0]:two_random_points[1]])
        c_cd.extend(p_cd[two_random_points[1]+1:])
        return c_rs, c_cd

    def exchange_mutation(self, p_rs, p_cd):
        two_random_points = random.sample(range(0, len(p_rs)), 2)
        mutation_position1, mutation_position2 = two_random_points[0], two_random_points[1]
        rs_temp1, cd_temp1 = p_rs[mutation_position1], p_cd[mutation_position1]
        rs_temp2, cd_temp2 = p_rs[mutation_position2], p_cd[mutation_position2]
        c_rs = copy.copy(p_rs)
        c_cd = copy.copy(p_cd)
        c_rs[mutation_position1], c_cd[mutation_position1] = rs_temp2, cd_temp2
        c_rs[mutation_position2], c_cd[mutation_position2] = rs_temp1, cd_temp1
        return c_rs, c_cd

    def reverse_mutation(self, p_rs, p_cd):
        two_random_points = random.sample(range(0, len(p_rs)), 2)
        two_random_points.sort()
        mutation_position1, mutation_position2 = two_random_points[0], two_random_points[1]
        rs_reverse = p_rs[mutation_position1:mutation_position2+1][::-1]
        cd_reverse = p_cd[mutation_position1:mutation_position2+1][::-1]
        c_rs, c_cd = [], []
        c_rs.extend(p_rs[:mutation_position1])
        c_rs.extend(rs_reverse)
        c_rs.extend(p_rs[mutation_position2+1:])
        c_cd.extend(p_cd[:mutation_position1])
        c_cd.extend(cd_reverse)
        c_cd.extend(p_cd[mutation_position2+1:])
        return c_rs, c_cd

    def cd_alone_mutation(self, p_rs, p_cd):
        c_rs = copy.copy(p_rs)
        c_cd = copy.copy(p_cd)
        mutation_num = random.randint(1, len(p_rs))
        mutation_points = random.sample(range(0, len(p_rs)), mutation_num)
        for mutation_point in mutation_points:
            available_trolley = available_trolley_data[p_rs[mutation_point]]
            if type(available_trolley) is tuple:
                while True:
                    replace_trolley = random.choice(available_trolley)
                    if replace_trolley != c_cd[mutation_point]:
                        break
                c_cd[mutation_point] = replace_trolley
        return c_rs, c_cd

    def selection(self, comb_fitness_list1, comb_fitness_list2, all_nondominated_idx, comb_RS, comb_CD):
        global alpha, beta
        sigma = 0
        iter_pond_RS, iter_pond_CD = [], []
        existed_individual = []
        for nondominated_idx in all_nondominated_idx:
            if len(iter_pond_RS) == self.pop_size:
                break
            elif len(nondominated_idx) <= self.pop_size - len(iter_pond_RS):
                for idx in nondominated_idx:
                    individual = []
                    individual.append(comb_fitness_list1[idx])
                    individual.append(comb_fitness_list2[idx])
                    if individual in existed_individual and random.random() <= sigma:
                        continue
                    else:
                        iter_pond_RS.append(comb_RS[idx])
                        iter_pond_CD.append(comb_CD[idx])
                        existed_individual.append(individual)
            else:
                nondominated_fitness_list1 = [comb_fitness_list1[idx] for idx in nondominated_idx]
                nondominated_fitness_list2 = [comb_fitness_list2[idx] for idx in nondominated_idx]
                fmax1, fmin1 = max(nondominated_fitness_list1), min(nondominated_fitness_list1)
                fmax2, fmin2 = max(nondominated_fitness_list2), min(nondominated_fitness_list2)
                comb_normal_nondomin_fit_list = []
                if fmax1 != fmin1 and fmax2 != fmin2:
                    comb_normal_nondomin_fit_list = [alpha*(fitness1-fmin1)/(fmax1-fmin1)+beta*(fitness2-fmin2)/(fmax2-fmin2)
                                                     for fitness1, fitness2 in zip(nondominated_fitness_list1, nondominated_fitness_list2)]
                elif fmax1 == fmin1 and fmax2 != fmin2:
                    comb_normal_nondomin_fit_list = [beta*(fitness2 - fmin2)/(fmax2 - fmin2)
                                                     for fitness2 in nondominated_fitness_list2]
                elif fmax1 != fmin1 and fmax2 == fmin2:
                    comb_normal_nondomin_fit_list = [alpha*(fitness1-fmin1)/(fmax1-fmin1)
                                                     for fitness1 in nondominated_fitness_list1]
                else:
                    pass
                if len(comb_normal_nondomin_fit_list) != 0:
                    sorted_nondominated_idx = [x for x, y in sorted(zip(nondominated_idx, comb_normal_nondomin_fit_list), key=lambda y: y[1])]
                else:
                    sorted_nondominated_idx = copy.copy(nondominated_idx)
                for idx in sorted_nondominated_idx:
                    individual = []
                    individual.append(comb_fitness_list1[idx])
                    individual.append(comb_fitness_list2[idx])
                    if individual in existed_individual and random.random() <= sigma:
                        continue
                    else:
                        iter_pond_RS.append(comb_RS[idx])
                        iter_pond_CD.append(comb_CD[idx])
                        existed_individual.append(individual)
                    if len(iter_pond_RS) == self.pop_size:
                        break
        if len(iter_pond_RS) < self.pop_size:
            RS, CD = init_pop(self.pop_size-len(iter_pond_RS), self.hunger_data)
            iter_pond_RS.extend(RS)
            iter_pond_CD.extend(CD)
        return iter_pond_RS, iter_pond_CD


alpha, beta = 0.55, 0.45
