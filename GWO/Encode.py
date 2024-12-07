from Data import available_trolley_data, hunger_threshold

import random


class Encode:
    def __init__(self, hunger_data):
        self.hunger_data = hunger_data
        self.rearing_pond_needed = [_ for _ in range(1, len(self.hunger_data) + 1) if self.hunger_data[_] >= hunger_threshold]
        self.rearing_pond_needed_hunger = [self.hunger_data[_] for _ in self.rearing_pond_needed]

    def encode_rs(self):
        rs_list = self.rearing_pond_needed.copy()
        random.shuffle(rs_list)
        return rs_list

    def encode_cd(self, rs_list):
        cd_list = []
        for rearing_pond_label in rs_list:
            cd_list.append(random.choice(available_trolley_data[rearing_pond_label]))
        return cd_list


def dynamic_urgency(hunger_data):
    total_dyn_urg = 0
    rearing_pond_num = len(hunger_data)
    for rearing_pond_label in range(1, rearing_pond_num+1):
        total_dyn_urg = total_dyn_urg + max(0, hunger_data[rearing_pond_label] / hunger_threshold - 1)
    dyn_urg = [(max(0, hunger_data[i] / hunger_threshold - 1) / total_dyn_urg) for i in range(1, rearing_pond_num + 1)]
    return dyn_urg


def init_pop(pop_size, hunger_data):
    RS, CD = [], []
    encode = Encode(hunger_data)
    for _ in range(pop_size):
        rs_list = encode.encode_rs()
        cd_list = encode.encode_cd(rs_list)
        RS.append(rs_list)
        CD.append(cd_list)
    del encode
    return RS, CD
