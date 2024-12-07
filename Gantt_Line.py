import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def gantt(figure_num, schedule_stime, plt_item, trolley_num=6):
    plt.figure(figure_num)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 35

    for single_scheduling in plt_item:
        if single_scheduling[3] < schedule_stime:
            color = "yellowgreen"
        elif single_scheduling[2] < schedule_stime < single_scheduling[3]:
            color = 'gold'
        else:
            color = 'indianred'
        car_id = single_scheduling[0]
        destination = single_scheduling[1]
        start_time = single_scheduling[2]
        end_time = single_scheduling[3]

        if destination[:4] == 'pond':
            text = 'RP' + destination[4:]
        else:
            text = 'SD' + destination[5:]
        plt.text(x=start_time + ((end_time - start_time) / 2 - 11), y=car_id - 0.1, s=text, size=35)
        plt.barh(y=car_id, width=end_time - start_time, height=0.8, left=start_time, color=color, edgecolor='black')
    plt.yticks(np.arange(1, trolley_num + 1), np.arange(1, trolley_num + 1))
    plt.ylabel("AFV")
    plt.xlabel("Time")
    plt.tick_params(labelsize=35)
    plt.tick_params(direction='in')
    ax = plt.gca()
    width = 1
    ax.spines['top'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)


def line_chart(label, fitness_iter_list, iteration_num):
    iteration_num_list = [_ for _ in range(iteration_num+1)]
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.plot(iteration_num_list, fitness_iter_list, label=label, linewidth=3, zorder=1)
    plt.xlabel("Iteration number", size=35)
    plt.ylabel("Fitness value", size=35)
    plt.tick_params(labelsize=35)
    plt.tick_params(direction='in')
    ax = plt.gca()
    width = 1
    ax.spines['top'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)
