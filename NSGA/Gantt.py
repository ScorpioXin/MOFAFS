"""
"aliceblue", "antiquewhite", "palevioletred", "aquamarine", "azure", "beige", "bisque", "blanchedalmond",
"blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral",
"cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray",
"darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred",
"darksalmon", "darkseagreen", "darkslateblue", "darkslategray", "darkturquoise", "darkviolet", "deeppink",
"deepskyblue", "dimgray", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro",
"ghostwhite", "gold", "goldenrod", "gray", "green", "greenyellow", "honeydew", "hotpink", "indianred",
"indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue",
"lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgreen", "lightgray", "lightpink", "lightsalmon",
"lightseagreen", "lightskyblue", "lightslategray", "lightsteelblue", "lightyellow", "lime", "limegreen",
"linen", "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
"mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred",
"midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive",
"olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "aqua",
"papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "rebeccapurple", "red",
"rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver",
"skyblue", "slateblue", "slategray", "snow", "springgreen", "steelblue", "tan", "teal", "thistle", "tomato",
"turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen
"""


from Data import trolley_num

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def gantt(figure_num, schedule_stime, plt_item):
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
            text = 'SD' + destination[4:]
        # plt.barh(y=car_id, width=end_time - start_time, height=0.8, left=start_time, color=color_set[color_sel], edgecolor='black')
        plt.text(x=start_time + ((end_time - start_time) / 2 - 11), y=car_id - 0.1, s=text, size=30)
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
    # plt.show()


def line_chart(label, fitness_iter_list, iteration_num):
    iteration_num_list = [_ for _ in range(1, iteration_num+1)]
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # plt.rcParams['font.size'] = 35
    plt.plot(iteration_num_list, fitness_iter_list, label=label, linewidth=3, zorder=1)
    # x = [i for i in range(iteration_num)]
    # y = [j for j in fitness_iter_list]
    # plt.scatter(x, y, color='yellow', marker='p', s=50, zorder=2)
    plt.xlabel("Iteration number", size=35)
    plt.ylabel("Fitness value", size=35)
    plt.legend(loc='upper right', fontsize=10)
    plt.tick_params(labelsize=25)
    plt.tick_params(direction='in')
    # plt.xticks(np.arange(0, iteration_num+10, 10))
    # plt.yticks(np.arange(100, 500, 10))
    ax = plt.gca()
    width = 1
    ax.spines['top'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)
    # plt.show()


