import json
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LABEL_SIZE = 18
LEGEND_SIZE = 14
TITLE_SIZE = 26
TICKS_SIZE = 20
OTHER_SIZES = 20
ALL_SIZE = 14

REWARDS_ACTIONS = "rewards_actions"

METRICS_OF_INTEREST = [
    {"code": "loss_trend", "description": "Loss"},
    {"code": "reward_trend", "description": "Reward"},
    # {"code": "learning_rate_trend", "description": "Learning rate"},
]

METRICS_OF_INTEREST_LIST = [
    {"code": "packet_delivery_ratio", "description": "Packet delivery ratio"},
    {"code": "packet_mean_delivery_time", "description": "Packet mean delivery time"},
    {"code": "mean_number_of_relays", "description": "Mean number of relays"},
]

dir_ql_data = {
    25000:
        {5: {'DIR_QL': {'packet_delivery_ratio': 0.4868398268398268, 'packet_mean_delivery_time': 98.26269106388924,
                        'mean_number_of_relays': 1.047388067472493, 'packet_delivery_times_std': 70.0975605905746,
                        'std_dev_numbers_of_possible_relays': 0.03176572896891097}}, 10: {
            'DIR_QL': {'packet_delivery_ratio': 0.5587878787878787, 'packet_mean_delivery_time': 97.26862487428401,
                       'mean_number_of_relays': 1.25035317029461, 'packet_delivery_times_std': 67.74647390120798,
                       'std_dev_numbers_of_possible_relays': 0.07423009694784577}}, 15: {
            'DIR_QL': {'packet_delivery_ratio': 0.6743722943722943, 'packet_mean_delivery_time': 101.66328410844564,
                       'mean_number_of_relays': 1.4674652433989317, 'packet_delivery_times_std': 65.42960015087664,
                       'std_dev_numbers_of_possible_relays': 0.10540298802603155}}, 20: {
            'DIR_QL': {'packet_delivery_ratio': 0.6996536796536799, 'packet_mean_delivery_time': 100.91552484741257,
                       'mean_number_of_relays': 1.603266265634031, 'packet_delivery_times_std': 64.58365965066474,
                       'std_dev_numbers_of_possible_relays': 0.12398794939374636}}, 25: {
            'DIR_QL': {'packet_delivery_ratio': 0.8005194805194804, 'packet_mean_delivery_time': 97.12477801675585,
                       'mean_number_of_relays': 1.7983360669920452, 'packet_delivery_times_std': 63.26328965037957,
                       'std_dev_numbers_of_possible_relays': 0.14252821800425472}}, 40: {
            'DIR_QL': {'packet_delivery_ratio': 0.8990476190476188, 'packet_mean_delivery_time': 91.47366413838027,
                       'mean_number_of_relays': 2.5445279928008224, 'packet_delivery_times_std': 60.23264882562124,
                       'std_dev_numbers_of_possible_relays': 0.21650327861232074}}, 50: {
            'DIR_QL': {'packet_delivery_ratio': 0.9289177489177489, 'packet_mean_delivery_time': 88.87439493923141,
                       'mean_number_of_relays': 3.132715109044326, 'packet_delivery_times_std': 57.767651545298214,
                       'std_dev_numbers_of_possible_relays': 0.2682499122323546}}}
}

MAP_ALGO_TO_LABEL = {
    "RND": "Rand",
    "GEO": "Geo",
    "DIR_QL": "Dir QL",
    "ARDEEP_QL": "ARdeep QL"
}

SCALE_LIM_DICT = {
    "number_of_packets_to_depot": {
        "scale": "linear",
        "ylim": (0, 1000)
    },
    "packet_mean_delivery_time": {
        "scale": "linear",
        "ylim": (0, 5)
    },
    "mean_number_of_relays": {
        "scale": "linear",
        "ylim": (0, 10)
    }
}

PLOT_DICT = {
    "RND": {
        "hatch": "",
        "markers": "",
        "linestyle": "-",
        "color": "red",
        "label": "Rand",
        "x_ticks_positions": np.array(np.linspace(0, 10_000, 20))
    },
    "GEO": {
        "hatch": "",
        "markers": "",
        "linestyle": "-",
        "color": "green",
        "label": "Geo",
        "x_ticks_positions": np.array(np.linspace(0, 10_000, 20))
    },
    "DIR_QL": {
        "hatch": "",
        "markers": "",
        "linestyle": "-",
        "color": "blue",
        "label": "Dir QL",
        "x_ticks_positions": np.array(np.linspace(0, 10_000, 20))
    },
    "ARDEEP_QL": {
        "hatch": "",
        "markers": "",
        "linestyle": "-",
        "color": "violet",
        "label": "ARdeep QL",
        "x_ticks_positions": np.array(np.linspace(0, 10_000, 20))
    },
}

with open('./dqn_data.json') as f:
    data = json.load(f)

packet_delivery_time = [
    7,
    561,
    75,
    720,
    592,
    938,
    982,
    327,
    323,
    608,
    137,
    557,
    140,
    0,
    944,
    1250,
    2,
    41,
    814,
    1388,
    296,
    1263,
    855,
    1399,
    595,
    122,
    871,
    166,
    769,
    1322,
    0,
    1307,
    527,
    812,
    522,
    410,
    302,
    57,
    340,
    467,
    400,
    509,
    473,
    57,
    917,
    738,
    756,
    829,
    1237,
    1477,
    462,
    288,
    7,
    657,
    795,
    1255,
    332,
    703,
    476,
    1150,
    1167,
    361,
    1458,
    647,
    1198,
    132,
    985,
    53,
    63,
    744,
    227,
    777,
    676,
    337,
    0,
    0,
    657,
    1414,
    695,
    0,
    1432,
    622,
    1452,
    314,
    444,
    1137,
    1142,
    1232,
    703,
    837,
    0,
    187,
    782,
    1050,
    17,
    694,
    227,
    708,
    587,
    402,
    1391,
    0,
    32,
    483,
    0,
    678,
    188,
    0,
    0,
    710,
    0,
    1261,
    1467,
    0,
    1465,
    439,
    275,
    280,
    932,
    47,
    69,
    0,
    1141,
    267,
    36,
    1066,
    465,
    1159,
    879,
    29,
    1003,
    1269,
    0,
    868,
    1380,
    187,
    443,
    1362,
    1029,
    0,
    1388,
    137,
    0,
    1037,
    210,
    0,
    852,
    873,
    1426,
    0,
    405,
    2,
    1076,
    0,
    1047,
    202,
    1139,
    75,
    0,
    912,
    0,
    0,
    1437,
    704,
    667,
    1024,
    0,
    478,
    1464,
    452,
    852,
    175,
    684,
    1167,
    392,
    301,
    1019,
    0,
    1011,
    825,
    44,
    942,
    865,
    660,
    657,
    402,
    360,
    1337,
    751,
    977,
    132,
    920,
    533,
    611,
    0,
    208,
    478,
    0,
    489,
    1028,
    316,
    1440,
    1158,
    232,
    1271,
    388,
    749,
    743,
    122,
    702,
    348,
    1337,
    0,
    22,
    576,
    127,
    1334,
    997,
    738,
    527,
    297,
    477,
    760,
    47,
    951,
    782,
    611,
    946,
    829,
    912,
    552,
    1131,
    267,
    589,
    18,
    237,
    1297,
    0,
    699,
    821,
    0,
    0,
    387,
    1227,
    1107,
    1276,
    1029,
    639,
    7,
    0,
    0,
    162,
    0,
    248,
    627,
    790,
    142,
    985,
    28,
    1009,
    1117,
    808,
    817,
    552,
    0,
    1042,
    347,
    832,
    822,
    37,
    792,
    12,
    281,
    1313,
    478,
    12,
    1074,
    163,
    377,
    673,
    378,
    1378,
    416,
    427,
]


def read_data_from_files():
    mypath = "C:/Users/paolo.bevilacqua/Desktop/uni/autonomous networking/homework1/" \
             "DroNETworkSimulator/data/evaluation_tests/dql/collect_final"

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    datas = {}

    for file_name in onlyfiles:
        with open(mypath + "/" + file_name, 'r') as in_file:
            data = json.load(in_file)

            curr_datas = {}

            setup = data["mission_setup"]

            algo = setup["routing_algorithm"].split(".")[1]
            len_simulation = setup["len_simulation"]
            n_drones = setup["n_drones"]

            packet_delivery_ratio = data["packet_delivery_ratio"]
            packet_mean_delivery_time = data["packet_mean_delivery_time"]
            mean_number_of_relays = data["mean_number_of_relays"]

            # packet_delivery_times_std = data["packet_delivery_times_std"] * 0.15
            # std_dev_numbers_of_possible_relays = data["std_dev_numbers_of_possible_relays"] * 0.15

            if len_simulation not in datas:
                datas[len_simulation] = {}

            if n_drones not in datas[len_simulation]:
                datas[len_simulation][n_drones] = {}

            if algo not in datas[len_simulation][n_drones]:
                datas[len_simulation][n_drones][algo] = {}

            datas[len_simulation][n_drones][algo]["packet_delivery_ratio"] = packet_delivery_ratio
            datas[len_simulation][n_drones][algo]["packet_mean_delivery_time"] = packet_mean_delivery_time
            datas[len_simulation][n_drones][algo]["mean_number_of_relays"] = mean_number_of_relays

            # datas[len_simulation][n_drones][algo]["packet_delivery_times_std"] = packet_delivery_times_std
            # datas[len_simulation][n_drones][algo]["std_dev_numbers_of_possible_relays"] = std_dev_numbers_of_possible_relays

    return datas


def plot_scatter(data, type):
    data.sort()
    plt.plot(data, '.')
    plt.savefig("./figures/" + type + ".png", dpi=400)
    plt.clf()


def plot(algorithm: str,
         x_data: list,
         y_data: list,
         y_data_std: list,
         type):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(15, 6.5))

    ax1.errorbar(x=x_data,
                 y=y_data,
                 yerr=None,
                 label=type["description"],
                 # marker=PLOT_DICT[algorithm]["markers"],
                 linestyle=PLOT_DICT[algorithm]["linestyle"],
                 color=PLOT_DICT[algorithm]["color"],
                 # fmt="o",
                 # elinewidth=1,
                 # markersize=8
                 )

    mean = np.mean(y_data)
    ax1.errorbar(x=x_data, y=mean, label="Mean")

    ax1.set_ylabel(ylabel=type["description"], fontsize=LABEL_SIZE)
    ax1.set_xlabel(xlabel="Timestamp", fontsize=LABEL_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=ALL_SIZE)

    # plt.xticks(ticks=np.linspace(0, max(y_data), 100))

    plt.legend(ncol=1,
               handletextpad=0.5,
               columnspacing=0.5,
               prop={'size': LEGEND_SIZE})

    plt.grid(linewidth=0.3)
    plt.tight_layout()
    # plt.savefig("src/plots/figures/" + type + ".svg")
    plt.savefig("./figures/final_" + type["code"] + ".png", dpi=400)
    plt.clf()


def plot_list(algorithms: list,
              x_data: list,
              y_data: dict,
              y_data_std: dict,
              type: str):
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))

    for alg in algorithms:
        ax1.errorbar(x=x_data,
                     y=y_data[alg],
                     yerr=y_data_std[alg],
                     label=MAP_ALGO_TO_LABEL[alg],
                     color="purple",
                     capsize=7,
                     ecolor=PLOT_DICT[alg]["color"],
                     elinewidth=2,
                     marker=".",
                     markersize=10,
                     fmt="o"
                     )

        # y1 = [y_data[alg][x] + y_data_std[alg] for x in range(len(x_data))]
        # y2 = [y_data[alg][x] - y_data_std[alg] for x in range(len(x_data))]

        # ax1.fill_between(x_data, y1, y2, alpha=.2, color=PLOT_DICT[alg]["color"])

    # ax1.plot(x_data, [np.mean(y_data[alg])] * 5)

    ax1.set_ylabel(ylabel=type["description"], fontsize=LABEL_SIZE)
    ax1.set_xlabel(xlabel="Number of drones", fontsize=LABEL_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=ALL_SIZE)

    plt.legend(ncol=1,
               handletextpad=0.5,
               columnspacing=0.5,
               prop={'size': LEGEND_SIZE})

    plt.grid(linewidth=0.3)
    plt.tight_layout()
    # plt.savefig("src/plots/figures/" + type + ".svg")
    # plt.savefig("./figures/" + type["code"] + "_list.png", dpi=400)

    plt.savefig("./figures/dir_ql_" + type["code"] + ".png", dpi=400)
    plt.clf()


def catplot(rewards_to_plot):
    values = []

    max_len = max([len(i) for i in rewards_to_plot])

    for i in range(len(rewards_to_plot)):
        values.append(rewards_to_plot[i] + [None] * (max_len - len(rewards_to_plot[i])))

    transformed_values = []

    for v in values:
        appo = []
        for i in range(0):
            appo.append(v[i])
        transformed_values.append(appo)

    statino = {}

    actions = [str(a) for a in range(len(rewards_to_plot))]

    i = 0
    for a in actions:
        statino[a] = values[i]
        i += 1

    rew = pd.DataFrame(statino)

    tkt_plot_detailed = sns.catplot(data=rew, color="blue", s=1.5, kind="swarm", aspect=2)

    tkt_plot_detailed.set_axis_labels('Actions', 'Observed values')

    plt.title("Rewards for actions")
    plt.subplots_adjust(bottom=0.12, left=.1, top=0.93)

    plt.savefig("./figures/" + REWARDS_ACTIONS + "_catplot.png", dpi=400)
    plt.clf()


def plot_violin(rewards_to_plot):
    # create test data
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharey=True)

    ax1.set_title('Rewards distribution per actions')
    ax1.set_ylabel('Reward')

    ax1.violinplot(rewards_to_plot, showextrema=True, widths=0.7)

    # set style for the axes
    actions = [str(a) for a in range(len(rewards_to_plot))]
    labels = actions

    ax1.xaxis.set_tick_params(direction='out')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks(np.arange(1, len(labels) + 1))
    ax1.set_xticklabels(labels)
    ax1.set_xlim(0.25, len(labels) + 0.75)
    ax1.set_xlabel('Actions')

    plt.subplots_adjust(bottom=0.15, wspace=0.05)

    plt.grid(linewidth=0.3)
    # plt.tight_layout()

    plt.savefig("./figures/" + REWARDS_ACTIONS + "_violin.png", dpi=400)
    plt.clf()


PLOT_SIMPLE = False
PLOT_LIST = True

if __name__ == "__main__":

    if PLOT_SIMPLE:
        for metric in METRICS_OF_INTEREST:
            values = data[metric["code"]]

            xvalue = []
            yvalue = []

            for i in values:  # x-axies
                if i[0] % 1 == 0:
                    xvalue.append(i[0])
                    yvalue.append(i[1])

            y_data_std = np.std(yvalue)

            plot(algorithm="ARDEEP_QL", x_data=xvalue, y_data_std=y_data_std, y_data=yvalue, type=metric)

    if PLOT_LIST:
        algorithms = ["DIR_QL"]  # , "RND", "GEO", "DIR_QL"]

        for metric in METRICS_OF_INTEREST_LIST:
            # datas = read_data_from_files()
            # datas = datas[50000]
            datas = dir_ql_data[25000]

            xvalue = []
            y_data_std = {}

            yvalue = {}
            yvalue_list = []

            for algo in algorithms:
                yvalue[algo] = []
                y_data_std[algo] = []

            for n_drone in datas.keys():  # x-axies

                collected_data = datas[n_drone]

                xvalue.append(str(n_drone))

                for algo in algorithms:
                    yvalue[algo].append(collected_data[algo][metric["code"]])

            for algo in algorithms:
                y_data_std[algo] = np.std(yvalue[algo])

            # std_dev_packet_delivery_time = {"ARDEEP_QL": [i * 0.15 for i in [
            #     450.0220723681188,
            #     477.1409833331955,
            #     472.6339160081898,
            #     481.15114468681185,
            #     476.4882852238401,
            # ]]}

            # std_dev_number_of_relays = {
            #     "ARDEEP_QL": [0.4428223297829301, 0.6288469012972688, 0.7593337370234032, 0.900685595321348,
            #                   0.936021233637639]
            # }

            std_dev_dir_ql = {
                5: 0.02133123197611567,
                10: 0.02170796044220235,
                15: 0.018651730043117168,
                20: 0.017365513675023522,
                25: 0.01967178828962232,
                40: 0.01681616205202393,
                50: 0.013150201280095091
            }

            y_data_std = None
            if metric["code"] is "packet_delivery_ratio":
                y_data_std = std_dev_dir_ql[n_drone]

            elif metric["code"] is "packet_mean_delivery_time":
                if not y_data_std:
                    y_data_std = []

                for n_drone in datas.keys():  # x-axies
                    y_data_std.append(datas[n_drone][algo]["packet_delivery_times_std"])

            elif metric["code"] is "mean_number_of_relays":
                if not y_data_std:
                    y_data_std = []
                for n_drone in datas.keys():  # x-axies
                    y_data_std.append(datas[n_drone][algo]["std_dev_numbers_of_possible_relays"])

            plot_list(algorithms=["DIR_QL"], x_data=xvalue, y_data_std={"DIR_QL": y_data_std}, y_data=yvalue,
                      # algorithms=algorithms
                      type=metric)

    rewards_actions = data[REWARDS_ACTIONS]

    rewards_to_plot = []

    for drone in rewards_actions.keys():
        rewards_to_plot.append([])

    for drone in rewards_actions.keys():

        reward_values = rewards_actions[drone].keys()

        for action in reward_values:
            # filter only positive values
            # rew = np.array(rewards_actions[drone][action])
            # filt_rew = rew > 0
            # rewards_to_plot[int(action)] += (list(rew[filt_rew]))

            rewards_to_plot[int(action)] += (rewards_actions[drone][action])

    # for drone in rewards_actions.keys():
    #     reward_values = rewards_actions[drone].keys()
    #     for action in reward_values:
    #         for i in rewards_to_plot[int(action)]:
    #             if i > 1:
    #                 print("sium")

    # catplot(rewards_to_plot)
    # plot_violin(rewards_to_plot)
