"""
Implement here your plotting functions
Below you can see a print function example.
You should use it as a reference to implements your own plotting function

IMPORTANT: if you need you can and should use other matplotlib functionalities! Use
            the following example only as a reference

The plot workflow is can be summarized as follows:

    1) Extensive simulations
    2) Json file containing results
    3) Compute averages and stds for each metric for each algorithm
    4) Plot the results

In order to maintain the code tidy you can use:

    - src.plots.config.py file to store all the parameters you need to
        get wonderful plots (see the file for an example)

    - src.plots.data.data_elaboration.py file to write the functions that compute averages and stds from json
        result files

    - src.plots.plot_data.py file to make the plots.

The script plot_data.py can be run using python -m src.plots.plot_data

"""

import json

# ***EXAMPLE*** #
import matplotlib.pyplot as plt
import numpy as np

from src.plots.config import PLOT_DICT, OTHER_SIZES, LABEL_SIZE, LEGEND_SIZE

with open('src/plots/data.json') as f:
    data = json.load(f)


def plot(algorithm: list,
         y_data: dict,
         y_data_std: dict or None,
         type: str):
    """
    This method has the ONLY responsibility to plot data
    @param y_data_std:
    @param y_data:
    @param algorithm:
    @param type:
    @return:
    """

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8.5, 6.5))

    """
    # ------------------- get line in plot for each algorithm ------------------- 
    for alg in algorithm:
        axs.errorbar(x=np.array(PLOT_DICT[alg]["x_ticks_positions"]),
                     y=y_data[alg],
                     label=PLOT_DICT[alg]["label"],
                     marker=PLOT_DICT[alg]["markers"],
                     linestyle=PLOT_DICT[alg]["linestyle"],
                     color=PLOT_DICT[alg]["color"],
                     markersize=5)
                     
    axs.set_ylabel(ylabel=MAP_METRIC_TO_TITLE[type], fontsize=LABEL_SIZE)
    axs.set_xlabel(xlabel="Number of Drones", fontsize=LABEL_SIZE)
    axs.tick_params(axis='both', which='major', labelsize=OTHER_SIZES)
    """

    # ------------------- get line in plot by seed -------------------

    axs.errorbar(x=np.array(PLOT_DICT["seed"]["x_ticks_positions"]),
                 y=y_data,
                 label=PLOT_DICT["seed"]["label"],
                 marker=PLOT_DICT["seed"]["markers"],
                 linestyle=PLOT_DICT["seed"]["linestyle"],
                 color=PLOT_DICT["seed"]["color"],
                 markersize=5)

    # seed
    axs.set_ylabel(ylabel="Packet delivery ratio", fontsize=LABEL_SIZE)
    axs.set_xlabel(xlabel="Seed", fontsize=LABEL_SIZE)
    axs.tick_params(axis='both', which='major', labelsize=OTHER_SIZES)

    plt.xticks(ticks=np.linspace(0, 30, 30))

    plt.legend(ncol=1,
               handletextpad=0.1,
               columnspacing=0.7,
               prop={'size': LEGEND_SIZE})

    plt.grid(linewidth=0.2)
    plt.tight_layout()
    plt.savefig("src/plots/figures/" + type + ".png", dpi=400)
    plt.clf()


if __name__ == "__main__":
    """
    Run this file to get the plots.
    Of course, since you need to plot more than a single data series (one for each algorithm) you need to modify
    plot() in a way that it can handle a multi-dimensional data (one data series for each algorithm). 
    y_data and y_data_std could be for example a list of lists o a dictionary containing lists. It up to you to decide
    how to deal with data
    """

    # packet_delivery_ratio

    """
    # metrics by algorithm
    for m in METRICS_OF_INTEREST:

        y_data = {}
        for alg in PLOTS_ALGORITMS:
            yvalue = []
            for ndrones in X_VALUES_N_DRONES:
                yvalue.append(data[alg][str(ndrones)][m])
            y_data[alg] = yvalue
        plot(algorithm=PLOTS_ALGORITMS, y_data_std=None, y_data=y_data, type=m) 
    """

    # seed
    yvalue = []
    for i in range(0, 31):
        yvalue.append(data[str(i)]["packet_delivery_ratio"])

    plot(algorithm=["DIR_QL"], y_data_std=None, y_data=yvalue, type="seed-packet_delivery_ratio")
