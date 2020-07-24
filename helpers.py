from shapely.geometry import Polygon
import os
import operator
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cv2
import argparse


def plot(data, label,title):
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(data, facecolor='red', edgecolor='black')
    ax.set_xticks(bins)

    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        ax.annotate(str(int(count)), xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # Label the percentages
        percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -32), textcoords='offset points', va='top', ha='center')
    plt.subplots_adjust(bottom=0.20)
    plt.xlabel('Pixels', labelpad=30)
    plt.ylabel('# of annotations')
    plt.title(f'{label} {title}')
    plt.show()