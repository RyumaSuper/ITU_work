# -*- coding: utf-8 -*-
"""
matplotlibでリアルタイムプロットする例

無限にsin関数をplotし続ける
"""
from __future__ import unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt
import time
from drawnow import drawnow, figure

def pause_plot():
    fig, ax = plt.subplots()
    a = np.array([])
    b = np.array([])
    c = np.array([])
    c1 = "b"
    l1 = "sample"
    lines, = ax.plot(a, b, color=c1, label=l1)
    liness, = ax.plot(a, c, color=c1, label=l1)
    for i in range(0,1000):
        # fig, ax = plt.subplots()
        a = np.append(a, i)
        b = np.append(b, i*i)
        c = np.append(c, i*i/2)
        ax.set_xlabel('time')
        ax.set_ylabel('noise pixels')
        ax.set_title('noise pixels graph')

        lines.set_data(a, b)
        liness.set_data(a, c)
        ax.set_xlim((a.min(), a.max()))
        ax.set_ylim((b.min(), b.max()))
    
        ax.legend(loc=0)
        fig.tight_layout()
        plt.pause(0.1)
        fig.savefig("d.png")

if __name__ == "__main__":
    pause_plot()
