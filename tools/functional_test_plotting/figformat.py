###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

import math
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0"
    size_name = ("\nB", "\nKiB", "\nMiB", "\nGiB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s%s" % (int(s), size_name[i])

def init():
    plt.style.use('dark_background')

    hex_colours = ["#ED1C23",
                   "#00C2DE",
                   "#D9D9D9",
                   "#F26422",
                   "#C1A968",
                   "#FFFFFF"]
    n_hex_colours = len(hex_colours)

    # Four Sets of this list as we have four main linestyles
    hex_colours = hex_colours \
                + hex_colours \
                + hex_colours \
                + hex_colours

    linestyle = ['-']  * n_hex_colours \
              + ['--'] * n_hex_colours \
              + [':']  * n_hex_colours \
              + ['-.'] * n_hex_colours

    marker = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', \
              'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

    # Make the marker list match the length of the other lists
    marker = marker \
           + marker \
           + marker \
           + marker

    marker = marker[:(4 * n_hex_colours)]

    # Set up the cycler
    default_cycler = (cycler(color=hex_colours) +
                      cycler(linestyle=linestyle) +
                      cycler(marker=marker))

    # Update Params
    plt.rcParams.update({
        "font.size"       : 14,
        "axes.prop_cycle" : default_cycler,
        "font.sans-serif" : "Arial",
        "font.family"     : "sans-serif"
    })

if __name__ == "__main__":
    print("Testing figformat.py runs")
    init()
