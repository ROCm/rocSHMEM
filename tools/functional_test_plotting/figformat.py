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
