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

import figformat as fmt # Needed for formating
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import re

# Used
import sys

test_name = None
log_dirs = []

def parse_args():
    global test_name
    global log_dirs
    global labels

    if len(sys.argv) % 2 != 0:
        print("Usage: python ploter.py <test_name> <lable1> <log_dir1> ...  <labelN> <log_dirN>")
        sys.exit(1)

    test_name = sys.argv[1]

    for i in range(2, len(sys.argv), 2):
        log_dir = (sys.argv[i], sys.argv[i+1])
        log_dirs.append(log_dir)

def read_test_data():
    global test_name
    global log_dirs

    dfs = []

    for i, (label, log_dir) in enumerate(log_dirs, 1):
        file_name = log_dir + "/" + test_name + ".log"

        header_line = "# Size (B)"

        with open(file_name, 'r') as file:
            lines = file.readlines()

        data = []
        recording = False

        for line in lines:
            line = line.strip()

            if header_line in line:
                recording = True
                continue
            if recording:
                csv_line = re.sub(r'\s+', ',', line.strip())
                data.append(csv_line)  # Add the line to the data list

        data = [row.split(",") for row in data]

        # Construct DataFrame
        df = pd.DataFrame(data)
        df.columns = ['Size', 'TimedMsgs', 'Latency', 'Bandwidth', 'MsgRate']
        df['Size'] = df['Size'].astype(int)
        df['Latency'] = df['Latency'].astype(float)
        df['Bandwidth'] = df['Bandwidth'].astype(float)
        df['MsgRate'] = df['MsgRate'].astype(float)

        dfs += [df]

    return dfs

def plot_bandwidth(dfs):
    global test_name

    fmt.init()

    plt.rcParams.update({
        "figure.figsize" : (12,8)
    })

    fig, ax = plt.subplots()

    for i, (label, log_dir) in enumerate(log_dirs, 0):
        df = dfs[i]
        x = np.arange(len(df['Size']))
        y = df['Bandwidth']
        ax.plot(x, y, label=str(label))

    # Bandwidth limit
    max_bw = 50
    plt.plot(x, ([max_bw] * len(x)), color='red', linestyle=':', linewidth=2, marker='')

    ax.set_ylabel('Bandwidth (GiB/s)')
    ax.set_xlabel('Message Size')

    # Convert Size in Bytes to formatted amount
    x_labels = [fmt.convert_size(s) for s in df['Size']]
    x_labels[1::2] = ['' for x in x_labels[1::2]]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.5)

    ax.legend()

    plt.savefig(test_name + "_bandwidth.png", dpi=1000, bbox_inches='tight')
    plt.clf()

def plot_latency(df):
    global test_name
    fmt.init()

    plt.rcParams.update({
        "figure.figsize" : (12,8)
    })

    fig, ax = plt.subplots()

    for i, (label, log_dir) in enumerate(log_dirs, 0):
        df = dfs[i]
        x = np.arange(len(df['Size']))
        y = df['Latency']
        ax.plot(x, y, label=str(label))

    ax.set_ylabel('Latency (us)')
    ax.set_xlabel('Message Size')

    # Convert Size in Bytes to formatted amount
    x_labels = [fmt.convert_size(s) for s in df['Size']]
    x_labels[1::2] = ['' for x in x_labels[1::2]]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.5)

    ax.legend()

    plt.savefig(test_name + "_latency.png", dpi=1000, bbox_inches='tight')
    plt.clf()

def plot_msgrate(df):
    global test_name
    fmt.init()

    plt.rcParams.update({
        "figure.figsize" : (12,8)
    })

    fig, ax = plt.subplots()

    for i, (label, log_dir) in enumerate(log_dirs, 0):
        df = dfs[i]
        x = np.arange(len(df['Size']))
        y = df['MsgRate']
        ax.plot(x, y, label=str(label))

    ax.set_ylabel('Message Rate (Msg/s)')
    ax.set_xlabel('Message Size')

    # Convert Size in Bytes to formatted amount
    x_labels = [fmt.convert_size(s) for s in df['Size']]
    x_labels[1::2] = ['' for x in x_labels[1::2]]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)

    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.5)

    ax.legend()

    plt.savefig(test_name + "_msgrate.png", dpi=1000, bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    parse_args()

    dfs = read_test_data()

    plot_bandwidth(dfs)
    plot_latency(dfs)
    plot_msgrate(dfs)
