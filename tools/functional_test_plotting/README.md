# rocSHMEM Functional Test Plotting Scripts

This directory contains scripts to plot perftest data

> [!WARNING]
> These scripts have only been tested for the put and get APIs of rocSHMEM

# Project Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib pandas
```

# Generating Figures
Figures can be generated with the following

```
python3 rocshmem_plot.py <TestName> <Test1_Label> <Test1_Log_dir> ...  <TestN_Label> <TestN_Log_dir>
```

We require:
- `TestName` as it is written in the functional test.
- `Test1_Label` For the label for test 1 that will be placed in the legend
- `Test1_Log_dir` The director of the tests labeled test 1

We can accept `N` different directories to plot.

An example command could be:

```
python rocshmem_plot.py waveput_n2_w1_z64_1048576B Test_A ./logsA Test_B ./logsB
```

For each test it should output a different file for latency, bandwidth and message rate.

# Example

![Example](./example.png)
