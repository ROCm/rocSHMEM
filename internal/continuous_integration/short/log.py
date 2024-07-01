#!/tool/pandora64/.package/python-3.8.0/bin/python3

import pprint
import re
import sys

class Log():
    def __init__(self, logfile_abspath):
        self._file_path = logfile_abspath
        self.latency = []
        self.bandwidth = []
        # regex matches the latency and bandwidth lines in the log files
        self._regex = '.*[0-9]+\.[0-9]+.*[0-9]\.[0-9].*'

    def open(self):
        try:
            self._file_handle = open(self._file_path, 'r')
        except:
            sys.exit('failed to open: ' + self._file_path)

    def parse(self):
        for line in self._file_handle:
            if re.match(self._regex, line):
                entries = line.split()
                self.latency.append(round(float(entries[0]), 4))
                self.bandwidth.append(round(float(entries[1]), 4))

class Pair():
    def __init__(self, first_logfile_abspath, second_logfile_abspath):
        self.first = Log(first_logfile_abspath)
        self.first.open()
        self.first.parse()
        self.second = Log(second_logfile_abspath)
        self.second.open()
        self.second.parse()

    def _ratio(self, a, b):
        diff = [round((x - y), 4) for x, y in zip(a, b)]
        ratio = []
        for numerator, denominator in zip(diff, a):
            try:
                ratio.append(round(numerator / denominator, 4))
            except:
                ratio.append(float(0.0000))
        return ratio

    def _percent(self, ratio):
        perc = ['{0:.2%}'.format(x) for x in ratio]
        return perc

    def _percentage_difference(self, a, b):
        ratio = self._ratio(a, b)
        percent = self._percent(ratio)
        return percent

    def calculate_differences(self):
        self.latency_percentage_differences = \
            self._percentage_difference(self.first.latency,
                                        self.second.latency)
        self.bandwidth_percentage_differences = \
            self._percentage_difference(self.first.bandwidth,
                                        self.second.bandwidth)

    def dump(self):
        delim = ', '
        output =  '\tlatency:'
        output += '\n\t\t'
        output += delim.join(map(str, self.first.latency))
        output += '\n\t\t'
        output += delim.join(map(str, self.second.latency))
        output += '\n\t\t'
        output += delim.join(map(str, self.latency_percentage_differences))
        output += '\n\tbandwidth:'
        output += '\n\t\t'
        output += delim.join(map(str, self.first.bandwidth))
        output += '\n\t\t'
        output += delim.join(map(str, self.second.bandwidth))
        output += '\n\t\t'
        output += delim.join(map(str, self.bandwidth_percentage_differences))
        return output

class Tracker():
    def __init__(self, args, archives):
        self._args = args
        self._archives = archives
        self._data = {}

    def add(self, changeset, most_recent_build_id):
        archive_path = self._archives.path_of_build(most_recent_build_id)
        for filename in self._args.logs:
            abs_file_path = archive_path + '/' + filename
            log = Log(abs_file_path)
            log.open()
            log.parse()
            key = (changeset, filename)
            self._data[key] = log

    def dump(self):
        out_str = ''
        for key in self._data.keys():
            log = self._data[key]
            line_str = pprint.pformat(key, width=120)
            line_str += ' = '
            line_str += pprint.pformat(log.latency, width=120)
            line_str += '\n'
            out_str += line_str
        return out_str
