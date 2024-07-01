#!/tool/pandora64/.package/python-3.8.0/bin/python3

import pprint
import report
import sys

class Threshold():
    def __init__(self, maximum_threshold, violation_type):
        self._violations = {}
        self._maximum_threshold = maximum_threshold
        self._violation_type = violation_type

    def check(self, value, changeset, filename):
        if value >= self._maximum_threshold:
            key = changeset + '|' + filename + '|' + self._violation_type
            self._violations[key] = value
            print(key + ': ' + str(value) + '%')

    def provide_violations_to_report(self, report):
        if self.has_violations():
            report.record('FAILURE')
            report.record(self.dump())
            sys.exit(1)
        else:
            report.record('SUCCESS')
            sys.exit(0)

    def has_violations(self):
        return bool(self._violations)

    def dump(self):
        str_out = pprint.pformat(self._violations, width=120)
        str_out += '\n'
        return str_out
