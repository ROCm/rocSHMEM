#!/tool/pandora64/.package/python-3.8.0/bin/python3

import sys

class Report():
    def __init__(self, identifier, path, filename):
        self._identifier = identifier
        self._path = path
        self._filename = filename

    def open(self):
        print('opening report for ' + self._identifier)
        try:
            report_path = self._path + '/' + self._filename
            print('report_path: ' + report_path)
            self._file_handle = open(report_path, 'w')
        except:
            sys.exit('failed to open report: ' + report_path)

    def record(self, message):
        self._file_handle.write(message + '\n')
