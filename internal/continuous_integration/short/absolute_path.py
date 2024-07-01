#!/tool/pandora64/.package/python-3.8.0/bin/python3

import glob
import pprint

class PathGlobber():
    def __init__(self, name, *partial_paths_to_concatenate):
        self._search_path = ''
        for partial_path in partial_paths_to_concatenate:
            self._search_path += partial_path
        self.dirs = []
        self._name = name

    def generate(self):
        self.dirs = glob.glob(self._search_path, recursive=True)

    def dump(self):
        str_out = self._name
        str_out += pprint.pformat(self.dirs, width=120)
        str_out += '\n'
        return str_out
