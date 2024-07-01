#!/tool/pandora64/.package/python-3.8.0/bin/python3

import abc
import os
import pprint
import subprocess
import sys

class BaseDict(metaclass=abc.ABCMeta):
    def __init__(self):
        self.data = {}
        self._delimiter_path = 'archive'
        self._changeset_delta_filename = 'changeset_delta.txt'

    def _build_id(self, build_directory):
        sub_directory_strings = build_directory.split('/')
        word_count = 0
        for word in sub_directory_strings:
            if word == '':
                continue
            if word == self._delimiter_path:
                break
            word_count += 1
        bld_id = sub_directory_strings[word_count]
        return bld_id

    def _open_changeset_delta_file(self, archive_directory):
        build_directory, config_directory = os.path.split(archive_directory)
        changeset_file_path =  build_directory + '/' + \
                               self._changeset_delta_filename
        try:
            file_handle = open(changeset_file_path, 'r')
        except:
            sys.exit('failed to open: ' + changeset_file_path)
        return file_handle

    @abc.abstractmethod
    def _changeset_delta_operations(self, file_handle, bld_id):
        pass

    def generate(self, archives):
        for d in archives:
            bld_id = self._build_id(d)
            f = self._open_changeset_delta_file(d)
            self._changeset_delta_operations(f, bld_id)

    def most_recent_build(self):
        build_id_strings = self.data.keys()
        build_id_ints = list(map(int, build_id_strings))
        most_recent_build_id_int = max(build_id_ints)
        return str(most_recent_build_id_int)

    def dump(self):
        str_out = self._print_text
        str_out += pprint.pformat(self.data, width=120)
        str_out += '\n'
        return str_out

class BuildToChangesetDict(BaseDict):
    def __init__(self, name=''):
        super().__init__()
        self._print_text = name

    def _changeset_delta_operations(self, file_handle, bld_id):
        commit_line = file_handle.readline()
        try:
            commit_hash = commit_line.split()[0]
        except IndexError:
            commit_hash = None
        if commit_hash != None:
            self.data[bld_id] = commit_hash

class BuildToRelationChainDict(BaseDict):
    def __init__(self, name=''):
        super().__init__()
        self._print_text = name

    def _changeset_delta_operations(self, file_handle, bld_id):
        changes = []
        for line in file_handle:
            changes.append(line.split()[0])
        self.data[bld_id] = changes

class ChangesetToBuildDict():
    def __init__(self, name=''):
        self.data = {}
        self._print_text = name

    def _invert_dict(self, dictionary):
        dict_with_duplicates = {}
        for key, value in dictionary.data.items():
            list_with_duplicates = dict_with_duplicates.get(value, [])
            list_with_duplicates.append(key)
            dict_with_duplicates[value] = list_with_duplicates
        return dict_with_duplicates

    def generate(self, dictionary):
        self.data = self._invert_dict(dictionary)

    def dump(self):
        str_out = self._print_text
        str_out += pprint.pformat(self.data, width=120)
        str_out += '\n'
        return str_out

class ChangelogToMostRecentBuild():
    def __init__(self, name=''):
        self._print_text = name
        self._all_changesets = []
        self._changesets_with_builds = []
        self._changesets_without_builds = []
        self.data = {}

    def _build_id(self, changeset_to_build, changeset):
        try:
            build_id_strings = changeset_to_build.data[changeset]
            build_id_ints = list(map(int, build_id_strings))
            most_recent_build_id_int = max(build_id_ints)
            build_id_str = str(most_recent_build_id_int)
        except:
            build_id_str = ''
        return build_id_str

    def _changelog(self):
        # print git hash along with file modification stats
        shellcmd =  'git log --pretty=tformat:"%H" --shortstat | '
        # condense the output down to single line
        shellcmd += "awk 'ORS=NR%3?\" \":\"\\n\"' | "
        # parse out the git hash by itself
        shellcmd += "awk '{print $1}'"
        x = subprocess.getoutput(shellcmd)
        self._all_changesets = x.split()

    def _with_builds(self, changeset_to_build):
        changesets =  list(changeset_to_build.data.keys())
        self._changesets_with_builds = changesets

    def _without_builds(self):
        self._changesets_without_builds = \
            list(set(self._all_changesets) - \
                 set(self._changesets_with_builds))

    def generate(self, changeset_to_build):
        self._changelog()
        self._with_builds(changeset_to_build)
        self._without_builds()
        for changeset in self._all_changesets:
            if changeset in self._changesets_with_builds:
                build = self._build_id(changeset_to_build, changeset)
                self.data[changeset] = build

    def dump(self):
        str_out = self._print_text
        str_out += 'git-log_changesets_in_order:\n'
        str_out += pprint.pformat(self._all_changesets, width=120)
        str_out += '\nfilesystem_with_builds:\n'
        str_out += pprint.pformat(self._changesets_with_builds, width=120)
        str_out += '\nfilesystem_without_builds:\n'
        str_out += pprint.pformat(self._changesets_without_builds, width=120)
        str_out += '\ngit-log_changesets_to_build-id_mappings:\n'
        str_out += pprint.pformat(self.data, width=120)
        str_out += '\n'
        return str_out
