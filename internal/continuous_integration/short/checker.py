#!/tool/pandora64/.package/python-3.8.0/bin/python3

import archive_path
import log
import dictionary
import report
import violation

class Performance():
    def __init__(self, args, archives, changeset_to_build,
                 build_to_relation_chain):
        self._args = args
        self._archives = archives
        self._changeset_to_build = changeset_to_build
        self._build_to_relation_chain = build_to_relation_chain
        self._build_id = build_to_relation_chain.most_recent_build()
        self._archive_path = archives.path_of_build(self._build_id)
        self._output = report.Report(self._build_id,
                                     self._archive_path,
                                     'performance_diff.txt')

    def _other_build_id(self, other_changeset):
        packed_id = [build_id for chng,
                     build_id in self._changeset_to_build.data.items()
                         if chng.startswith(other_changeset)]

        # The 'packed_id' variable is a list containing lists.
        # We need the content inside the packed_id data structure.
        try:
            build_id = packed_id[0][0]
            return True, build_id
        except IndexError:
            # An index error can occur if builds in the relation chain
            # have not been tested before attempting to test this
            # changeset.
            return False, 0

    def _log_difference(self, log_filename, other_changeset,
                        other_archive_path, violations):
        print('determining difference of log file ' + log_filename)
        self._output.record(log_filename)

        current_file_path = self._archive_path + '/' + log_filename
        other_file_path = other_archive_path + '/' + log_filename
        log_pair = log.Pair(current_file_path, other_file_path)
        log_pair.calculate_differences()

        latency_perc = [float(i.strip('%')) \
                for i in log_pair.latency_percentage_differences]
        max_latency = max(latency_perc)
        violations.check(max_latency, other_changeset, log_filename)

        self._output.record(log_pair.dump())

    def _changeset_difference(self, current_changeset, other_changeset):
        violations = violation.Threshold(self._args.latency_max, 'latency')

        change_pair = '(' + current_changeset + ',' + other_changeset + ')'
        print('comparing changesets ' + change_pair)
        self._output.record(change_pair)

        status, other_build_id = self._other_build_id(other_changeset)
        if status == False:
            message = 'skipping changeset ' + other_changeset
            print(message)
            self._output.record(message)
            return violations

        other_archive_path = self._archives.path_of_build(other_build_id)
        print(self._archive_path)
        print(other_archive_path)

        for filename in self._args.logs:
            self._log_difference(filename, other_changeset,
                                 other_archive_path, violations)
        print('\n')

        return violations

    def _calculate_performance_differences(self):
        current_changeset = \
            self._build_to_relation_chain.data[self._build_id][0]
        other_changesets = \
            self._build_to_relation_chain.data[self._build_id][1:]

        for other_changeset in other_changesets:
            violations = self._changeset_difference(current_changeset,
                                                    other_changeset)

        # Only report on the last pairwise changeset combination.
        # This combination represents the changeset being tested and
        # the amd-master:HEAD.
        violations.provide_violations_to_report(self._output)

    def run(self):
        self._output.open()
        self._calculate_performance_differences()
