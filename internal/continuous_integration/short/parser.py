#!/tool/pandora64/.package/python-3.8.0/bin/python3

import argparse

class Parser():
    def __init__(self):
        # A parent directory containing log file output from one of the
        # configuration runs. The output directories are intended to
        # be symmetric in naming with the various configurations supplied
        # by the library's build_configs.
        self._default_config = 'RC_SINGLE'

        # The list of log files which need to be checked for performance
        # differences.
        self._default_logs = ['get.log',
                              'get_nbi.log',
                              'get_swarm.log',
                              'put.log',
                              'put_nbi.log']

        # The maximum pairwise difference for the log file latencies.
        self._default_latency_max = 5.0

        # The minimum bandwidth difference for the log file bandwidths.
        self._default_bandwidth_min = -50.0

        # The Jenkins tester archives slave output on the master's
        # filesystem which currently uses this top-level path (as the
        # resperf account).
        self._default_jenkins_path = \
                '/proj/radl_extra/users/resperf/jenkins-2.192/'

        # The performance tester runs as part of the 'short' job to
        # verify that no performance degradation has occurred between
        # commits. This archive path is the generic archive path
        # for all of the builds. The Kleene star is used as a place
        # holder for the Jenkins build number.
        self._default_archive_path = \
                'jobs/shmem_short/builds/*/archive/'

        # The default benchmark path can be used to alter archive
        # output placement. Currently, this is initialized to an empty
        # string, but subsequently initialized to inject the config
        # path.
        self._default_benchmark_path = ''

    def setup_options(self, argparser):
        argparser.add_argument('-j',
                               dest='jenkins_path',
                               default=self._default_jenkins_path)
        argparser.add_argument('-a',
                               dest='archive_path',
                               default=self._default_archive_path)
        argparser.add_argument('-b',
                               dest='benchmark_path',
                               default=self._default_benchmark_path)
        argparser.add_argument('-c',
                               dest='config',
                               default=self._default_config)
        argparser.add_argument('-l',
                               dest='logs',
                               nargs='*',
                               default=self._default_logs)
        argparser.add_argument('-x',
                               dest='latency_max',
                               type=float,
                               default=self._default_latency_max)
        argparser.add_argument('-y',
                               dest='bandwidth_min',
                               type=float,
                               default=self._default_bandwidth_min)
        argparser.add_argument('-o',
                               dest='one_changeset')
        argparser.add_argument('-r',
                               dest='changeset_range',
                               nargs=2,
                               metavar=("most_recent_changeset", "least_recent_changeset"))
        return argparser

    def parse_command_line(self):
        p = argparse.ArgumentParser()
        p = self.setup_options(p)
        args = p.parse_args()
        args.benchmark_path = args.config + args.benchmark_path
        return args
