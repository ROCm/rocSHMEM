#!/tool/pandora64/.package/python-3.8.0/bin/python3

import absolute_path
import glob

class Archive(absolute_path.PathGlobber):
    def __init__(self, args, name=''):
        archive_path = args.archive_path
        super().__init__(name, args.jenkins_path, archive_path,
                         args.benchmark_path)

    def path_of_build(self, build_id):
        path = self._search_path.replace('*/archive', build_id + '/archive')
        path = glob.glob(path)
        return path[0]
