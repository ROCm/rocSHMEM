#!/tool/pandora64/.package/python-3.8.0/bin/python3

import parser
import dictionary
import archive_path
import checker

def main():
    # This script accepts command line values, but has reasonable defaults
    # needed to run as part of the CI infrastructure.
    p = parser.Parser()
    args = p.parse_command_line()

    # Jenkins is configured to archive build artifacts in a directory.
    # The 'archives' variable holds the set of directories for
    # successful Jenkins builds (those which run to completion).
    # Partitioning of successful builds is useful since we can ignore
    # failed build directories while searching for performance data.
    archives = archive_path.Archive(args)
    archives.generate()
    print(archives.dump())

    # Jenkins records changeset information in a changeset_delta.txt file.
    # We parse the changelog for the commit hash and save it into
    # 'builds_to_changesets'.
    build_to_changeset = dictionary.BuildToChangesetDict()
    build_to_changeset.generate(archives.dirs)
    print(build_to_changeset.dump())

    # 'changeset_to_build' holds the changeset mappings with a
    # list of build numbers that match the changeset value.
    # Builds may be executed many times with the same changeset.
    # The most recent build (identified by the largest build number) will
    # be used to retrieve performance data.
    changeset_to_build = dictionary.ChangesetToBuildDict()
    changeset_to_build.generate(build_to_changeset)
    print(changeset_to_build.dump())

    # Jenkins is configured to dump Gerrit-esque relation chain changesets
    # to an archived output file 'changeset-delta.txt'.
    # The relation chain will be used to determine changeset performance
    # data for each changeset in the relation chain (when possible).
    build_to_relation_chain = dictionary.BuildToRelationChainDict()
    build_to_relation_chain.generate(archives.dirs)
    print(build_to_relation_chain.dump())

    perf_checker = checker.Performance(args,
                                       archives,
                                       changeset_to_build,
                                       build_to_relation_chain)
    perf_checker.run()

if __name__ == '__main__':
    main()
