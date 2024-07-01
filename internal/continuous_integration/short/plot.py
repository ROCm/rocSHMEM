#!/tool/pandora64/.package/python-3.8.0/bin/python3

import parser
import dictionary
import archive_path
import plotter

def main():
    p = parser.Parser()
    args = p.parse_command_line()

    archives = archive_path.Archive(args)
    archives.generate()
    print(archives.dump())

    build_to_changeset = dictionary.BuildToChangesetDict()
    build_to_changeset.generate(archives.dirs)
    print(build_to_changeset.dump())

    changeset_to_build = dictionary.ChangesetToBuildDict()
    changeset_to_build.generate(build_to_changeset)
    print(changeset_to_build.dump())

    plot = plotter.Plot(args,
                        archives,
                        changeset_to_build)

    # either plot with all the changesets or the slice provided
    plot.changeset_slice()
    
    if (args.one_changeset):
        plot.one_changeset_plot()

if __name__ == '__main__':
    main()
