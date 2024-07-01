#!/tool/pandora64/.package/python-3.8.0/bin/python3

import dictionary
import log
#import matplotlib.pyplot
import numpy
import csv
import os
import subprocess
import sys

class Plot():
    def __init__(self, args, archives, changeset_to_build):
        self._args = args
        self._archives = archives
        self._changelog = dictionary.ChangelogToMostRecentBuild()
        self._changelog.generate(changeset_to_build)
        print(self._changelog.dump())

    def abbreviate_changesets(self, changesets):
        return [changeset[0:8] for changeset in changesets]

    @staticmethod
    def write_dict_to_file(tracker, field_names, file_name):
        with open(file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(tracker)
    
    @staticmethod
    def check_and_add_to_dict(dictionary, key, array):
        if len(array) > 0:
            dictionary[key] = array[0]
        else:
            dictionary[key] = 0

    def changeset_slice(self):
        self._log_tracker = log.Tracker(self._args, self._archives)
        for changeset in self._changelog._all_changesets:
            if changeset in self._changelog.data.keys():
                build_id = self._changelog.data[changeset]
                self._log_tracker.add(changeset, build_id)
        print(self._log_tracker.dump())
        
        """
        separate out dictionaries based on operation
        and prepare them in a format that works with
        the csv module
        """
        put_tracker = []
        put_nbi_tracker = []
        get_tracker = []
        get_nbi_tracker = []
        amo_tracker = []
        ping_pong_tracker = []
        prev_commit = list(self._log_tracker._data.keys())[0][0]
        amo_dict = {}
        for key, value in self._log_tracker._data.items():
            if (key[1] == "put.log"):
                put_tracker.append({'Commit':key[0][0:7],
                                    'b1':value.latency[0],
                                    'b2':value.latency[1],
                                    'b4':value.latency[2],
                                    'b8':value.latency[3],
                                    'b16':value.latency[4],
                                    'b32':value.latency[5],
                                    'b64':value.latency[6],
                                    'b128':value.latency[7],
                                    'b256':value.latency[8],
                                    'b512':value.latency[9],
                                    'b1024':value.latency[10],
                                    'b2048':value.latency[11],
                                    'b4096':value.latency[12],
                                    'b8192':value.latency[13],
                                    'b16384':value.latency[14],
                                    'b32768':value.latency[15]
                                    })
            if (key[1] == "put_nbi.log"):
                put_nbi_tracker.append({'Commit':key[0][0:7],
                                    'b1':value.latency[0],
                                    'b2':value.latency[1],
                                    'b4':value.latency[2],
                                    'b8':value.latency[3],
                                    'b16':value.latency[4],
                                    'b32':value.latency[5],
                                    'b64':value.latency[6],
                                    'b128':value.latency[7],
                                    'b256':value.latency[8],
                                    'b512':value.latency[9],
                                    'b1024':value.latency[10],
                                    'b2048':value.latency[11],
                                    'b4096':value.latency[12],
                                    'b8192':value.latency[13],
                                    'b16384':value.latency[14],
                                    'b32768':value.latency[15]
                                    })
            if (key[1] == "get.log"):
                get_tracker.append({'Commit':key[0][0:7],
                                    'b1':value.latency[0],
                                    'b2':value.latency[1],
                                    'b4':value.latency[2],
                                    'b8':value.latency[3],
                                    'b16':value.latency[4],
                                    'b32':value.latency[5],
                                    'b64':value.latency[6],
                                    'b128':value.latency[7],
                                    'b256':value.latency[8],
                                    'b512':value.latency[9],
                                    'b1024':value.latency[10],
                                    'b2048':value.latency[11],
                                    'b4096':value.latency[12],
                                    'b8192':value.latency[13],
                                    'b16384':value.latency[14],
                                    'b32768':value.latency[15]
                                    })
            if (key[1] == "get_nbi.log"):
                get_nbi_tracker.append({'Commit':key[0][0:7],
                                    'b1':value.latency[0],
                                    'b2':value.latency[1],
                                    'b4':value.latency[2],
                                    'b8':value.latency[3],
                                    'b16':value.latency[4],
                                    'b32':value.latency[5],
                                    'b64':value.latency[6],
                                    'b128':value.latency[7],
                                    'b256':value.latency[8],
                                    'b512':value.latency[9],
                                    'b1024':value.latency[10],
                                    'b2048':value.latency[11],
                                    'b4096':value.latency[12],
                                    'b8192':value.latency[13],
                                    'b16384':value.latency[14],
                                    'b32768':value.latency[15]
                                    })
            if (key[1] == "ping_pong.log"):
                ping_pong_tracker.append({'Commit':key[0][0:7],
                                          'latency':value.latency[0]
                                         })

            # check to see if we have moved to a new commit
            # if we have, store the dict in the amo_tracker
            if (key[0] != prev_commit):
                amo_dict['Commit'] = prev_commit[0:7]
                amo_tracker.append(amo_dict.copy())
                amo_dict.clear()
            
            prev_commit = key[0]
            
            if (key[1] == "amo_add.log"):
                self.check_and_add_to_dict(amo_dict, 'add', value.latency)
            if (key[1] == "amo_cswap.log"):
                self.check_and_add_to_dict(amo_dict, 'cswap', value.latency)
            if (key[1] == "amo_fadd.log"):
                self.check_and_add_to_dict(amo_dict, 'fadd', value.latency)
            if (key[1] == "amo_fcswap.log"):
                self.check_and_add_to_dict(amo_dict, 'fcswap', value.latency)
            if (key[1] == "amo_fetch.log"):
                self.check_and_add_to_dict(amo_dict, 'fetch', value.latency)
            if (key[1] == "amo_finc.log"):
                self.check_and_add_to_dict(amo_dict, 'finc', value.latency)
            if (key[1] == "amo_inc.log"):
                self.check_and_add_to_dict(amo_dict, 'inc', value.latency)
        
        # store the last commit's amo data
        amo_dict['Commit'] = prev_commit[0:7]
        amo_tracker.append(amo_dict.copy())

        # write put results into a file:
        size_field_names= ['Commit','b1','b2','b4','b8','b16','b32','b64','b128','b256','b512','b1024','b2048','b4096','b8192','b16384','b32768']
        amo_field_names= ['Commit','add','cswap','fadd','fcswap','fetch','finc','inc']
        ping_pong_field_names= ['Commit','latency']

        self.write_dict_to_file(put_tracker, size_field_names, "put.csv")
        self.write_dict_to_file(put_nbi_tracker, size_field_names, "put_nbi.csv")
        self.write_dict_to_file(get_tracker, size_field_names, "get.csv")
        self.write_dict_to_file(get_nbi_tracker, size_field_names, "get_nbi.csv")
        self.write_dict_to_file(amo_tracker, amo_field_names, "amo.csv")
        self.write_dict_to_file(ping_pong_tracker, ping_pong_field_names, "ping_pong.csv")

        # make a directory and execute the R script to generate plots in that directory
        current_dir = os.getcwd()
        plot_dir = os.path.join(current_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        changeset_a = list(self._log_tracker._data.keys())[0][0]
        changeset_b = list(self._log_tracker._data.keys())[-1][0]

        # check if the provided changesets are correct
        if (self._args.changeset_range):
            found_changeset_a = False
            found_changeset_b = False
            for key, value in self._log_tracker._data.items():
                if (found_changeset_a and found_changeset_b):
                    break
                if (not found_changeset_a):
                    if (self._args.changeset_range[0] == key[0]):
                        found_changeset_a = True
                if (not found_changeset_b):
                    if (self._args.changeset_range[1] == key[0]):
                        found_changeset_b = True
            
            if ((not found_changeset_a) and (not found_changeset_b)):
                sys.exit("One of the specified changesets was not found. Please specify correct/complete commit IDs.")
            else:
                changeset_a = self._args.changeset_range[0]
                changeset_b = self._args.changeset_range[1]

        r_command = "Rscript ./plotter.R -o ./plots -a " + changeset_a[0:7] + " -b " + changeset_b[0:7]

        print(r_command)
        subprocess.check_call(r_command, shell=True)


    def one_changeset_plot(self):
        found_changeset = 0
        non_amo_tracker = []
        amo_tracker = []
        ping_pong_tracker = []
        for key, value in self._log_tracker._data.items():
            if (key[0] == self._args.one_changeset):
                found_changeset = 1
                if (key[1] == "put.log"):
                    put_vals = value.latency
                if (key[1] == "put_nbi.log"):
                    put_nbi_vals = value.latency
                if (key[1] == "get.log"):
                    get_vals = value.latency
                if (key[1] == "get_nbi.log"):
                    get_nbi_vals = value.latency
                if (key[1] == "amo_add.log"):
                    amo_tracker.append({'op':'add',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "amo_add.log"):
                    amo_tracker.append({'op':'add',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "amo_cswap.log"):
                    amo_tracker.append({'op':'cswap',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "amo_fadd.log"):
                    amo_tracker.append({'op':'fadd',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "amo_fcswap.log"):
                    amo_tracker.append({'op':'fcswap',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "amo_fetch.log"):
                    amo_tracker.append({'op':'fetch',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "amo_finc.log"):
                    amo_tracker.append({'op':'finc',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "amo_inc.log"):
                    amo_tracker.append({'op':'inc',
                                        'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })
                if (key[1] == "ping_pong.log"):
                    ping_pong_tracker.append({'latency': value.latency[0] if len(value.latency) > 0 else 0
                                        })


        if (not found_changeset):
            sys.exit("The requested changeset was not found. Please specify correct/complete commit IDs.")
        
        index = 0
        for size in [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]:
            non_amo_tracker.append({'size':size,
                                    'put':put_vals[index],
                                    'put_nbi':put_nbi_vals[index],
                                    'get':get_vals[index],
                                    'get_nbi':get_nbi_vals[index]
                                    })
            index = index + 1
        
        # write results into a file:
        non_amo_field_names= ['size','put','put_nbi','get','get_nbi']
        amo_field_names= ['op','latency']
        ping_pong_field_names= ['latency']

        self.write_dict_to_file(non_amo_tracker, non_amo_field_names, "non_amo_one_changeset.csv")
        self.write_dict_to_file(amo_tracker, amo_field_names, "amo_one_changeset.csv")
        self.write_dict_to_file(ping_pong_tracker, ping_pong_field_names, "ping_pong_one_changeset.csv")

        # call the R script with an option that tells it to plot figures for
        r_command = "Rscript ./plotter.R -o ./plots -c " + self._args.one_changeset

        print(r_command)
        subprocess.check_call(r_command, shell=True)

