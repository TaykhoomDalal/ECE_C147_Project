import numpy as np
import os
from argparse import ArgumentParser
import yaml


def load_data(data_root='data/'):
    """
    Loads the project data as a dict of np arrays

    :param data_root: the source where the np arrays are located
    :return: data - a dict mapping the data name to the np array
    """

    data = {}
    for name in os.listdir(data_root):
        ns = name.split('.')
        if ns[-1] == 'npy' and len(ns) == 2:
            data[ns[0]] = np.load(os.path.join(data_root, name))

    return data


def parse_args_with_config():
    """
    First parses args for a -c [config] or --config [config] to find a config, then loads the config as defaults and
    parses again.  All options and values are lowercased

    :return: args - a namespace of options for the program
    """

    # parse config first
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default="exps/template.yaml")
    known_args, _ = parser.parse_known_args()

    # open config
    f = open(known_args.config)
    config = yaml.safe_load(f.read())
    f.close()

    # add config options as defaults
    for key, val in config.items():
        if type(val) is str:
            val = val.lower()
        if type(val) is bool:
            parser.add_argument(f"--{key.lower()}", action='store_true', default=val)
        else:
            parser.add_argument(f"--{key.lower()}", default=val, type=type(val))

    args = parser.parse_args()
    return args

