#!/usr/bin/env python3
from awesome.util.path_tools import format_os_independent, relpath
import argparse
import logging  # noqa
import os

import matplotlib
import matplotlib.pyplot as plt
from awesome.run.config import Config
from awesome.util.logging import basic_config

from awesome.run.awesome_config import AwesomeConfig
from awesome.run.awesome_runner import AwesomeRunner

plt.ioff()
matplotlib.use('agg')


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    from awesome.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    basic_config()


def get_config() -> Config:
    parser = argparse.ArgumentParser(
        description='Just does segmentation stuff.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Adding a config
    parser.add_argument(
        "--config-path", type=str, default=None, required=False)

    parser = AwesomeConfig.get_parser(parser)
    args = parser.parse_args()

    config: AwesomeConfig = None
    if args.config_path:
        args.config_path = args.config_path.strip("\"").strip("\'")
        config = AwesomeConfig.load_from_file(args.config_path)
        config.apply_parsed_args(args)
    else:
        config = AwesomeConfig.from_parsed_args(args)

    # Set run script path
    config.run_script_path = format_os_independent(
        relpath(os.getcwd(), os.path.abspath(__file__), is_from_file=False))

    config.prepare()
    return config


def main(config: Config):
    logging.info(f"Setup: {config.name_experiment}")
    # Setup

    runner = AwesomeRunner(config)
    runner.build()

    # Save config and log it
    cfg_file = runner.store_config()
    runner.log_config()

    logging.info(f"Stored config in: {cfg_file}")

    # Training
    logging.info(f"Start training of: {config.name_experiment}")
    with plt.ioff():
        runner.train()


if __name__ == "__main__":
    config()
    cfg = get_config()
    main(cfg)
    logging.info(f"Finished {current_filename()} successfully. Goodbye!")

