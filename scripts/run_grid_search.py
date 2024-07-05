#!/usr/bin/env python3
import argparse
import asyncio
import logging  # noqa
import os
from typing import List, Type, Tuple

import matplotlib
import matplotlib.pyplot as plt
from awesome.run.config import Config
from awesome.run.runner import Runner
from awesome.serialization.json_convertible import JsonConvertible
from awesome.util.logging import basic_config

from awesome.run.grid_search_config import GridSearchConfig
from awesome.run.grid_search_runner import GridSearchRunner

plt.ioff()
matplotlib.use('agg')


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config(app_name: str = None):
    from awesome.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    basic_config(app_name=app_name)


def get_config() -> GridSearchConfig:
    parser = argparse.ArgumentParser(
        description='Performs a grid search by creating multiple jobs / runners based on a given param grid.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Adding a config
    parser.add_argument(
        "--config-path",
        help="Path to load the config from. Can be json or yaml.",
        type=str, default=None, required=False)

    parser = GridSearchConfig.get_parser(parser)
    args = parser.parse_args()

    config: GridSearchConfig = None
    if args.config_path:
        config = GridSearchConfig.load_from_file(args.config_path)
        config.apply_parsed_args(args)
    else:
        config = GridSearchConfig.from_parsed_args(args)
    if config.diff_config is not None:
        config.diff_config = JsonConvertible.from_json(config.diff_config)
    config.prepare()
    return config


async def main(config: GridSearchConfig):
    runner = GridSearchRunner(config)
    runner.build(build_children=False)

    # Training
    if config.create_job_file:
        logging.info(f"Creating job file...")
        file = runner.create_job_file()
        logging.info(f"Created job file at: {file}")

    if not config.dry_run:
        logging.info(f"Start training of: {config.name_experiment}")
        runner.train()

if __name__ == "__main__":
    config()
    cfg = get_config()
    config(app_name=cfg.name_experiment)
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(cfg))
        loop.close()
    except Exception as err:
        logging.exception(
            f"Raised {type(err).__name__} in {current_filename()}, exiting...")
        exit(1)
    exit(0)
