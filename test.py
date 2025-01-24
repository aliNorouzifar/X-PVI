from functions.python_emsc.algorithm import stochastic

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import pm4py


logger = logging.getLogger(__name__)


def _read_stochastic_language(path_log: Path) -> Dict[Tuple[str], float]:
    logger.info(f"Importing log {path_log}")
    log = pm4py.read_xes(str(path_log))
    slang = stochastic.log_to_stochastic_language(log)
    return slang


def _main_emscc_log_log():
    logging.basicConfig(level=logging.INFO)
    ##############################
    # Arguments
    ##############################
    parser = argparse.ArgumentParser(
        prog='EMSCC for Log vs Log',
        description=''
    )

    parser.add_argument('logOne', help=r'E:\PADS\Projects\X-PVI\assets\test.xes', type=str)
    parser.add_argument('logTwo', help=r'E:\PADS\Projects\X-PVI\assets\test.xes', type=str)

    # args = parser.parse_args()
    # path_log_1 = Path(args.logOne)
    # path_log_2 = Path(args.logTwo)
    path_log_1 = r'E:\PADS\Projects\X-PVI\assets\test.xes'
    path_log_2 = r'E:\PADS\Projects\X-PVI\assets\test.xes'

    (slang_1, slang_2) = tuple(_read_stochastic_language(path_log) for path_log in (path_log_1, path_log_2))

    for i in range(10):
        emsc_log_log = stochastic.compare_languages_levenshtein(slang_1, slang_2)

    print(emsc_log_log)


if __name__ == '__main__':
    _main_emscc_log_log()
    # TODO Distinguish between timed and immediate transitions
    # print(_main_emsc_impl())
