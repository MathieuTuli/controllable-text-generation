"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""

from argparse import ArgumentParser

from ..data import DATASETS, TASKS
from . import DECODE_METHODS


def decode_args(sub_parser: ArgumentParser) -> None:
    decode_group = sub_parser.add_argument_group('decode')
    decode_group.add_argument("--model", type=str)
    decode_group.add_argument("--weights", type=str)
    decode_group.add_argument("--tokenizer", type=str)
    decode_group.add_argument("--batch-size", type=int)
    decode_group.add_argument("--max-length", type=int)
    decode_group.add_argument("--method", choices=DECODE_METHODS)
    decode_group.add_argument("--beam-width", default=-1, type=int)
    decode_group.add_argument("--beam-depth", default=-1, type=int)
    decode_group.add_argument("--top-k", default=-1, type=int)
    decode_group.add_argument("--context-length", default=2, type=int)
    decode_group.add_argument("--oracle", default=False, action='store_true')
    decode_group.add_argument("--max-samples", default=-1, type=int)

    data_group = sub_parser.add_argument_group('data')
    data_group.add_argument("--name", type=str, choices=DATASETS)
    data_group.add_argument("--src", type=str)
    data_group.add_argument("--vocab", type=str)
    data_group.add_argument("--task", type=str, choices=TASKS)
    data_group.add_argument("--overwrite-cache", type=str)
    data_group.add_argument("--num-workers", type=int, default=4)
    data_group.add_argument("--prefix", type=str, default='')

    io_group = sub_parser.add_argument_group('io')
    io_group.add_argument("--output", type=str)
    io_group.add_argument("--cache-dir", type=str)
