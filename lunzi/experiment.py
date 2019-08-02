import numpy as np
from logging import getLogger, FileHandler, Logger

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

import lunzi as lz
from .base_flags import MetaFLAGS, merge, set_value
from .file_storage import FileStorage


def add_file_handler(logger: Logger, file_path: str):
    import coloredlogs
    file_handler = FileHandler(file_path)
    file_handler.setFormatter(coloredlogs.BasicFormatter(fmt='%(asctime)s - %(filename)s:%(lineno)d - %(message)s'))
    logger.addHandler(file_handler)


def get_logger(name: str) -> Logger:
    import coloredlogs

    logger = getLogger(name)
    coloredlogs.install(
        logger=logger,
        milliseconds=True,
        fmt='%(asctime)s - %(filename)s:%(lineno)d - %(message)s',
        field_styles={**coloredlogs.DEFAULT_FIELD_STYLES, 'filename': {'color': 'cyan'}},
    )

    return logger


def set_random_seed(seed: int):
    import random
    random.seed(seed)

    np.random.seed(seed)

    try:
        import tensorflow
        tensorflow.set_random_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_string(s: str):
    try:
        import ast
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def init(root: MetaFLAGS, doc: str = ''):
    if 'seed' not in root:
        import os
        root.add('seed', int.from_bytes(os.urandom(3), 'little'))
    args, unknown = parse(root, doc)

    seed = root.seed
    root.freeze()
    log_dir = args.log_dir

    set_random_seed(seed)

    if log_dir is not None:
        lz.fs.init(log_dir)
        lz.writer = SummaryWriter(logdir=str(lz.fs.log_dir))
        dump(root)
        add_file_handler(lz.log, lz.fs.resolve('$LOGDIR/out.log'))
        lz.log.warning(f'log_dir = {str(lz.fs.log_dir)}')
    else:
        lz.log.critical('no log_dir provided')
    if unknown:
        lz.log.critical('unknown arguments: %s', unknown)

    if args.print_config:
        import toml
        print('----- FLAGS begin -----')
        print(toml.dumps(root.as_dict()))
        print('----- FLAGS end -----')


def dump(root: MetaFLAGS):
    import toml

    with open(lz.fs.resolve('$LOGDIR/config.toml'), 'w') as f:
        toml.dump(root.as_dict(), f)

    lz.info['log_dir'] = lz.fs.log_dir
    with open(lz.fs.resolve('$LOGDIR/meta.toml'), 'w') as f:
        toml.dump(lz.info, f)


def set_default_injector():
    from .injector import ParamInjector, DefaultInjector
    injectors = [
        ParamInjector('_seed', lambda *_: np.random.randint(1, 10 ** 9), int, cache=True),
        ParamInjector('_rng', lambda *_: np.random.RandomState(np.random.randint(0, 2 ** 32 - 1)),
                      np.random.RandomState, cache=True),
        ParamInjector('_fs', lambda *_: lz.fs, FileStorage),
        ParamInjector('_writer', lambda *_: lz.writer, SummaryWriter),
        ParamInjector('_log', lambda *_: lz.log, Logger),
        ParamInjector('_info', lambda *_: lz.info, dict),
    ]

    for injector in injectors:
        DefaultInjector.register(injector.key, injector)


set_default_injector()


def parse(root: MetaFLAGS, doc=''):
    import toml
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument('-c', '--config', help='configuration file (TOML)', action='append', metavar='FILE')
    parser.add_argument('-s', '--set', help='additional options', nargs=2, action='append', metavar=('PATH', 'VALUE'))
    parser.add_argument('--print_config', help='print configs', action='store_true')
    parser.add_argument('--log_dir', help='the directory to logs', default='/tmp')

    args, unknown = parser.parse_known_args()
    if args.config:
        for config in args.config:
            merge(root, toml.load(open(Path(config).expanduser())))
    if args.set:
        for path, value in args.set:
            set_value(root, path.split('.'), parse_string(value))

    return args, unknown


def close():
    lz.writer.close()


def main(root: MetaFLAGS, doc: str = ''):
    def decorate(fn):
        def decorated():
            init(root, doc)
            fn()
            close()
        return decorated
    return decorate
