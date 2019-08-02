from lunzi import log

skips = {
    # 'lunzi',
    # 'lunzi.*',
    'lunzi.injector',
    'lunzi.base_flags',
    'lunzi.experiment',
    'ipdb.*', 'pdb',
    'numpy', 'numpy.*',
    'torch', 'torch.*',
    'tensorflow', 'tensorflow.*',
}


def _monkey_patch():
    try:
        import ipdb
    except ImportError:
        log.critical(f'skip patching `ipdb`: `ipdb` not found.')
        return

    import os
    env_var = 'PYTHONBREAKPOINT'
    if env_var in os.environ:
        log.critical(f'skip patching `ipdb`: environment variable `{env_var}` has been set.')
        return
    os.environ[env_var] = 'ipdb.set_trace'

    old_init_pdb = ipdb.__main__._init_pdb

    def _init_pdb(*args, **kwargs):
        p = old_init_pdb(*args, **kwargs)
        p.skip = skips
        return p

    ipdb.__main__._init_pdb = _init_pdb
    log.critical(f'`ipdb` patched...')


_monkey_patch()
