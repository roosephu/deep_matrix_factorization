from typing import Union, List, Any

from .injector import inject, ParamInjector


class MetaFLAGS(type):
    _frozen = False
    seed: int

    def __setattr__(cls, key: str, value: Any):
        # assert not cls._frozen, 'Modifying frozen FLAGS.'
        super().__setattr__(key, value)

    def __getitem__(cls, item: str):
        return cls.__dict__[item]

    def add(cls, key: str, value: Any, overwrite=False, overwrite_false=False):
        if key not in cls or overwrite or not getattr(cls, key) and overwrite_false:
            setattr(cls, key, value)

    def __iter__(cls):
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not isinstance(value, classmethod):
                if isinstance(value, MetaFLAGS):
                    value = dict(value)
                yield key, value

    def as_dict(cls):
        return dict(cls)

    def freeze(cls):
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and isinstance(value, MetaFLAGS):
                value.freeze()
        cls.finalize()
        cls._frozen = True

    def _injector(cls, _, parameters):
        claims = []
        for param in parameters.keys() & cls.__dict__.keys():
            annotation = cls.__annotations__.get(param, None)
            claims.append(ParamInjector(param, lambda *_, param_=param: cls.__dict__[param_], annotation))
        return claims

    def finalize(cls):
        pass

    @property
    def inject(cls):
        """
            Generate a new `inject` instance, in case `fn.__injectors__`
            is changed.
        """
        return inject(cls._injector)


class BaseFLAGS(metaclass=MetaFLAGS):
    pass


def merge(lhs: Union[MetaFLAGS, dict], rhs: dict):
    # import ipdb; ipdb.set_trace()
    for key in rhs:
        keys = lhs if isinstance(lhs, dict) else lhs.__dict__
        assert key in keys, f"Can't find key `{key}`"
        if isinstance(lhs[key], (MetaFLAGS, dict)) and isinstance(rhs[key], dict):
            merge(lhs[key], rhs[key])
        else:
            if isinstance(lhs, dict):
                lhs[key] = rhs[key]
            else:
                setattr(lhs, key, rhs[key])


def set_value(cls: Union[MetaFLAGS, dict], path: List[str], value: Any):
    key, *rest = path
    keys = cls if isinstance(cls, dict) else cls.__dict__
    assert key in keys, f"Can't find key `{key}`"
    if not rest:
        if isinstance(cls, dict):
            cls[key] = value
        else:
            setattr(cls, key, value)
    else:
        assert isinstance(cls[key], (MetaFLAGS, dict))
        set_value(cls[key], rest, value)
