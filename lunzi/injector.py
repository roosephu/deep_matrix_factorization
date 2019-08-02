from typing import Dict, Callable, Any, Tuple, List, Optional
from inspect import signature, Parameter
from dataclasses import dataclass

import wrapt

Injector = Callable[[Callable, dict], List[Tuple[str, Callable]]]


@dataclass
class ParamInjector:
    key: str
    getter: Callable
    annotation: Optional[type]
    cache: bool = False


class DefaultInjector:
    _params: Dict[str, ParamInjector] = {}
    _cache: Dict[Tuple[str, Callable], Any] = {}

    @staticmethod
    def register(tag: str, injector: ParamInjector):
        DefaultInjector._params[tag] = injector

    @staticmethod
    def inject(fn, parameters: dict) -> List[ParamInjector]:
        claims = []
        for param in DefaultInjector._params.keys() & parameters.keys():
            injector = DefaultInjector._params[param]

            key = (param, fn)
            if injector.cache:
                if key not in DefaultInjector._cache:
                    injection = injector.getter(fn)
                    DefaultInjector._cache[key] = injection
                else:
                    injection = DefaultInjector._cache[key]
                getter = lambda *_, injection_=injection: injection_
            else:
                getter = injector.getter
            claims.append(ParamInjector(param, getter, injector.annotation))
        return claims


# By default, all injections starting with _ are ignored, unless we run it from ours.
_default_injectors = [DefaultInjector.inject]


def inject(*injectors: Callable):
    injectors = list(injectors)

    def decorate(fn: Callable):
        sig = signature(fn)
        parameters = sig.parameters
        assigners = {}
        annotations = fn.__annotations__.copy()

        new_params: Dict[str, Parameter] = parameters.copy()
        for injector in injectors + _default_injectors:
            for param_injector in injector(fn, parameters):
                key = param_injector.key
                if key not in assigners:
                    assigners[key] = param_injector.getter
                    if param_injector.annotation:
                        annotations[key] = param_injector.annotation
                    # we enforce these parameters to be keyword-only parameters.
                    # to change the default value, you  have to explicitly specify it.
                    new_params[key] = new_params[key].replace(kind=Parameter.KEYWORD_ONLY)

        def adapter(): pass
        adapter.__annotations__ = annotations
        adapter.__signature__ = sig.replace(parameters=new_params.values())

        @wrapt.decorator(adapter=adapter)
        def injecting(wrapped, instance, args, kwargs):
            # easier for pdb... only need to step over one line
            new_kwargs = {key: getter() for key, getter in assigners.items()}
            new_kwargs.update(kwargs)
            return wrapped(*args, **new_kwargs)

        injected_fn = injecting(fn)
        injected_fn.__unwrapped__ = fn
        injected_fn.more = lambda *extra: inject(*extra, *injectors)
        injected_fn.injectors = injectors
        return injected_fn
    return decorate
