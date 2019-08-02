from typing import Union, IO, Any
import numpy as np


def save(obj: Any, file: Union[str, IO]):
    np.save(file, obj)


def load(file: Union[str, IO]):
    return np.load(file)[()]

