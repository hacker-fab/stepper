import base64
import copy
import logging
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

# TypedDict was added to typing in 3.8. Use typing_extensions for <3.8
if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


class Timer(object):
    def __init__(self, name: str):
        self.name: str = name
        self.start: Optional[float] = None
        self.end: Optional[float] = None

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type_, value, traceback):
        self.end = time.time()
        logging.debug("%s time: %s", self.name, self.end - self.start)


JSONArrayType = TypedDict(
    "JSONArrayType",
    {"@type": str, "base64": str, "dtype": str, "shape": Tuple[int, ...]},
)


def deserialise_array_b64(
    b64_string: str, dtype: Union[Type[np.dtype], str], shape: Tuple[int, ...]
):
    flat_arr: np.ndarray = np.frombuffer(base64.b64decode(b64_string), dtype)
    return flat_arr.reshape(shape)


def serialise_array_b64(npy_arr: np.ndarray) -> Tuple[str, str, Tuple[int, ...]]:
    b64_string: str = base64.b64encode(npy_arr.tobytes()).decode("ascii")
    dtype: str = str(npy_arr.dtype)
    shape: Tuple[int, ...] = npy_arr.shape
    return b64_string, dtype, shape


def ndarray_to_json(arr: np.ndarray) -> JSONArrayType:
    if isinstance(arr, memoryview):
        # We can transparently convert memoryview objects to arrays
        # This comes in very handy for the lens shading table.
        arr = np.array(arr)
    b64_string, dtype, shape = serialise_array_b64(arr)
    return {"@type": "ndarray", "dtype": dtype, "shape": shape, "base64": b64_string}


def json_to_ndarray(json_dict: JSONArrayType):
    if json_dict.get("@type") != "ndarray":
        logging.warning("No valid @type attribute found. Conversion may fail.")
    for required_param in ("dtype", "shape", "base64"):
        if not json_dict.get(required_param):
            raise KeyError(f"Missing required key {required_param}")

    b64_string: Optional[str] = json_dict.get("base64")
    dtype: Optional[str] = json_dict.get("dtype")
    shape: Optional[Tuple[int, ...]] = json_dict.get("shape")

    if b64_string and dtype and shape:
        return deserialise_array_b64(b64_string, dtype, shape)
    else:
        raise ValueError("Required parameters for decoding are missing")


@contextmanager
def set_properties(obj, **kwargs):
    """A context manager to set, then reset, certain properties of an object.

    The first argument is the object, subsequent keyword arguments are properties
    of said object, which are set initially, then reset to their previous values.
    """
    saved_properties = {}
    for k in kwargs.keys():
        try:
            saved_properties[k] = getattr(obj, k)
        except AttributeError:
            print(
                "Warning: could not get {} on {}.  This property will not be restored!".format(
                    k, obj
                )
            )
    for k, v in kwargs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved_properties.items():
            setattr(obj, k, v)


def axes_to_array(
    coordinate_dictionary: Dict[str, Optional[int]],
    axis_keys: Sequence[str] = ("x", "y", "z"),
    base_array: Optional[List[int]] = None,
    asint: bool = True,
) -> List[int]:
    """Takes key-value pairs of a JSON value, and maps onto an array
    
    This is designed to take a dictionary like `{"x": 1, "y":2, "z":3}`
    and return a list like `[1, 2, 3]` to convert between the argument
    format expected by most of our stages, and the usual argument
    format in JSON.
    
    `axis_keys` is an ordered sequence of key names to extract from
    the input dictionary.

    `base_array` specifies a default value for each axis.  It must 
    have the same length as `axis_keys`.
    
    `asint` casts values to integers if it is `True` (default).
    
    Missing keys, or keys that have a `None` value will be left
    at the specified default value, or zero if none is specified.
    """
    # If no base array is given
    if not base_array:
        # Create an array of zeros
        base_array = [0] * len(axis_keys)
    else:
        # Create a copy of the passed base_array
        base_array = copy.copy(base_array)

    # Do the mapping
    for axis, key in enumerate(axis_keys):
        if key in coordinate_dictionary:
            value = coordinate_dictionary[key]
            if value is None:
                # Values set to None should be treated as if they
                # are missing
                # i.e. we leave the default value in place.
                break
            if asint:
                value = int(value)
            base_array[axis] = value

    return base_array
