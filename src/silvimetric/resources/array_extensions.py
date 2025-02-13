from __future__ import annotations

from typing_extensions import Sequence

import re

import numpy as np
import pandas as pd
from pandas._libs import lib
from pandas._typing import npt

@pd.api.extensions.register_extension_dtype
class AttributeDtype(pd.api.extensions.ExtensionDtype):
    type = np.generic
    na_value = None

    def __init__(self, subtype=np.float32):
        self._subtype = np.dtype(subtype)

    def __str__(self) -> str:
        return f'AttributeDtype[{self.subtype}]'

    def __repr__(self) -> str:
        return f'AttributeDtype[{self.subtype}]'

    # TestDtypeTests
    def __hash__(self) -> int:
        return hash(str(self))


    @property
    def subtype(self):
        return self._subtype

    @property
    def name(self):
        return f"AttributeDtype[{self.subtype}]"

    @classmethod
    def construct_array_type(cls):
        return AttributeArray

    @classmethod
    def construct_from_string(cls, string):
        if string.lower() == "attribute":
            return cls()
        match = re.match(r"^attribute\[(\w+)\]$", string, re.IGNORECASE)
        if match:
            return cls(match.group(1))
        raise TypeError(f"Cannot construct a 'AttributeDtype' from '{string}'")


class AttributeArray(pd.api.extensions.ExtensionArray):
    def __init__(self, arrays, dtype):
        assert isinstance(dtype, AttributeDtype)
        self._dtype = dtype
        # self._data = np.array([np.array(array, dtype=dtype.subtype) for array in arrays], dtype=dtype.subtype)
        self._data = [ np.asarray(array, dtype=dtype.subtype) for array in arrays ]

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if dtype is None and bool(scalars.dtype):
            return cls(scalars, AttributeDtype(scalars.dtype))
        elif dtype is None:
            cls(scalars, AttributeDtype(np.generic))
        return cls(scalars, dtype)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    @property
    def dtype(self):
        return self._dtype

    def copy(self):
        return type(self)(self._data, self._dtype)

    def take(self, indices, allow_fill=False, fill_value=None):
        """ Take elements from an array.  """

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = pd.core.algorithms.take(self._data, indices, allow_fill=allow_fill,
                                         fill_value=fill_value)
        return self._from_sequence(result)

    @classmethod
    def _from_factorized(cls, uniques: np.ndarray, original: AttributeArray):
        """ Reconstruct an AttributeArray after factorization.  """
        return cls(uniques, AttributeDtype(uniques.dtype))

    @pd.core.ops.unpack_zerodim_and_defer('__eq__')
    def __eq__(self, other):
        return self._apply_operator('__eq__', other, recast=False)

    def nbytes(self):
        return self._data.nbytes

    def isna(self):
        """ A 1-D array indicating if each value is missing.  """
        return pd.isnull(self._data)

    def copy(self):
        """ Return a copy of the array.  """
        copied = self._data.copy()
        return type(self)(copied, self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[AttributeArray]) -> AttributeArray:
        """
        Concatenate multiple AngleArrays.
        """
        return cls(np.concatenate(to_concat))

