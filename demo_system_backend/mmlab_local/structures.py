# Extracted from mmengine.structures and mmaction.structures.
# BaseDataElement, InstanceData, ActionDataSample — stripped of registry deps.

import copy
import itertools
from collections.abc import Sized
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
#  BaseDataElement  (mmengine/structures/base_data_element.py)
# ---------------------------------------------------------------------------

class BaseDataElement:
    """A base data interface that supports Tensor-like and dict-like ops.

    Attributes are split into ``metainfo`` (image metadata) and ``data``
    (annotations / predictions).
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:
        self._metainfo_fields: set = set()
        self._data_fields: set = set()
        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        assert isinstance(metainfo, dict)
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_data(self, data: dict) -> None:
        assert isinstance(data, dict)
        for k, v in data.items():
            setattr(self, k, v)

    def new(self, *, metainfo: Optional[dict] = None,
            **kwargs) -> 'BaseDataElement':
        new_data = self.__class__()
        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def keys(self) -> list:
        private_keys = {
            '_' + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        return list(self._metainfo_fields)

    def values(self) -> list:
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        return self.metainfo_keys() + self.keys()

    def items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} is immutable.')
        else:
            self.set_field(
                name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item: str):
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} is immutable.')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        assert len(args) < 3
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(name)
            return self.__dict__.pop(*args)
        elif name in self._data_fields:
            self._data_fields.remove(name)
            return self.__dict__.pop(*args)
        elif len(args) == 2:
            return args[1]
        else:
            raise KeyError(f'{name} is not contained in metainfo or data')

    def __contains__(self, item: str) -> bool:
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(self, value: Any, name: str,
                  dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
                  field_type: str = 'data') -> None:
        assert field_type in ('metainfo', 'data')
        if dtype is not None:
            assert isinstance(value, dtype)
        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(
                    f'Cannot set {name} as metainfo — already a data field')
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'Cannot set {name} as data — already a metainfo field')
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like helpers
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                new_data.set_data({k: v.to(*args, **kwargs)})
        return new_data

    def cpu(self) -> 'BaseDataElement':
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                new_data.set_data({k: v.cpu()})
        return new_data

    def cuda(self) -> 'BaseDataElement':
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                new_data.set_data({k: v.cuda()})
        return new_data

    def __repr__(self) -> str:
        parts = [f'{self.__class__.__name__}(']
        for k, v in self.metainfo_items():
            parts.append(f'  META {k}={v!r}')
        for k, v in self.items():
            parts.append(f'  DATA {k}={v!r}')
        parts.append(')')
        return '\n'.join(parts)


# ---------------------------------------------------------------------------
#  InstanceData  (mmengine/structures/instance_data.py)
# ---------------------------------------------------------------------------

IndexType = Union[str, slice, int, list, torch.LongTensor,
                  torch.BoolTensor, np.ndarray]


class InstanceData(BaseDataElement):
    """Instance-level data container.  All data fields must have same length."""

    def __setattr__(self, name: str, value):
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super(BaseDataElement, self).__setattr__(name, value)
            else:
                raise AttributeError(f'{name} is immutable.')
        else:
            assert isinstance(value, Sized), \
                'value must contain `__len__` attribute'
            if len(self) > 0:
                assert len(value) == len(self), \
                    f'Length mismatch: {len(value)} vs {len(self)}'
            # bypass BaseDataElement.__setattr__ length-unaware path
            self.set_field(name=name, value=value,
                           field_type='data', dtype=None)

    __setitem__ = __setattr__

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                raise IndexError(f'Index {item} out of range!')
            item = slice(item, None, len(self))
        new_data = self.__class__(metainfo=self.metainfo)
        for k, v in self.items():
            new_data[k] = v[item]
        return new_data

    @staticmethod
    def cat(instances_list: List['InstanceData']) -> 'InstanceData':
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]
        new_data = instances_list[0].__class__(
            metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [inst[k] for inst in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_data[k] = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_data[k] = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                merged = v0[:]
                for v in values[1:]:
                    merged = merged + v
                new_data[k] = merged
            elif hasattr(v0, 'cat'):
                new_data[k] = v0.cat(values)
            else:
                raise ValueError(f'Cannot cat type {type(v0)} for key {k}')
        return new_data

    def __len__(self) -> int:
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        return 0


# ---------------------------------------------------------------------------
#  ActionDataSample  (mmaction/structures/action_data_sample.py)
# ---------------------------------------------------------------------------

class ActionDataSample(BaseDataElement):
    """Spatio-temporal action data sample."""

    @property
    def proposals(self):
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self):
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self):
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances
