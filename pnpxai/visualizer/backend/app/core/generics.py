from flask_restx import Resource
from json import JSONEncoder as BaseJSONEncoder
from abc import abstractclassmethod
from pnpxai.visualizer.backend.app.core.constants import APIItems
import numpy as np
import torch
from torch import Tensor


class Controller(Resource):
    @classmethod
    def response(cls, code: int = 200, message: str = 'Success', data=None, errors=None):
        return {
            APIItems.MESSAGE.value: message,
            APIItems.DATA.value: data,
            APIItems.ERRORS.value: errors,
        }, code


class Response:
    @abstractclassmethod
    def to_dict(cls, object):
        raise NotImplementedError

    @classmethod
    def dump(cls, object, many=False):
        data = None
        if many:
            if isinstance(object, dict):
                data = {k: cls.to_dict(v) for k, v in object.items()}
            else:
                data = [cls.to_dict(o) for o in object]
        else:
            data = cls.to_dict(object)

        return data


class JSONEncoder(BaseJSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            if obj.ndim == 0:
                return obj.item()
            return obj.tolist()

        return super(JSONEncoder, self).default(obj)
