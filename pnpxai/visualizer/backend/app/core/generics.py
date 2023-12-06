from flask import jsonify
from flask_restx import Resource
from abc import abstractclassmethod
from pnpxai.visualizer.backend.app.core.constants import APIItems


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
