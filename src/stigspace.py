# coding=utf-8

from __future__ import division

from logger import msg_counter


class Stigspace(object):

    active = False
    _data = {}
    _updates = {}


    @classmethod
    def set_active(cls, active):
        cls.active = active

    @classmethod
    def put(cls, aid, sol):
        assert cls.active
        msg_counter()
        cls._updates[aid] = sol

    @classmethod
    def read(cls):
        assert cls.active
        msg_counter()
        return dict(cls._data)

    @classmethod
    def step(cls):
        assert cls.active
        cls._data.update(cls._updates)
