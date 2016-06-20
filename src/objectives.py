# coding=utf-8

from __future__ import division

import numpy as np


class Objective(object):

    instance = 0
    calls = 0

    def __init__(self):
        Objective.instance += 1
        self.instance = Objective.instance

    def _preprocess(self, x):
        return x

    def __call__(self, x, record_call=True):
        x = self._preprocess(x)
        if record_call:
            Objective.calls += 1
        return self._f(x)

    def _f(self, x):
        raise NotImplementedError('Subclass must implement _f(x).')

    def __repr__(self):
        return 'Objective #%d' % self.instance

    def _reset_call_counter(self):
        Objective.calls = 0


class Objective_Singlevalue(Objective):

    def __init__(self, target):
        Objective.__init__(self)

        self.target = target

    def _f(self, x):
        return abs(self.target - x)

    def __repr__(self):
        # return '%s %s' % (Objective.__repr__(self), str(self.target))
        return str({'instance': self.instance, 'target': self.target})


class Objective_Manhattan(Objective):

    def __init__(self, target):
        Objective.__init__(self)

        self.target = target

    def _preprocess(self, x):
        assert type(x) == np.ndarray or type(x) == np.ma.MaskedArray, type(x)
        assert len(x.shape) <= 2, x.shape
        if len(x.shape) == 2:
            x = np.sum(x, 0)
        if len(x) == 0:
            x = np.zeros(self.target.shape)
        if isinstance(self.target, np.ma.MaskedArray):
            x = np.ma.array(x, mask=self.target.mask)
        return x

    def _f(self, x):
        diff = np.abs(self.target - x)

        # return np.sum(diff)**2
        return np.sum(diff)

    def __repr__(self):
        # return '%s %s' % (Objective.__repr__(self), str(self.target))
        return str({'instance': self.instance, 'target': self.target})


class Objective_Peakshaving(Objective):

    def __init__(self, shape):
        Objective.__init__(self)
        self.target = np.zeros(shape)

    def _preprocess(self, x):
        assert type(x) == np.ndarray or type(x) == np.ma.MaskedArray, type(x)
        assert len(x.shape) <= 2, x.shape
        if len(x.shape) == 2:
            x = np.sum(x, 0)
        return x

    def _f(self, x):
        return np.max(x)


class Objective_Valleyfilling(Objective_Peakshaving):

    def __init__(self, shape):
        Objective_Peakshaving.__init__(self, shape)

    def _f(self, x):
        return -np.min(x)


class Objective_Spreadreduce(Objective_Peakshaving):

    def __init__(self, shape):
        Objective_Peakshaving.__init__(self, shape)

    def _f(self, x):
        return np.max(x) - np.min(x)


class Objective_Spreadreduce_SLP(Objective):

    def __init__(self, slp, sol_init):
        Objective.__init__(self)
        self.target = slp
        self.sol_init = sol_init

    def _preprocess(self, x):
        assert type(x) == np.ndarray or type(x) == np.ma.MaskedArray, type(x)
        assert len(x.shape) <= 2, x.shape
        if len(x.shape) == 2:
            x = np.sum(x, 0)
        return x

    def _f(self, x):
        diff = x - self.sol_init
        slp_new = self.target + diff
        return np.max(slp_new) - np.min(slp_new)
