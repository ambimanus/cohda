# coding=utf-8

from __future__ import division

import os
import time
import inspect
import numpy as np
from functools import wraps
from itertools import izip, ifilter, starmap
from contextlib import contextmanager

import progressbar as pbar


# http://code.activestate.com/recipes/551763/
def autoassign(*names, **kwargs):
    """
    autoassign(function) -> method
    autoassign(*argnames) -> decorator
    autoassign(exclude=argnames) -> decorator

    allow a method to assign (some of) its arguments as attributes of
    'self' automatically.  E.g.

    >>> class Foo(object):
    ...     @autoassign
    ...     def __init__(self, foo, bar): pass
    ...
    >>> breakfast = Foo('spam', 'eggs')
    >>> breakfast.foo, breakfast.bar
    ('spam', 'eggs')

    To restrict autoassignment to 'bar' and 'baz', write:

        @autoassign('bar', 'baz')
        def method(self, foo, bar, baz): ...

    To prevent 'foo' and 'baz' from being autoassigned, use:

        @autoassign(exclude=('foo', 'baz'))
        def method(self, foo, bar, baz): ...
    """
    if kwargs:
        exclude, f = set(kwargs['exclude']), None
        sieve = lambda l: ifilter(lambda nv: nv[0] not in exclude, l)
    elif len(names) == 1 and inspect.isfunction(names[0]):
        f = names[0]
        sieve = lambda l: l
    else:
        names, f = set(names), None
        sieve = lambda l: ifilter(lambda nv: nv[0] in names, l)

    def decorator(f):
        fargnames, _, _, fdefaults = inspect.getargspec(f)
        # Remove self from fargnames and make sure fdefault is a tuple
        fargnames, fdefaults = fargnames[1:], fdefaults or ()
        defaults = list(sieve(izip(reversed(fargnames), reversed(fdefaults))))

        @wraps(f)
        def decorated(self, *args, **kwargs):
            assigned = dict(sieve(izip(fargnames, args)))
            assigned.update(sieve(kwargs.iteritems()))
            for _ in starmap(assigned.setdefault, defaults):
                pass
            #self.__dict__.update(assigned)
            # better (more compatible):
            for k, v in assigned.iteritems():
                setattr(self, k, v)
            return f(self, *args, **kwargs)
        return decorated

    return f and decorator(f) or decorator


def import_object(module, obj=None):
    components = module.split('.')
    if obj != None:
        components.append(obj)
    c = __import__(components[0])
    for component in components[1:]:
        c = getattr(c, component)
    return c


class GeneratorSpeed(pbar.ProgressBarWidget):
    def __init__(self):
        self.fmt = 'Speed: %d/s'
    def update(self, pbar):
        if pbar.seconds_elapsed < 2e-6:#== 0:
            bps = 0.0
        else:
            bps = float(pbar.currval) / pbar.seconds_elapsed
        return self.fmt % bps


class PBar(pbar.ProgressBar):
    def __init__(self, maxval):
        pbar.ProgressBar.__init__(self, widgets=[pbar.Percentage(), ' ',
                pbar.Bar(), ' ', pbar.ETA(), ' ', GeneratorSpeed()],
                maxval=maxval)

    # def update(self, value=None):
    #     if value is None:
    #         pbar.ProgressBar.update(self, self.currval + 1)
    #     else:
    #         pbar.ProgressBar.update(self, value)

    def update(self, value=None):
        "Updates the progress bar to a new value."
        if value is None:
            value = self.currval + 1
        assert 0 <= value <= self.maxval
        self.currval = value
        if not self._need_update() or self.finished:
            return
        if not self.start_time:
            self.start_time = time.time()
        self.seconds_elapsed = time.time() - self.start_time
        self.prev_percentage = self.percentage()
        if value != self.maxval:
            self.fd.write(self._format_line() + '\r')
        else:
            self.finished = True
            self.fd.write(self._format_line() + '\r')


def get_repo_revision():
    repo_path, repo_type = get_repo_root()
    if repo_type == 'hg':
        from mercurial import hg, ui, commands
        ui = ui.ui()
        repo = hg.repository(ui, repo_path)
        ui.pushbuffer()
        commands.identify(ui, repo, rev='.')
        return ui.popbuffer().split()[0]
    elif repo_type == 'git':
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def get_repo_root():
    path = os.path.dirname(os.path.realpath(__file__))
    while os.path.exists(path):
        if '.git' in os.listdir(path):
            return path, 'git'
        elif '.hg' in os.listdir(path):
            return path, 'hg'
        path = os.path.realpath(os.path.join(path, '..'))
    raise RuntimeError('No git/hg repository found!')


def current_method():
    mod = inspect.getmodulename(inspect.stack()[1][1])
    func = inspect.stack()[1][3]
    return mod, func


def locdict(fl, exclude=()):
    return dict([(k, fl[k]) for k in sorted(fl.keys()) if k not in exclude])


def norm(minimum, maximum, value):
    # return value
    if maximum == minimum:
        return maximum
    return (value - minimum) / (maximum - minimum)


@contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


# http://code.activestate.com/recipes/577586-converts-from-decimal-to-any-base-between-2-and-26/
def base10toN(num, base):
    """Change ``num'' to given base
    Upto base 36 is supported."""

    converted_string, modstring = "", ""
    currentnum = num
    if not 1 < base < 37:
        raise ValueError("base must be between 2 and 36")
    if not num:
        return '0'
    while currentnum:
        mod = currentnum % base
        currentnum = currentnum // base
        converted_string = chr(48 + mod + 7*(mod > 10)) + converted_string
    return converted_string


def resample(d, resolution):
    return (d.reshape(d.shape[0]/resolution, resolution).sum(1)/resolution)
