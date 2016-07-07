# coding=utf-8

import os
import logging
import datetime
import pickle
import time
import random

from util import get_repo_revision

FORMAT = '[%(levelname)-9s] %(message)s'

_LVL_MSG = 9
_LVL_AGENTV = 12
_LVL_AGENT = 13
_LVL_STATS = 21
_LVL_SOLUTION = 22
_LVL_TARGET = 23

_LVL_MSG_NAME = 'MSG'
_LVL_AGENTV_NAME = 'AGENTV'
_LVL_AGENT_NAME = 'AGENT'
_LVL_STATS_NAME = 'STATS'
_LVL_SOLUTION_NAME = 'SOLUTION'
_LVL_TARGET_NAME = 'TARGET'

LOG_LEVEL = logging.INFO
# LOG_LEVEL = _LVL_AGENTV

FILTER = None
#FILTER = ('bkc','BKC')
FILTER_LVL = None
#FILTER_LVL = _LVL_SOLUTION


def setup(sc, makedir=True, lvl=LOG_LEVEL):
    basepath = sc.log_basepath

    if not '_logger' in globals():
        logging.addLevelName(_LVL_MSG, _LVL_MSG_NAME)
        logging.addLevelName(_LVL_AGENT, _LVL_AGENT_NAME)
        logging.addLevelName(_LVL_AGENTV, _LVL_AGENTV_NAME)
        logging.addLevelName(_LVL_STATS, _LVL_STATS_NAME)
        logging.addLevelName(_LVL_SOLUTION, _LVL_SOLUTION_NAME)
        logging.addLevelName(_LVL_TARGET, _LVL_TARGET_NAME)
        logging.basicConfig(level=lvl, format=FORMAT)
        globals()['_logger'] = logging.getLogger('cohda')
    elif 'filehandler' in globals():
        _logger.removeHandler(filehandler)

    globals()['sc.log_to_file'] = sc.log_to_file
    if sc.log_to_file:

        if makedir:
            ts = datetime.datetime.now().isoformat().split('T')[0]
            globals()['rev'] = get_repo_revision()
            globals()['dirname'] = '_'.join((ts, str(sc.title), rev))
            globals()['logdir'] = str(os.path.join(basepath, dirname))
            if not os.path.exists(logdir):
                try:
                    os.makedirs(logdir)
                except:
                    pass
            # try again after a short sleep
            time.sleep(random.random())
            if not os.path.exists(logdir):
                try:
                    os.makedirs(logdir)
                except:
                    raise RuntimeError('Cannot create directory')
        else:
            globals()['logdir'] = basepath

        filename = '.'.join((str(sc.seed), 'log'))
        f = str(os.path.join(logdir, filename))
        if os.path.exists(f):
            raise RuntimeError('Logfile already exists: %s' % f)
        globals()['filehandler'] = logging.FileHandler(f, mode='w')
        filehandler.setFormatter(logging.Formatter(fmt=FORMAT))
        _logger.addHandler(filehandler)

    globals()['scenario'] = sc
    globals()['first_time'] = True


def set_mas(mas):
    globals()['mas'] = mas


def store_scenario(sc, overwrite=False):
    if sc.log_to_file:
        scfilepath = str(os.path.join(logdir, '.'.join(
            ('scenario', str(sc.seed), 'pickle'))))
        if os.path.exists(scfilepath):
            if overwrite:
                os.remove(scfilepath)
            else:
                print 'WARNING: scenario file already exists!!'
        with open(scfilepath, 'w') as scfile:
            pickle.dump(sc, scfile)


def log(lvl, *msg):
    message = _string(*msg)
    if (lvl == FILTER_LVL or
            FILTER is None or any([True for f in FILTER if f in message])):
        _logger.log(lvl, message)


def _string(*msg):
    if globals()['first_time']:
        out = '  msg |   obj | '
        globals()['first_time'] = False
    else:
        message_counter = 0
        if 'mas' in globals():
            message_counter = mas.message_counter
        objective_counter = scenario.objective.call_counter
        out = '%5d | %5d | ' % (message_counter, objective_counter)
    if len(msg) > 1:
        if type(msg[0] == int or (type(msg[0] == str and len(msg[0]) > 0))):
            sep = ' | '
        else:
            sep = ' '
        if type(msg[0]) == int:
            out += '%d' % msg[0] + sep
        else:
            out += str(msg[0]) + sep
        msg = msg[1:]
    for s in msg:
        out += str(s) + ' '
    return out


def MSG(*msg):
    log(_LVL_MSG, *msg)


def AGENTV(*msg):
    log(_LVL_AGENTV, *msg)


def AGENT(*msg):
    log(_LVL_AGENT, *msg)


def STATS(*msg):
    log(_LVL_STATS, *msg)


def SOLUTION(*msg):
    log(_LVL_SOLUTION, *msg)


def DEBUG(*msg):
    log(logging.DEBUG, *msg)


def INFO(*msg):
    log(logging.INFO, *msg)


def WARNING(*msg):
    log(logging.WARNING, *msg)


def ERROR(*msg):
    log(logging.ERROR, *msg)
