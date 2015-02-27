# coding=utf-8

import os
import logging
import datetime
import pickle
import time
import random

from definitions import *
from objectives import Objective

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


def setup_logger(cfg, makedir=True, lvl=LOG_LEVEL):
    from util import get_repo_revision

    if hasattr(cfg, 'basepath'):
        basepath = cfg.basepath
    else:
        basepath = '../data'

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

    globals()['cfg.log_to_file'] = cfg.log_to_file
    if cfg.log_to_file:

        if makedir:
            ts = datetime.datetime.now().isoformat().split('T')[0]
            globals()['rev'] = get_repo_revision()
            globals()['dirname'] = '_'.join((ts, str(cfg.scenario[KW_TITLE]), rev))
            globals()['dir'] = str(os.path.join(basepath, dirname))
            if not os.path.exists(dir):
                try:
                    os.makedirs(dir)
                except:
                    pass
            # try again after a short sleep
            time.sleep(random.random())
            if not os.path.exists(dir):
                try:
                    os.makedirs(dir)
                except:
                    raise RuntimeError('Cannot create directory')
        else:
            globals()['dir'] = basepath


        filename = '.'.join((str(cfg.seed), 'log'))
        file = str(os.path.join(dir, filename))
        if os.path.exists(file):
            raise RuntimeError('Logfile already exists: %s' % file)
        globals()['filehandler'] = logging.FileHandler(file, mode='w')
        filehandler.setFormatter(logging.Formatter(fmt=FORMAT))
        _logger.addHandler(filehandler)

    globals()['message_counter'] = 0
    globals()['first_time'] = True


def reset_message_counter():
    globals()['message_counter'] = 0


def msg_counter():
    globals()['message_counter'] += 1


def store_cfg(cfg, overwrite=False):
    if cfg.log_to_file:
        cfgfilepath = str(os.path.join(dir, '.'.join(('cfg', str(cfg.seed),
                'pickle'))))
        if os.path.exists(cfgfilepath):
            if overwrite:
                os.remove(cfgfilepath)
            else:
                print 'WARNING: cfg already exists!!'
        with open(cfgfilepath, 'w') as cfgfile:
            pickle.dump(cfg, cfgfile)


def log(lvl, *msg):
    message = _string(*msg)
    if (lvl == FILTER_LVL or
            FILTER == None or any([True for f in FILTER if f in message])):
        _logger.log(lvl, message)


def _string(*msg):
    if globals()['first_time']:
        out = '  msg |   obj | '
        globals()['first_time'] = False
    else:
        out = '%5d | %5d | ' % (message_counter, Objective.calls)
    if len(msg) > 1:
        if type(msg[0] == int or (type(msg[0] == str and len(msg[0]) > 0))):
            sep = ' | '
        else:
            sep = ' '
        if type(msg[0]) == int:
            out += 'a%d' % msg[0] + sep
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
