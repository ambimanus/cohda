# coding=utf-8

import sys
import os
import random
import json
import datetime

import numpy as np

import util
import scenarios
import cli
from configuration import Configuration
from logger import *


class ScenarioEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return list(obj.timetuple())
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class Scenario(object):

    def from_JSON(self, js):
        if type(js) == str:
            js = json.loads(js)
        # Import data
        self.__dict__ = js

        # Recreate sub-instances
        for k in list(js.keys()):
            if k[:2] == 't_':
                tt = js[k]
                setattr(self, k, datetime.datetime(*tt[:6]))
        self.rnd = random.Random(self.seed)


    def load_JSON(self, filename):
        # Read file
        with open(filename, 'r') as fp:
            js = json.load(fp)

        self.from_JSON(js)


    def to_JSON(self):
        d = dict(self.__dict__)
        for k in list(d.keys()):
            if k == 'devices' or k == 'rnd' or k[:2] == 'i_':
                del d[k]

        return json.dumps(d, indent=2, cls=ScenarioEncoder)


    def save_JSON(self, filename):
        with open(filename, 'w') as fp:
            fp.write(self.to_JSON())


    def __str__(self):
        return self.to_JSON()


if __name__ == '__main__':
    sc_file = sys.argv[1]
    basedir = os.path.dirname(sc_file)
    s = Scenario()
    s.load_JSON(sc_file)

    dfn = str(os.path.join(basedir, '.'.join((str(s.seed), 'cohda', 'npy'))))
    if os.path.exists(dfn):
        raise RuntimeError('File already exists: %s' % dfn)

    # now = datetime.datetime.now()
    # ts = now.isoformat().split('T')[0]
    s.run_cohda_ts = datetime.datetime.now()
    s.rev_cohda = util.get_repo_revision()

    opt_q = int((s.t_block_end - s.t_block_start).total_seconds() / 60)
    if opt_q == 0:
        opt_m = sum([d[1] for d in s.device_templates])
        result = np.zeros((opt_m, opt_q))
        np.save(dfn, result)
        s.sched_file = os.path.basename(dfn)
        s.save_JSON(sc_file)
        sys.exit(0)

    cfg = Configuration(seed=s.seed, log_to_file=True, basepath=basedir)
    sc = scenarios.APPSIM_ENUM(cfg.rnd, s, basedir)
    Objective.calls = 0
    Objective.instance = 0
    cfg.scenario = sc


    stats = cli.run(cfg)


    solution = stats.solution
    result = np.array([solution[i] for i in sorted(solution.keys())])
    np.save(dfn, result)
    s.sched_file = os.path.basename(dfn)
    s.save_JSON(sc_file)
