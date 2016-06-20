# coding=utf-8

from __future__ import division

import random

from definitions import *
from util import autoassign


class Configuration(object):

    @autoassign
    def __init__(self,
                 seed=0,
                 log_to_file=False,
                 rng_seed_max=2**30,
                 max_simulation_steps=None,
                 msg_delay_min=0,
                 msg_delay_max=2,
                 agent_delay_min=1,
                 agent_delay_max=5,
                 random_speaker=True):
        self.rnd = random.Random(seed)


if __name__ == '__main__':
    import sys
    import pickle

    if len(sys.argv) == 1:
        cfg1 = Configuration()
        pickle.dump(cfg1, open('test.pickle', 'w'))
        cfg2 = pickle.load(open('test.pickle'))
        print cfg1.__dict__
        print cfg2.__dict__
    else:
        cfg = pickle.load(open(sys.argv[1]))
        for k, v in cfg.__dict__.items():
            if isinstance(v, dict):
                print k
                for k2, v2 in v.items():
                    if k2 == KW_SOL:
                        print '    ', k2, '<%d items>' % len(v2)
                    else:
                        print '    ', k2, v2
            else:
                print k, v

