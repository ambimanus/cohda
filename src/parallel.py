# coding=utf-8

from __future__ import division

import sys
import os

from definitions import *
from logger import *
from util import import_object
from configuration import Configuration
import cli


def run(par):
    scenario, cfg_dict, sc_dict = par
    cfg = Configuration(log_to_file=True, **cfg_dict)
    sc = import_object('scenarios', scenario)(cfg.rnd, cfg.seed, **sc_dict)
    cfg.scenario = sc

    cli.run(cfg)


if __name__ == '__main__':
    params = []
    runs = 100

    # for h in range(1, 101):
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,h%03d' % h,
    #             'opt_h': h,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for m in (20, 30, 40, 50, 100):
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,m-%03d' % m,
    #             'opt_m': m,
    #             'opt_h': 5,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))


    # for n in (10, 100, 1000, 10000):
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,n-%03d' % n,
    #             'opt_n': n,
    #             'opt_h': 5,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))


    # for q in (10, 20, 30, 40, 50, 100):
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,q-%03d' % q,
    #             'opt_q': q,
    #             'opt_h': 5,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for network in ('sequence', 'ring', 'stigspace', 'smallworld', 'random', 'mesh_rect', 'full'):
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,network-%s' % network,
    #             'opt_h': 5,
    #             'network_type': network,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for phi in (0.0, 0.1, 0.5, 1.0, 2.0, 4.0):
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,h-rnd,phi-%.1f' % phi,
    #             'opt_h': 'random',
    #             'network_phi': phi,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    for msgdelay in range(1, 11):
        for seed in range(runs):
            scenario = 'SC'
            cfg_dict = {
                'seed': seed,
                'msg_delay_min': 0,
                'msg_delay_max': msgdelay,
                'agent_delay_min': 0,
                'agent_delay_max': 0,
            }
            sc_dict = {
                KW_TITLE: 'SC,h-rnd,msgdelay-%02d' % msgdelay,
                'opt_h': 'random',
            }
            params.append((scenario, cfg_dict, sc_dict))

    # for agentdelay in range(1, 11):
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': agentdelay,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,h-rnd,agentdelay-%02d' % agentdelay,
    #             'opt_h': 'random',
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for p in range(10):
    #     p = p / 10.0
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,p_static-%0.1f' % p,
    #             'opt_h': 5,
    #             'opt_p_refuse_max': p,
    #             'opt_p_refuse_dynamic': False,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for p in range(10):
    #     p = p / 10.0
    #     for seed in range(runs):
    #         scenario = 'SC'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 0,
    #             'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'SC,p_dynamic-%0.1f' % p,
    #             'opt_h': 5,
    #             'opt_p_refuse_max': p,
    #             'opt_p_refuse_dynamic': True,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for m in (10, 20, 30, 40, 50, 100):
    #     for seed in range(runs):
    #         scenario = 'CHP'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             # 'agent_delay_min': 0,
    #             # 'agent_delay_max': 0,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'CHP,m%03d,n10,q4' % m,
    #             'opt_m': m,
    #             'opt_n': 10,
    #             'opt_q': 4,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for agentdelay, opt_n in ((30, 200),):
    #     for seed in range(runs):
    #         scenario = 'CHP'
    #         cfg_dict = {
    #             'seed': seed,
    #             'msg_delay_min': 0,
    #             'msg_delay_max': 0,
    #             'agent_delay_min': 1,
    #             'agent_delay_max': agentdelay,
    #             'max_simulation_steps': 100,
    #         }
    #         sc_dict = {
    #             KW_TITLE: 'CHP,stigspace,n%d,agentdelay%d' % (opt_n, agentdelay),
    #             'agent_type': 'AgentStigspaceMMMSSP',
    #             'opt_m': 30,
    #             'opt_n': opt_n,
    #             'opt_q': 96,
    #         }
    #         params.append((scenario, cfg_dict, sc_dict))

    # for seed in range(runs):
    #     scenario = 'CHP'
    #     cfg_dict = {
    #         'seed': seed,
    #         'msg_delay_min': 0,
    #         'msg_delay_max': 0,
    #         'agent_delay_min': 0,
    #         'agent_delay_max': 0,
    #     }
    #     sc_dict = {
    #         KW_TITLE: 'CHP,stigspace-ring',
    #         'agent_type': 'AgentMMMSSP',
    #         'network_type': 'stigspace',
    #         'opt_m': 30,
    #         'opt_n': 200,
    #         'opt_q': 96,
    #     }
    #     params.append((scenario, cfg_dict, sc_dict))


    if 'SGE_TASK_ID' in os.environ:
        # HERO HPC cluster
        run(params[int(os.environ['SGE_TASK_ID'])-1])
    elif 'PARALLEL_SEQ' in os.environ:
        # GNU parallel
        run(params[int(os.environ['PARALLEL_SEQ'])-1])
    else:
        # sequential
        start, stop = 0, len(params)
        if len(sys.argv) == 2:
            print len(params)
            sys.exit(0)
        elif len(sys.argv) == 3:
            start, stop = int(sys.argv[1]), int(sys.argv[2])
        if start >= len(params):
            sys.exit(0)
        for p in params[start:stop]:
            run(p)
