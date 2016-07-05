# coding=utf-8

from __future__ import division

import datetime

from logger import *
from visualizations import Stats
from stigspace import Stigspace
from simulator import Simulator
from util import import_object, norm


def run(cfg):
    ts = datetime.datetime.now().replace(microsecond=0).isoformat('_')
    cfg.time_start = ts
    setup_logger(cfg)
    store_cfg(cfg)
    sc = cfg.scenario
    INFO('Init', ts)

    def pr(name, low, high, val):
        return '%s=%f [normalized %f]' % (name, val, norm(low, high, val))

    INFO(pr('d_min', sc[KW_SOL_D_MIN], sc[KW_SOL_D_MAX], sc[KW_SOL_D_MIN]))
    INFO(pr('d_max', sc[KW_SOL_D_MIN], sc[KW_SOL_D_MAX], sc[KW_SOL_D_MAX]))
    INFO(pr('d_avg', sc[KW_SOL_D_MIN], sc[KW_SOL_D_MAX], sc[KW_SOL_D_AVG]))

    INFO('Creating', sc[KW_OPT_M], 'agents of type', sc[KW_AGENT_TYPE])
    agents = dict()
    Agent = import_object(sc[KW_AGENT_MODULE], sc[KW_AGENT_TYPE])
    for i, aid in enumerate(sc[KW_AGENT_IDS]):
        # Get weights
        if KW_OPT_W_DICT in sc:
            w = sc[KW_OPT_W_DICT][aid]
        else:
            w = sc[KW_OPT_W][i]
        # Get sol_init
        if KW_SOL_INIT_DICT in sc:
            sol_init = sc[KW_SOL_INIT_DICT][aid]
        else:
            sol_init = sc[KW_SOL_INIT][i]
        # Start agent process
        a = Agent(aid, w, sol_init)
        # If desired, set p_refuse and seed for feasibility check
        if KW_OPT_P_REFUSE in sc and sc[KW_OPT_P_REFUSE][i] > 0:
            seed = cfg.rnd.randint(0, cfg.rng_seed_max)
            a.set_p_refuse(sc[KW_OPT_P_REFUSE][i], seed)
        if 'Stigspace' in sc[KW_AGENT_TYPE]:
            Stigspace.set_active(True)
        agents[aid] = a

    INFO('Connecting agents')
    for a, neighbors in sc[KW_NETWORK].items():
        for n in neighbors:
            # Consistency check
            assert a != n, 'cannot add myself as neighbor!'
            # Add neighbor
            DEBUG('', 'Connecting', a, '->', n)
            if n not in agents[a].neighbors:
                agents[a].neighbors[n] = agents[n]
            else:
                WARNING(n, 'is already neighbor of', a)

    sim = Simulator(cfg, agents)

    INFO('Starting simulation')
    stats = Stats(sc, agents)
    sim.init()
    stats.eval(sim.current_time)
    while sim.is_active():
        sim.step()
        stats.eval(sim.current_time)
    stats.eval_final()

    ts = datetime.datetime.now().replace(microsecond=0).isoformat('_')
    INFO('End %s' % ts)

    # Store cfg again, this time with simulation result
    store_cfg(cfg, overwrite=True)

    return stats
