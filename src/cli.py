# coding=utf-8

from __future__ import division

import datetime

from configuration import Configuration
from logger import *
from visualizations import Stats
from stigspace import Stigspace
from simulator import Simulator
from util import import_object, norm
import scenarios


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
    for i in range(sc[KW_OPT_M]):
        aid = int(sc[KW_AGENT_IDS][i])
        # Determine seed for MOCOMixin
        seed = cfg.rnd.randint(0, cfg.rng_seed_max)
        # Start agent process
        a = Agent(aid, sc[KW_OPT_W][i], sc[KW_SOL_INIT][i],
            cfg.rnd.randint(cfg.agent_delay_min, cfg.agent_delay_max),
            seed, sc[KW_OPT_P_REFUSE][i])
        if 'Stigspace' in sc[KW_AGENT_TYPE]:
            Stigspace.set_active(True)
        agents[aid] = a

    INFO('Connecting agents')
    for a, neighbors in sc[KW_NETWORK].items():
        for n in neighbors:
            DEBUG('', 'Connecting', n, '->', a)
            agents[a].add_peer(n, agents[n])

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


if __name__ == '__main__':
    cfg = Configuration(
        seed=0,
        msg_delay_min=None,
        msg_delay_max=None,
        agent_delay_min=None,
        agent_delay_max=None,
        log_to_file=False,
    )
    sc = scenarios.SC(cfg.rnd, cfg.seed, opt_h='random')
    # sc = scenarios.SVSM(cfg.rnd, cfg.seed)
    # sc = scenarios.CHP(cfg.rnd, cfg.seed, opt_m=10, opt_n=400, opt_q=16, opt_q_constant=20.0)
    #
    # sc = scenarios.SC(cfg.rnd, cfg.seed, opt_h=5,
    #                   agent_type='AgentStigspaceMMMSSP')
    # sc = scenarios.SVSM(cfg.rnd, cfg.seed, agent_type='AgentStigspaceSVSM')
    # sc = scenarios.CHP(cfg.rnd, cfg.seed, opt_m=30, opt_n=200,
    #                    agent_type='AgentStigspaceMMMSSP')
    cfg.scenario = sc

    run(cfg)
