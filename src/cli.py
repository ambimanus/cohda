# coding=utf-8

from __future__ import division

import datetime

import logger
from logger import *
from stats import Stats
from mas import Mas, Message
from util import norm
from agent import Agent
from model import Working_Memory, Solution_Candidate


def run(sc):
    ts = datetime.datetime.now().replace(microsecond=0).isoformat('_')
    sc.sim_time_start = ts
    logger.setup(sc)
    store_scenario(sc)
    INFO('Init (%s)' % ts)

    INFO('Fitness (minimal): %.6f' % sc.sol_fitness_min)
    INFO('Fitness (maximal): %.6f' % sc.sol_fitness_max)
    INFO('Fitness (average): %.6f' % sc.sol_fitness_avg)

    INFO('Creating %d agents' % sc.opt_m)
    agents = dict()
    for aid, search_space, initial_value in zip(
            sc.agent_ids, sc.agent_search_spaces, sc.agent_initial_values):
        agents[aid] = Agent(aid, search_space, initial_value)

    INFO('Connecting agents')
    for a, neighbors in sc.network.items():
        for n in neighbors:
            # Consistency check
            assert a != n, 'cannot add myself as neighbor!'
            # Add neighbor
            DEBUG('', 'Connecting', a, '->', n)
            if n not in agents[a].neighbors:
                agents[a].neighbors[n] = agents[n]
            else:
                WARNING(n, 'is already neighbor of', a)

    INFO('Starting simulation')
    mas = Mas(sc, agents)
    logger.set_mas(mas)
    stats = Stats(sc, agents)
    stats.eval(mas.current_time)
    AGENT(mas.aid, 'Notifying initial agent (%s)' % sc.sim_initial_agent)
    kappa = Working_Memory(sc.objective, dict(),
                           Solution_Candidate(None, dict(), float('-inf')))
    msg = Message(mas.aid, sc.sim_initial_agent, kappa)
    mas.msg(msg)
    while mas.is_active():
        mas.step()
        stats.eval(mas.current_time)
    if not stats.is_converged():
        ERROR('convergence not reached!')

    ts = datetime.datetime.now().replace(microsecond=0).isoformat('_')
    INFO('End (%s)' % ts)

    # Store scenario again, this time with simulation result
    store_scenario(sc, overwrite=True)

    return stats
