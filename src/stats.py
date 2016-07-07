# coding=utf-8

from __future__ import division

from collections import OrderedDict as odict

import numpy as np

from logger import *


def contribution(agent):
    return agent.kappa.configuration[agent.aid]


class Stats(object):
    def __init__(self, scenario, agents):
        self.sc = scenario
        self.agents = agents
        self.solution = {aid: contribution(agents[aid]).value
                         for aid in agents}
        self.sol_counters = {aid: contribution(agents[aid]).v_lambda
                             for aid in agents}
        self.cand_sizes = {}
        self.cand_fitness = {}
        self.first_time = True
        self.new_solution = False
        self.candmax_fit = float('-inf')
        self.candmax_dist = 0.0
        self.candmax_size = 0.0
        self.sel = None
        self.eq = None
        self.fit_sel = None
        self.time_delta = 0.0

    def eval(self, current_time):
        opt_m, opt_q = self.sc.opt_m, self.sc.opt_q
        d_min, d_max = self.sc.sol_fitness_min, self.sc.sol_fitness_max
        obj = self.sc.objective

        # Collect agent states
        for aid in sorted(self.agents.keys()):
            a = self.agents[aid]
            if self.sol_counters[aid] != contribution(a).v_lambda:
                self.sol_counters[aid] = contribution(a).v_lambda
                self.solution[aid] = contribution(a).value
                self.new_solution = True
            if a.kappa.solution_candidate is not None:
                self.cand_sizes[aid] = len(a.kappa.solution_candidate.configuration)
                self.cand_fitness[aid] = a.kappa.solution_candidate.fitness

        # Set current timestamp relative to beginning of the heuristic
        current_time -= self.time_delta

        # sel := keys of solution candidates that should be considered.
        # At the beginning of the simulation, report all solution candidates.
        # But as soon as the first complete one has be found, restrict the
        # report to only complete solution candidates.
        # In either case, consider only the set of the largest solution
        # candidates available.
        sel = []
        max_size = max(self.cand_sizes.values())
        for k in sorted(self.cand_sizes.keys()):
            if self.cand_sizes[k] == max_size:
                sel.append(k)
        if len(self.cand_fitness) > 0 and len(sel) > 0:
            fit_sel = [self.cand_fitness[k] for k in sel]
            # candmax_fit := maximal fitness among all solution candidates
            candmax_fit = max(fit_sel)
            # eq := keys of fit_sel whith maximal value
            eq = []
            for k in sorted(self.cand_fitness.keys()):
                if abs(self.cand_fitness[k] - candmax_fit) < 0.00001:
                    eq.append(k)
                # TODO: Find better way to determine eq
            # candmax_dist := distribution of best solution candidate
            candmax_dist = len(eq) / opt_m
            # candmax_size := completeness of best solution candidate
            candmax_size = 0.0
            for k in eq:
                s = self.cand_sizes[k]
                if s > candmax_size:
                    candmax_size = s
            candmax_size = candmax_size / opt_m
            # Sanity check:
            if (candmax_size < self.candmax_size or
                    (candmax_size == self.candmax_size and
                     candmax_fit < self.candmax_fit)):
                ERROR('cand convergence problem!')
                ERROR('previous values:')
                ERROR('  cand_sizes:', self.cand_sizes_bak)
                ERROR('  sel:', self.sel)
                ERROR('  fit_sel:', self.fit_sel)
                ERROR('  eq:', self.eq)
                ERROR('current values:')
                ERROR('  cand_sizes:', self.cand_sizes)
                ERROR('  sel:', sel)
                ERROR('  fit_sel:', fit_sel)
                ERROR('  eq:', eq)
            # Store values
            self.fit_sel = fit_sel
            self.candmax_fit = candmax_fit
            self.candmax_size = candmax_size
            self.candmax_dist = candmax_dist
            self.cand_sizes_bak = odict(self.cand_sizes)
            self.sel = sel
            self.eq = eq
            # Prevent rounding to 1.0 for display purposes
            if 0.99 < candmax_dist < 1.0:
                candmax_dist = 0.99
            if 0.99 < candmax_size < 1.0:
                candmax_size = 0.99

        # print runtime values
        if self.candmax_fit == float('-inf'):
            candmax_fit_s = '     %s' % self.candmax_fit
        else:
            candmax_fit_s = '% .6f' % self.candmax_fit
        if self.first_time:
            self.time_delta = current_time - 1
            INFO(' time |   fitness | size | dist')
            SOLUTION('%5.1f' % 0.0, '%s |' % candmax_fit_s,
                     '%.2f |' % self.candmax_size,
                     '%.2f' % self.candmax_dist)
            self.first_time = False
        elif self.new_solution:
            # Store current solution in scenario
            sol = self.solution.values()
            if len(sol) != opt_m:
                ERROR('Solution not complete.')
            if not hasattr(self.sc, 'solution'):
                self.sc.solutions = odict()
            self.sc.solutions[current_time] = sol
            # Console output
            SOLUTION('%5.1f' % current_time, '%s |' % candmax_fit_s,
                     '%.2f |' % self.candmax_size,
                     '%.2f' % self.candmax_dist)
            self.new_solution = False
        else:
            STATS('%5.1f' % current_time, '%s |' % candmax_fit_s,
                  '%.2f |' % self.candmax_size,
                  '%.2f' % self.candmax_dist)

    def is_converged(self):
        if self.candmax_dist != 1.0:
            return False
        for a in self.agents.values():
            cand = a.kappa.solution_candidate.configuration
            if contribution(a).value != cand[a.aid].value:
                return False
        return True
