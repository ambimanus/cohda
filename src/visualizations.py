# coding=utf-8

from __future__ import division

from collections import OrderedDict as odict

import numpy as np

from logger import *
import util


class Stats(object):
    def __init__(self, scenario, agents):
        self.sc = scenario
        self.agents = agents
        self.solution = {aid: agents[aid].sol for aid in agents}
        self.sol_counters = {aid: agents[aid].sol_counter for aid in agents}
        self.bkc_sizes = {}
        self.bkc_ratings = {}
        self.first_time = True
        self.new_solution = False
        self.distance = 1.0
        self.bkcmin = 1.0
        self.bkcmin_dist = 0.0
        self.bkcmin_size = 0.0
        self.sel = None
        self.eq = None
        self.bkc_sel = None
        self.time_delta = 0.0


    def eval(self, current_time):
        opt_m, opt_q = self.sc[KW_OPT_M], self.sc[KW_OPT_Q]
        d_min, d_max = self.sc[KW_SOL_D_MIN], self.sc[KW_SOL_D_MAX]
        obj = self.sc[KW_OBJECTIVE]

        # Collect agent states
        for aid in sorted(self.agents.keys()):
            a = self.agents[aid]
            if self.sol_counters[aid] != a.sol_counter:
                self.sol_counters[aid] = a.sol_counter
                self.solution[aid] = a.sol
                self.new_solution = True
            if a.bkc is not None:
                self.bkc_sizes[aid] = len(a.bkc)
            if a.bkc_f is not None:
                self.bkc_ratings[aid] = a.bkc_f

        # Set current timestamp relative to beginning of the heuristic
        current_time -= self.time_delta

        # sel := keys of bkc values that should be considered.
        # At the beginning of the simulation, report all bkc values.
        # But as soon as the first complete one has be found, restrict the
        # report to only complete bkc values.
        # In either case, consider only the set of the largest bkcs available.
        sel = []
        bm = max(self.bkc_sizes.values())
        for k in sorted(self.bkc_sizes.keys()):
            if self.bkc_sizes[k] == bm:
                sel.append(k)
        if len(self.bkc_ratings) > 0 and len(sel) > 0:
            bkc_sel = [self.bkc_ratings[k] for k in sel]
            # bkcmin := minimal found bkc value
            bkcmin = min(bkc_sel)
            # eq := keys of bkc_sel whith minimal value
            eq = []
            for k in sorted(self.bkc_ratings.keys()):
                if abs(self.bkc_ratings[k] - bkcmin) < 0.00001:
                    eq.append(k)
            bkcmin = util.norm(d_min, d_max, bkcmin)
            # bkcmin_dist := distribution of minimal bkc in population
            bkcmin_dist = len(eq) / opt_m
            # bkcmin_size := completeness of minimal bkc with respect to opt_m
            bkcmin_size = None
            for k in eq:
                s = self.bkc_sizes[k]
                if bkcmin_size is None or s > bkcmin_size:
                    bkcmin_size = s
            bkcmin_size = bkcmin_size / opt_m
            # Sanity check:
            if (bkcmin_size < self.bkcmin_size or
                    (bkcmin_size == self.bkcmin_size and
                     bkcmin > self.bkcmin)):
                ERROR('bkc convergence problem!')
                ERROR('previous values:')
                ERROR('  bkc_sizes:', self.bkc_sizes_bak)
                ERROR('  sel:', self.sel)
                ERROR('  bkc_sel:', self.bkc_sel)
                ERROR('  eq:', self.eq)
                ERROR('current values:')
                ERROR('  bkc_sizes:', self.bkc_sizes)
                ERROR('  sel:', sel)
                ERROR('  bkc_sel:', bkc_sel)
                ERROR('  eq:', eq)
            # Store values
            self.bkc_sel = bkc_sel
            self.bkcmin = bkcmin
            self.bkcmin_size = bkcmin_size
            self.bkcmin_dist = bkcmin_dist
            self.bkc_sizes_bak = odict(self.bkc_sizes)
            self.sel = sel
            self.eq = eq
            # Prevent rounding to 1.0 for display purposes
            if 0.99 < bkcmin_dist < 1.0:
                bkcmin_dist = 0.99
            if 0.99 < bkcmin_size < 1.0:
                bkcmin_size = 0.99

        # print runtime values
        if self.first_time:
            self.time_delta = current_time - 1
            INFO(' time |  distance | bkc-value | bkc-size | bkc-dist')
            if KW_SOL_INIT_DICT in self.sc:
                sol = np.array(self.sc[KW_SOL_INIT_DICT].values())
            else:
                sol = self.sc[KW_SOL_INIT].reshape((opt_m, opt_q))
            SOLUTION('%5.1f' % 0.0,
                     '% .6f |' % util.norm(d_min, d_max,
                                           obj(sol, record_call=False)),
                     ' %.6f |' % self.bkcmin, '  %.2f   |' % self.bkcmin_size,
                     '  %.2f   |' % self.bkcmin_dist)
            self.first_time = False
        elif self.new_solution:
            # Store current solution in scenario
            sol = np.array(self.solution.values())
            if len(sol) != opt_m:
                ERROR('Solution not complete.')
            if not KW_SOL in self.sc:
                self.sc[KW_SOL] = odict()
            self.sc[KW_SOL][current_time] = sol
            # Console output
            self.distance = util.norm(d_min, d_max,
                                      obj(sol, record_call=False))
            SOLUTION('%5.1f' % current_time, '% .6f |' % self.distance,
                     ' %.6f |' % self.bkcmin, '  %.2f   |' % self.bkcmin_size,
                     '  %.2f   |' % self.bkcmin_dist)
            self.new_solution = False
        else:
            STATS('%5.1f' % current_time, 'None      |',
                  ' %.6f |' % self.bkcmin, '  %.2f   |' % self.bkcmin_size,
                  '  %.2f   |' % self.bkcmin_dist)


    def is_converged(self):
        if self.bkcmin_dist != 1.0:
            return False
        for a in self.agents.values():
            if any(a.sol != a.bkc[a.aid]):
                return False
        return True


    def eval_final(self):
        if not self.is_converged():
            ERROR('convergence not reached!')
        INFO('Target', self.sc[KW_OBJECTIVE].target)
        INFO('Result', self.sc[KW_SOL].values()[-1].sum(0))
