# coding=utf-8

from __future__ import division

import os
import csv
from datetime import datetime

import numpy as np

from definitions import *
from util import import_object, current_method, locdict, base10toN
from objectives import *


# [Han, 2009] eq. (13)-(16),(20)-(24)
#   use L(10*i-1, 10*i) (see table 2)
#   use S(10) or S(k+5) (see table 2)
def _weights(opt_m, opt_n, opt_q, inversed=False, dim_dependent=True):
    # Generate profits (opt_m * opt_n)
    p = np.empty((opt_m, opt_n))
    for i in range(opt_m):
        for j in range(opt_n):
            # table 2 --> eq. (16) --> eq. (13)
            p_i_min, p_i_max = 10 * i, 10 * (i + 1)
            p[i][j] = (j / (opt_n - 1)) * (p_i_max - p_i_min) + p_i_min

    # Repeat profits in q dimensions (makes the following calculations easier)
    p = p.repeat(opt_q).reshape(opt_m, opt_n, opt_q)

    # Find maximal profit in each class
    p_max = p.max(axis=1)
    # Shape to n*q dimensions (makes the following calculations easier)
    p_max = p_max.repeat(opt_n).reshape((opt_m, opt_n, opt_q))

    # Calculate delta-value for eq. (20)
    if not dim_dependent:
        # table 2 --> eq. (20)
        d = np.array([10] * opt_q)
    else:
        # table 2 --> eq. (24) --> eq. (20)
        d = np.arange(1, opt_q + 1) + 5

    # Minor error in [Han, 2009]. Reverse delta-values to correct.
    # d = d[::-1]

    # Derive weights as strongly correlated to profits
    if not inversed:
        # eq. (20)
        w = (p + (p_max / d))
    else:
        # eq. (22)
        w = (p_max - (p / d))

    # Return a copy of the array, because we need contigous data!
    return np.array(w.astype(int))


# [Han, 2009] eq. (26)
def _capacity(k, h, S, opt_m, w):
    # Filter k_th dimension in source array:
    # - Transpose, so that the dimension index is the most outer one
    # - Select k_th dimension
    # - Re-transpose, so that the array comprises m classes with n entries
    dim_k = w.T[k - 1].T

    # Sum minimal and maximal elements of each class, respectively.
    w_max_k = sum([np.max(dim_k[i]) for i in range(opt_m)])
    w_min_k = sum([np.min(dim_k[i]) for i in range(opt_m)])

    # Calculate capacity according to eq. (26)
    return int(((h / (S + 1)) * (w_max_k - w_min_k)) + w_min_k)


def _bounds(opt_w, opt_m, objective, zerobound=True):
    """
    Estimates the optimal and worst solution of the given scenario data.
    The zerobound parameter defines whether zero is assumed always. For the
    worst solution, an estimation based on the largest and smallest load
    profiles is performed.
    """
    w_max, w_min, w_max_i, w_min_i, w_max_r, w_min_r = (
            np.array([None for i in range(opt_m)]),
            np.array([None for i in range(opt_m)]),
            [None for i in range(opt_m)],
            [None for i in range(opt_m)],
            [None for i in range(opt_m)],
            [None for i in range(opt_m)])

    # Find minimal and maximal combination of elements
    for i in range(opt_w.shape[0]):
        for j in range(opt_w.shape[1]):
            r = objective(opt_w[i][j])
            Objective.calls -= 1
            if w_min_r[i] == None or r < w_min_r[i]:
                w_min[i] = opt_w[i][j]
                w_min_i[i] = j
                w_min_r[i] = r
            if w_max_r[i] == None or r > w_max_r[i]:
                w_max[i] = opt_w[i][j]
                w_max_i[i] = j
                w_max_r[i] = r

    # Assume 0 as optimal solution, select worst solution from (w_min,w_max)
    r_1, r_2 = objective(w_max.sum(0)), objective(w_min.sum(0))
    Objective.calls -= 2
    if zerobound:
        sol_d_max, sol_d_min = max(r_1, r_2), 0
    else:
        sol_d_max, sol_d_min = max(r_1, r_2), min(r_1, r_2)
    sol_j_max = w_max_i if r_1 >= r_2 else w_min_i

    return sol_d_max, sol_d_min, sol_j_max


def _bounds_bruteforce(h, n, m):
    d = np.load('../sc_data/bruteforce_h%03d.npy' % h)
    imin, imax = np.argmin(d), np.argmax(d)
    vmin, vmax = d.min(), d.max()
    conv = base10toN(imax, n)
    prefix = '0' * (m - len(conv))
    idx_max = [int(s) for s in prefix + conv]

    return vmax, vmin, idx_max


def _sol_avg(rnd, opt_w, opt_m, objective):
    lim = opt_w.shape[0] * opt_w.shape[1]
    d = np.empty((lim,))
    for k in range(lim):
        # Choose random j for each class
        idx = [rnd.randint(0, opt_w.shape[1] - 1) for j in range(opt_m)]
        # Select according weight in each class
        w = opt_w[np.arange(opt_m), idx]
        # Sum weights and calculate rating (result is a scalar value!)
        d[k] = objective(w.sum(0))
        Objective.calls -= 1

    # Return average values over 'lim' runs
    return np.mean(d)


# SC problem instance [Pisinger EJOR, 83, 1995],
# capacity generation from [Han, 2010]
def SC(rnd, seed,
       title=None,
       agent_module='agent',
       agent_type='Agent',
       agent_arp=False,
       network_module='networks',
       network_type='smallworld',
       network_c=3,
       network_k=1,
       network_phi=0.5,
       objective_module='objectives',
       objective_type='Objective_Manhattan',
       opt_m=10,                       # number of classes
       opt_n=5,                        # number of items in each class
       opt_q=5,                        # number of dimensions
       opt_s=100,                      # capacity divider (no. of instances)
       opt_h=5,                        # capacity selector (no. of instance)
       opt_sol_init_type='worst',      # initial solution: random or worst?
       opt_p_refuse_type='constant',   # p_refuse: constant (max), random?
       opt_p_refuse_min=0.0,           # p_refuse: minimal value
       opt_p_refuse_max=0.0,           # p_refuse: maximal value
       opt_p_refuse_dynamic=False,     # apply p_refuse in scenario (False) or agent (True)
       ):
    np.random.seed(seed)
    scenario_module, scenario_type = current_method()

    opt_w = np.ma.array(_weights(opt_m, opt_n, opt_q))
    opt_w_ranges = np.array([np.min(np.min(opt_w, 1), 1),
                             np.max(np.max(opt_w, 1), 1)]).T

    if opt_h == 'random':
        opt_h = int(round(rnd.uniform(1, opt_s)))

    objective = import_object(objective_module, objective_type)(
            np.array([_capacity(i, opt_h, opt_s, opt_m, opt_w)
                    for i in range(1, opt_q + 1)]))

    if opt_p_refuse_type == 'constant':
        opt_p_refuse = np.array([opt_p_refuse_max for i in range(opt_m)])
    elif opt_p_refuse_type == 'random':
        opt_p_refuse = np.array(
                [rnd.uniform(opt_p_refuse_min, opt_p_refuse_max)
                 for i in range(opt_m)])
    else:
        raise RuntimeError(opt_p_refuse_type)

    if opt_m == 10 and opt_n == 5 and opt_q == 5:
        sol_d_max, sol_d_min, sol_j_max = _bounds_bruteforce(opt_h, opt_n, opt_m)
    else:
        sol_d_max, sol_d_min, sol_j_max = _bounds(opt_w, opt_m, objective)

    if opt_sol_init_type == 'random':
        sol_init = opt_w[np.arange(opt_m),
                [rnd.randint(0, opt_n - 1) for i in range(opt_m)]]
        sol_d_avg = _sol_avg(rnd, opt_w, opt_m, objective)
    elif opt_sol_init_type == 'worst':
        sol_init = opt_w[np.arange(opt_m), sol_j_max]
        sol_d_avg = _sol_avg(rnd, opt_w, opt_m, objective)
    else:
        raise RuntimeError(opt_sol_init_type)

    if not opt_p_refuse_dynamic:
        # Generate a random number for each element, repeat to shape (m, n, q)
        choice = np.random.uniform(size=opt_w.shape[:-1]
                ).repeat(opt_w.shape[-1]
                ).reshape(opt_w.shape)
        # Repeat p_refuse to (m, n, q)
        p = opt_p_refuse.repeat(opt_w.shape[1] * opt_w.shape[2]
                ).reshape(opt_w.shape)
        # Mask elements according to p_refuse
        opt_w[choice < p] = np.ma.masked
        del p, choice
        # For each class that now has only masked elements,
        # unmask the first element
        opt_w.soften_mask()
        for i in range(opt_w.shape[0]):
            if opt_w.mask[i].all():
                opt_w[i][0].mask = np.ma.nomask
        # reset opt_p_refuse to prevent additional dynamic refusal
        opt_p_refuse_max = 0.0
        opt_p_refuse = np.array([opt_p_refuse_max for i in range(opt_m)])

    agent_ids = [i for i in range(opt_m)]
    network = import_object(network_module, network_type)(locals())

    del rnd, i
    return locdict(locals())


def _read_slp_2010(sc, bd):
    # Read csv data
    slp = []
    found = False
    with open(sc.slp_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if not row:
                continue
            if not found and row[0] == 'Datum':
                found = True
            elif found:
                date = datetime.strptime('_'.join(row[:2]), '%d.%m.%Y_%H:%M:%S')
                if date < sc.t_start:
                    continue
                elif date >= sc.t_end:
                    break
                # This is a demand, so negate the values
                slp.append(-1.0 * float(row[2].replace(',', '.')))
    slp = np.array(slp)
    # Scale values
    # if hasattr(sc, 'run_unctrl_datafile'):
    #     slp_norm = norm(slp.min(), slp.max(), slp)
    #     unctrl = np.load(os.path.join(bd, sc.run_unctrl_datafile)).sum(0) / 1000
    #     slp = slp_norm * (unctrl.max() - unctrl.min()) + unctrl.min()
    MS_day_mean = 13.6 * 1000 * 1000    # kWh, derived from SmartNord Scenario document
    MS_15_mean = MS_day_mean / 96
    slp = slp / np.abs(slp.mean()) * MS_15_mean

    return slp
    # return np.array(np.roll(slp, 224, axis=0))


def _bounds_spreadreduce_slp(opt_w, objective, zerobound=True, initialbound=True):
    assert zerobound
    assert type(objective) == Objective_Spreadreduce_SLP
    if initialbound:
        return objective(objective.sol_init), 0, None
    else:
        diffs = opt_w - objective.sol_init
        max_diff_spread = np.sum(np.max(diffs, 1) - np.min(diffs, 1), 0)
        return objective(max_diff_spread), 0, None


def APPSIM_ENUM(rnd, appsim_scenario, basedir,
         agent_module='agent',
         agent_type='Agent',
         agent_arp=False,
         network_module='networks',
         network_type='smallworld',
         network_c=3,
         network_k=1,
         network_phi=0.5,
         objective_module='objectives',
         objective_type='Objective_Manhattan',
         opt_sol_init_type='given',      # initial solution: random, worst or given?
         opt_p_refuse_type='constant',   # p_refuse: constant, random, sc?
         opt_p_refuse_min=0.0,           # p_refuse: minimal value
         opt_p_refuse_max=0.0,           # p_refuse: maximal value
         ):
    np.random.seed(appsim_scenario.seed)
    scenario_module, scenario_type = current_method()

    title = appsim_scenario.title
    opt_m = len(appsim_scenario.aids)
    opt_w_dict = dict(np.load(os.path.join(basedir, appsim_scenario.run_pre_samplesfile)))
    opt_w = np.array([opt_w_dict[aid] for aid in appsim_scenario.aids])
    opt_w_ranges = np.array([np.min(np.min(opt_w, 2), 1),
                             np.max(np.max(opt_w, 2), 1)]).T
    opt_q = opt_w.shape[-1]

    b_start = appsim_scenario.t_start
    b_end = appsim_scenario.t_block_end
    div = 1
    if (b_end - b_start).total_seconds() / 60 == opt_q * 15:
        div = 15
    b_s = (b_start - appsim_scenario.t_start).total_seconds() / 60 / div
    b_e = (b_end - appsim_scenario.t_start).total_seconds() / 60 / div

    if opt_p_refuse_type == 'constant':
        opt_p_refuse = [opt_p_refuse_max for i in range(opt_m)]
    elif opt_p_refuse_type == 'random':
        opt_p_refuse = [rnd.uniform(opt_p_refuse_min, opt_p_refuse_max)
                        for i in range(opt_m)]
    else:
        raise RuntimeError(opt_p_refuse_type)

    if opt_sol_init_type == 'random':
        sol_init_dict = {aid: opt_w_dict[rnd.randint(0, opt_n - 1)]
                         for aid in appsim_scenario.aids}
    elif opt_sol_init_type == 'given':
        unctrl = np.load(os.path.join(basedir,
                                      appsim_scenario.run_unctrl_datafile))
        sol_init_dict = {}
        for aid in appsim_scenario.aids:
            data = unctrl[aid][0,b_s:b_e]
            if data.shape[-1] == opt_q:
                sol_init_dict[aid] = data
            else:
                raise RuntimeError('cannot match shape of sol_init and target')
            del data
        del unctrl
    else:
        raise RuntimeError(opt_sol_init_type)

    if appsim_scenario.objective == 'epex':
        block = np.ma.array(appsim_scenario.block)
        if block.shape == (1,):
            block = block.repeat(opt_q)
        elif block.shape[0] == opt_q / 15:
            block = block.repeat(15)
        block_start = (appsim_scenario.t_block_start -
                       appsim_scenario.t_start).total_seconds() / 60 / div
        block[:block_start] = np.ma.masked
        objective = Objective_Manhattan(block)
        sol_d_max, sol_d_min, _ = _bounds(opt_w, opt_m, objective,
                                        zerobound=True)
    elif appsim_scenario.objective == 'peakshaving':
        objective = Objective_Peakshaving((opt_q,))
        sol_d_max, sol_d_min, _ = _bounds_peakshaving(opt_w, objective)
    elif appsim_scenario.objective == 'valleyfilling':
        objective = Objective_Valleyfilling((opt_q,))
        sol_d_max, sol_d_min, _ = _bounds_valleyfilling(opt_w, objective)
    elif appsim_scenario.objective == 'spreadreduce':
        objective = Objective_Spreadreduce((opt_q,))
        sol_d_max, sol_d_min, _ = _bounds_spreadreduce(opt_w, objective)
    elif appsim_scenario.objective == 'spreadreduce-slp':
        slp = _read_slp_2010(appsim_scenario, basedir)[b_s:b_e]
        objective = Objective_Spreadreduce_SLP(slp,
                np.array(sol_init_dict.values()).sum(0))
        sol_d_max, sol_d_min, _ = _bounds_spreadreduce_slp(opt_w, objective)
    else:
        raise RuntimeError(appsim_scenario.objective)

    sol_d_avg = _sol_avg(rnd, opt_w, opt_m, objective)

    # agent_ids = [i for i in range(opt_m)]
    agent_ids = appsim_scenario.aids
    network = import_object(network_module, network_type)(locals())

    # del rnd
    del i, b_start, b_end, b_s, b_e, appsim_scenario
    ld = locdict(locals())
    del ld['rnd']
    return ld


def ids(m, n):
    l = np.zeros(m, dtype=np.int32)
    yield l
    for i in xrange(n**m - 1):
        for j in xrange(m - 1, -1, -1):
            if l[j] == n - 1:
                l[j] = 0
            else:
                l[j] += 1
                break
        yield l


def bruteforce(sc, progress=True, thres=None):
    # print
    # print 'brute force'

    obj = sc[KW_OBJECTIVE]
    m, n, w = sc[KW_OPT_M], sc[KW_OPT_N], sc[KW_OPT_W]

    import util, time
    tm = time.time()
    counter, min_counter, max_counter = 0, 0, 0
    s_min, s_max = None, None
    r_min, r_max = None, None
    x = np.arange(m)

    if thres is None:
        SIZE = n**m
    else:
        SIZE = thres
    # selections = np.empty((SIZE,m))
    results = np.empty((SIZE,))

    if progress:
        progress = util.PBar(SIZE).start()
    for i in ids(m, n):
        s = w[x, i]
        r = obj(s)
        # selections[counter] = i
        results[counter] = r
        # if r_min is None or r < r_min:
        #     s_min = s
        #     r_min = r
        #     min_counter = counter
        #     min_ids = list(i)
        # if r_max is None or r > r_max:
        #     s_max = s
        #     r_max = r
        #     max_counter = counter
        #     max_ids = list(i)
        counter += 1
        if counter >= SIZE:
            break
        if progress:
            progress.update(counter)
    if progress:
        progress.finish()
    tm = time.time() - tm

    print '%d solutions calculated in %f seconds' % (counter, tm)
    # print
    # print 'Optimal Solution'
    # print '----------------------------------'
    # print '  solution number:', min_counter
    # print '  residual error:', r_min
    # print '  solution ids:', min_ids
    # print '  solution:', s_min
    # print
    # print 'Worst Solution'
    # print '----------------------------------'
    # print '  solution number:', max_counter
    # print '  residual error:', r_max
    # print '  solution ids:', max_ids
    # print '  solution:', s_max

    return results


if __name__ == '__main__':
    import random
    seed = 0
    sc = SC(random.Random(seed), seed)

    print sc
    # print
    # target = sc[KW_OBJECTIVE].target
    # if isinstance(target, np.ma.MaskedArray):
    #     mask = np.array([not b for b in sc[KW_OBJECTIVE].target.mask])
    # else:
    #     mask = np.array([True] * sc[KW_OPT_Q])
    # print sc[KW_OPT_W][:,:,mask]
    # print sc[KW_OPT_W][:,:,mask].sum(2)

    # import pdb
    # pdb.set_trace()

    # for h in range(1, 101):
    #     c = [_capacity(k, h, sc[KW_OPT_S], sc[KW_OPT_M], sc[KW_OPT_W])
    #          for k in range(1, sc[KW_OPT_Q] + 1)]
    #     print 'h%03d = %s' % (h, c)

    # import jpype as jp
    # jp.shutdownJVM()

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # data = sc[KW_OPT_W].mean(1).T
    # data = sc[KW_OPT_W].T.reshape((sc[KW_OPT_Q], sc[KW_OPT_N] * sc[KW_OPT_M]))
    # print data.shape
    # ax.plot(data)
    # plt.show()


    # d = bruteforce(sc, thres=10000)
    # d *= (1.0 / d.max())

    # BINS = 100
    # hist, bins = np.histogram(d, bins=BINS, density=True)
    # hist /= BINS

    # import matplotlib.pyplot as plt

    # plt.bar(bins[:-1], hist, width=1/BINS)
    # # labels
    # plt.xlabel('Error [0, 1]')
    # plt.ylabel('Probability')
    # plt.title('Histogram of instance G-C(L)-D(S), h=(%d of %d)' % (1, 100))

    # plt.show()
