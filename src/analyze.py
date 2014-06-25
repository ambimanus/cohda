# coding=utf-8

from __future__ import division

import sys
import os
import pickle
from collections import OrderedDict     # used in eval()

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator, FixedLocator
from matplotlib.patches import Polygon

from util import norm
from definitions import *


KW_OPT_C = 'opt_c'

threshold = None
bkc_thres = None
# bkc_thres = 0.01
file_thres = None
# file_thres = 88

MISSING_RESULTS_SKIP = False

# http://www.javascripter.net/faq/hextorgb.htm
PRIMA = (148/256, 164/256, 182/256)
PRIMB = (101/256, 129/256, 164/256)
PRIM  = ( 31/256,  74/256, 125/256)
PRIMC = ( 41/256,  65/256,  94/256)
PRIMD = ( 10/256,  42/256,  81/256)
EC = (1, 1, 1, 0)
GRAY = (0.5, 0.5, 0.5)
WHITE = (1, 1, 1)


def read_solutions(file, opt_m):
    msgs, pending, objcalls = [], [], []
    msgs_diff, pending_diff, objcalls_diff = 0, 0, 0
    sol_d, sol_p, sol_u = [], [], []
    sol_bkc, sol_bkc_size, sol_bkc_dist = [], [], []
    result_indices = None
    result_ok = False
    with open(file) as f:
        h = ['messages', 'objcalls']
        for line in f:
            if ((threshold is not None and len(sol_d) >= threshold) or
                (bkc_thres is not None and len(sol_bkc) > 0 and
                    sol_bkc[-1] <= bkc_thres)):
                # for l in (msgs, pending, objcalls, sol_d, sol_p, sol_u, sol_bkc, sol_bkc_size, sol_bkc_dist):
                #     if len(l) > 0:
                #         del l[-1]
                break
            try:
                l = line.split('] ')
                if len(l) < 2:
                    continue
                key, data = l[0], l[1]
                if 'ERROR' in key:
                    # print line
                    pass
                elif 'INFO' in key and 'time' in data:
                    d = [s.strip() for s in data.split('|')]
                    time_idx = d.index('time')
                    h = h[:time_idx]
                    if 'messages' in h:
                        msgs_diff = int(d[h.index('messages')])
                    if 'pending' in h:
                        pending_diff = int(d[h.index('pending')])
                    if 'objcalls' in h:
                        objcalls_diff = int(d[h.index('objcalls')])
                    h += [n.strip() for n in d[time_idx:]]
                elif 'STATS' in key or 'SOLUTION' in key:
                    d = data.split('|')
                    if 'messages' in h:
                        msgs.append(int(d[h.index('messages')]) - msgs_diff)
                    if 'pending' in h:
                        pending.append(int(d[h.index('pending')]) - pending_diff)
                    if 'objcalls' in h:
                        objcalls.append(int(d[h.index('objcalls')]) - objcalls_diff)
                    if 'distance' in h:
                        if 'None' in d[h.index('distance')]:
                            if len(sol_d) > 0:
                                sol_d.append(sol_d[-1])
                            if len(sol_p) > 0:
                                sol_p.append(sol_p[-1])
                            if len(sol_u) > 0:
                                sol_u.append(sol_u[-1])
                        else:
                            if 'distance' in h:
                                sol_d.append(float(d[h.index('distance')]))
                            if 'penalty' in h:
                                sol_p.append(float(d[h.index('penalty')]))
                            if 'utility' in h:
                                sol_u.append(float(d[h.index('utility')]))
                        if 'bkc-value' in h:
                            sol_bkc.append(float(d[h.index('bkc-value')]))
                        if 'bkc-size' in h:
                            sol_bkc_size.append(float(d[h.index('bkc-size')]))
                        if 'bkc-dist' in h:
                            sol_bkc_dist.append(float(d[h.index('bkc-dist')]))
                elif 'INFO' in key and 'Result' in data:
                    d = data.split('|')
                    for i in range(len(d)):
                        if 'Result' in d[i]:
                            r = d[i + 1].strip()
                            if r[:12] == 'OrderedDict(' and r[-1:] == ')':
                                result_indices = eval(r)
                                result_ok = True
                            elif r[0] == '[':
                                # r.append(']')
                                # result = eval(r)
                                result_ok = True
                            else:
                                raise AssertionError, '"%s"' %r
            except:
                print 'Parse error in line', line
                raise
    if len(sol_d) == 1:
        for l in (msgs, sol_d, sol_p, sol_u, sol_bkc, sol_bkc_size,
                  sol_bkc_dist):
            if len(l) > 0:
                l.append(l[0])

    if bkc_thres is None and not result_ok:
        print 'Warning:', file, 'has no final result!'
        if MISSING_RESULTS_SKIP:
            return None

    if sol_bkc[-1] > 0.1:
        print file, 'Warning: bad fitness', sol_bkc[-1]

    return (ma.asanyarray(msgs, int),
            ma.asanyarray(pending, int),
            ma.asanyarray(objcalls, int),
            ma.asanyarray(sol_d, np.float64),
            ma.asanyarray(sol_p, np.float64),
            ma.asanyarray(sol_u, np.float64),
            ma.asanyarray(sol_bkc, np.float64),
            ma.asanyarray(sol_bkc_size, np.float32),
            ma.asanyarray(sol_bkc_dist, np.float32),
            result_indices)


def analyze(dir):
    if dir[-1] == '/':
        dir = dir[:-1]

    if os.path.isdir(dir):
        files = sorted([f for f in os.listdir(dir) if '.log' in f])
    else:
        dir, file = os.path.split(dir)
        files = [file]
    if file_thres is not None:
        files = files[:file_thres]
    cfgs = [os.path.join(dir, '.'.join(('cfg', f.split('.')[0], 'pickle')))
            for f in files]

    sizes = np.zeros(len(files), int)
    msgs, pendings, objcalls = [], [], []
    sol_d, sol_p, sol_u = [], [], []
    sol_bkc, sol_bkc_size, sol_bkc_dist = [], [], []
    targets, results = [], []
    scs = []

    bound_d, bound_p = np.empty(len(cfgs)), np.empty(len(cfgs))
    bound_u = np.empty(len(cfgs))

    opt_w = None
    opt_w_params = None
    for i in range(len(files)):
        with open(cfgs[i]) as c:
            # get scenario data
            cfg = pickle.load(c)
            sc = cfg.scenario
            if KW_OPT_C in sc:
                target = sc[KW_OPT_C]
            elif KW_OBJECTIVE in sc:
                target = sc[KW_OBJECTIVE].target
            else:
                raise RuntimeError('no target!')
            # if len(targets) == 0 or any(targets[-1] != target):
            #     targets.append(target)
            targets.append(target)
            if KW_OPT_W in sc:
                opt_w = sc[KW_OPT_W]
                del sc[KW_OPT_W]
            elif (((KW_OPT_W_PICKLE in sc and not sc[KW_OPT_W_PICKLE]) or
                   ('opt_w_generated' in sc and not sc['opt_w_generated'])) and
                  KW_OPT_W_PATH in sc and
                  'opt_w_scale_max' in sc):
                par = (sc[KW_OPT_W_PATH], sc[KW_OPT_M], sc[KW_OPT_N],
                       sc[KW_OPT_Q], sc['opt_w_scale_max'])
                if opt_w is None or par != opt_w_params:
                    opt_w_params = par
                    opt_w = _read_chp(*opt_w_params)
                    # FIXME
                    # raise NotImplementedError()

            # read data from simulation log
            fp = os.path.join(dir, files[i])
            data = read_solutions(fp, sc[KW_OPT_M])
            if data is None:
                print 'Skipping', cfgs[i]
                continue
            m, pend, calls, d, p, u, bv, bs, bd, ri = data

            # if ri is None:
            #     print 'Skipping', cfgs[i]
            #     continue

            lengths = [len(l) for l in (m, pend, calls, d, p, u, bv, bs, bd)]
            ls = set(lengths)
            ls.remove(0)
            if len(ls) > 1:
                print cfgs[i], 'Warning: lengths differ:', lengths
            sizes[i] = len(m)

            # Extract selected opt_w value for each agent from indices in ri
            if (ri is not None and
                    len(ri) > 0 and
                    opt_w is not None and
                    len(opt_w) > 0):
                r = np.zeros((sc[KW_OPT_M], sc[KW_OPT_Q]))
                for j in range(sc[KW_OPT_M]):
                    a = sc[KW_AGENT_IDS][j]
                    r[j] = opt_w[j, ri[a]]
                results.append(r)
            elif KW_SOL in sc:
                key = max(sc[KW_SOL].keys())
                results.append(sc[KW_SOL][key])
            elif KW_SOL_INIT in sc:
                results.append(sc[KW_SOL_INIT])

            # Copy all values into running lists
            for l, v in zip((msgs, pendings, objcalls, sol_d, sol_p, sol_u,
                             sol_bkc, sol_bkc_size, sol_bkc_dist, scs),
                            (m, pend, calls, d, p, u, bv, bs, bd, sc)):
                l.append(v)

            # calculate reference bounds
            d_min, d_max = sc[KW_SOL_D_MIN], sc[KW_SOL_D_MAX]
            d_avg = sc[KW_SOL_D_AVG]
            bound_d[i] = norm(d_min, d_max, d_avg)
            if 'KW_SOL_P_MIN' in locals():
                p_min, p_max = sc[KW_SOL_P_MIN], sc[KW_SOL_P_MAX]
                p_avg = sc[KW_SOL_P_AVG]
                bound_p[i] = norm(p_min, p_max, p_avg)
            if 'KW_SOL_U_MIN' in locals():
                u_min, u_max = sc[KW_SOL_U_MIN], sc[KW_SOL_U_MAX]
                u_avg = sc[KW_SOL_U_AVG]
                bound_u[i] = norm(u_min, u_max, u_avg)

    dim = sizes.max()
    dmean = bound_d.mean()
    dmean_data = np.array([dmean for i in range(dim)])
    pmean = bound_p.mean()
    pmean_data = np.array([pmean for i in range(dim)])
    umean = bound_u.mean()
    umean_data = np.array([umean for i in range(dim)])

    # create ndarrays with sufficient dimensions
    sh = (len(files), dim)
    y_msgs = np.zeros(sh, int)
    y_pendings = np.zeros(sh, int)
    y_objcalls = np.zeros(sh, int)
    y_sol_d = np.zeros(sh, np.float64)
    y_sol_p = np.zeros(sh, np.float64)
    y_sol_u = np.zeros(sh, np.float64)
    y_sol_bv = np.zeros(sh, np.float64)
    y_sol_bs = np.zeros(sh, np.float32)
    y_sol_bd = np.zeros(sh, np.float32)
    mask_m = np.zeros(sh, np.int)
    mask_pend = np.zeros(sh, np.int)
    mask_calls = np.zeros(sh, np.int)
    mask_d = np.zeros(sh, np.int)
    mask_p = np.zeros(sh, np.int)
    mask_u = np.zeros(sh, np.int)
    mask_bv = np.zeros(sh, np.int)
    mask_bs = np.zeros(sh, np.int)
    mask_bd = np.zeros(sh, np.int)
    # fill sets and create mask where len(data) < dim
    for i in range(len(sol_d)):
        for src, dst in zip((msgs, pendings, objcalls, sol_d, sol_p, sol_u,
                             sol_bkc, sol_bkc_size, sol_bkc_dist),
                            (y_msgs, y_pendings, y_objcalls, y_sol_d, y_sol_p,
                             y_sol_u, y_sol_bv, y_sol_bs, y_sol_bd)):
            l = len(src[i])
            dst[i, :l] = src[i] if len(src[i]) else None
        for y, mask in zip((msgs, pendings, objcalls, sol_d, sol_p, sol_u,
                            sol_bkc, sol_bkc_size, sol_bkc_dist),
                           (mask_m, mask_pend, mask_calls, mask_d, mask_p,
                            mask_u, mask_bv, mask_bs, mask_bv)):
            mask[i][l:] = [1 for v in range(dim-l)]
            mask[i][:l] |= y[i].mask
    # free some memory
    del msgs, pendings, objcalls, sol_d, sol_p, sol_u, sol_bkc, sol_bkc_size, sol_bkc_dist
    # convert to masked array
    y_msgs = ma.array(y_msgs, mask=mask_m)
    y_pendings = ma.array(y_pendings, mask=mask_pend)
    y_objcalls = ma.array(y_objcalls, mask=mask_calls)
    y_sol_d = ma.array(y_sol_d, mask=mask_d)
    y_sol_p = ma.array(y_sol_p, mask=mask_p)
    y_sol_u = ma.array(y_sol_u, mask=mask_u)
    y_sol_bv = ma.array(y_sol_bv, mask=mask_bv)
    y_sol_bs = ma.array(y_sol_bs, mask=mask_bs)
    y_sol_bd = ma.array(y_sol_bd, mask=mask_bd)

    return (dir, sizes, dmean, dmean_data, pmean, pmean_data, umean,
            umean_data, y_sol_d, y_sol_p, y_sol_u, y_sol_bv, y_sol_bs,
            y_sol_bd, y_msgs, y_pendings, y_objcalls,
            np.array(targets, np.float64), np.array(results, np.float64), scs)


# def stats(dir, sizes, dmean, dmean_data, pmean, pmean_data, umean, umean_data,
#           y_sol_d, y_sol_p, y_sol_u, y_sol_bv, y_sol_bs, y_sol_bd, y_msgs,
#           targets, results):

#     # Mittelwerte + Standardabweichung:
#     #  Laufzeit
#     #  Finale Fitness (normierte Distanz)
#     #  Finale Distanz
#     #  Finale Nachrichtenzahl
#     print os.path.basename(dir), ',', len(sizes), 'simulation runs'
#     print '\tmean, std'
#     print 'sim steps:', sizes.mean(), sizes.std()
#     d_final = np.array([d[~d.mask][-1] for d in y_sol_d])
#     print 'fitness:', d_final.mean(), d_final.std()
#     bv_final = np.array([d[~d.mask][-1] for d in y_sol_bv])
#     print 'fitness (bkc):', bv_final.mean(), bv_final.std()
#     p_final = np.array([p[~p.mask][-1] for p in y_sol_p])
#     print 'penalty:', p_final.mean(), p_final.std()
#     if len(results.shape) > 1:
#         diff_final = np.abs((targets - results.sum(1))).sum(1)
#     else:
#         diff_final = np.array([0.0], np.float64)
#     print 'difference:', diff_final.mean(), diff_final.std()
#     targets_sum = targets.sum(1)
#     print 'target (sum):', targets_sum.mean(), targets_sum.std()
#     print 'difference (%):', diff_final.mean() / targets_sum.mean(), diff_final.std() / targets_sum.mean()
#     m_final = np.array([m[~m.mask][-1] for m in y_msgs])
#     # m_final = np.array([m[~m.mask][-1] / s for m, s in zip(y_msgs, sizes)])
#     print 'messages:', m_final.mean(), m_final.std()
#     print 'messages / sim step:', (m_final / sizes).mean(), (m_final / sizes).std()
#     print

#     return (sizes.mean(), sizes.std(), d_final.mean(), d_final.std(),
#             bv_final.mean(), bv_final.std(), diff_final.mean(),
#             diff_final.std(), m_final.mean(), m_final.std(),
#             (m_final / sizes).mean(), (m_final / sizes).std())


def stats_raw(dir, sizes, dmean, dmean_data, pmean, pmean_data, umean,
              umean_data, y_sol_d, y_sol_p, y_sol_u, y_sol_bv, y_sol_bs,
              y_sol_bd, y_msgs, y_pendings, y_objcalls, targets, results, scs):

    opt_m = np.array([sc[KW_OPT_M] for sc in scs])
    opt_n = np.array([sc[KW_OPT_N] for sc in scs])
    opt_q = np.array([sc[KW_OPT_Q] for sc in scs])

    print os.path.basename(dir), ',', len(sizes), 'simulation runs'
    print '\tmean, std'
    print 'sim steps:', sizes.mean(), sizes.std()
    d_final = np.array([d[~d.mask][-1] for d in y_sol_d])
    print 'fitness:', d_final.mean(), d_final.std()
    bv_final = np.array([d[~d.mask][-1] for d in y_sol_bv])
    print 'fitness (bkc):', bv_final.mean(), bv_final.std()
    p_final = np.array([p[~p.mask][-1] for p in y_sol_p])
    print 'penalty:', p_final.mean(), p_final.std()
    if len(results.shape) > 1:
        diff_final = np.abs((targets - results.sum(1))).sum(1)
    else:
        diff_final = np.array([0.0 for x in range(len(sizes))], np.float64)
    print 'difference:', diff_final.mean(), diff_final.std()
    targets_sum = targets.sum(1)
    print 'target (sum):', targets_sum.mean(), targets_sum.std()
    print 'difference (%):', diff_final.mean() / targets_sum.mean(), diff_final.std() / targets_sum.mean()
    m_final = np.array([m[~m.mask][-1] for m in y_msgs])
    # m_final = np.array([m[~m.mask][-1] / s for m, s in zip(y_msgs, sizes)])
    print 'messages/agent/simstep:', (m_final / sizes / opt_m).mean(), (m_final / sizes / opt_m).std()
    print 'messages/agent:', (m_final / opt_m).mean(), (m_final / opt_m).std()
    print 'messages:', m_final.mean(), m_final.std()
    pendings_final = np.array([d[~d.mask][-1] for d in y_pendings])
    print 'pending messages:', pendings_final.mean(), pendings_final.std()
    objcalls_final = np.array([d[~d.mask][-1] for d in y_objcalls])
    print 'objective calls:', objcalls_final.mean(), objcalls_final.std()
    print

    return sizes, d_final, bv_final, diff_final, m_final, pendings_final, objcalls_final, opt_m, opt_n, opt_q


def err(v, std):
    std_minus = np.array(std)
    # Limit negative std to zero
    std_minus[v - std < 0.0] = v[v - std < 0.0]
    return np.array([std_minus, std])


def plot(ax, xl, yl, x, y, err, logx=False, xticknames=None, xlabelpad=0):
    if xl is not None:
        ax.set_xlabel(xl, labelpad=xlabelpad)
    if yl is not None:
        ax.set_ylabel(yl)
    ax.errorbar(x, y, fmt='o-', yerr=err)
    ax.set_xlim(x[0]-0.1, x[-1]+0.1)
    ax.grid(True, linestyle='-', which='major', color='lightgrey',
            alpha=0.5)
    ax.grid(True, linestyle='-', which='minor', color='lightgrey',
            alpha=0.5, axis='x')
    if logx:
        ax.set_xscale('log')
    if xticknames is not None:
        # xn = ax.set_xticklabels(xticknames)
        ax.xaxis.set_major_locator(IndexLocator(1, 0))
        # xn = plt.setp(ax, xticklabels=xticknames)
        # plt.setp(xn, rotation=90, fontsize=8)
    if len(err.shape) > 1:
        ax.set_ylim((y-err[0]).min()-(0.1*y.min()),
                    (y+err[1]).max()+(0.1*y.max()))
    else:
        ax.set_ylim((y-err).min()-(0.1*y.min()),
                    (y+err).max()+(0.1*y.max()))


def boxplot(ax, data, names, xl, yl, vert=True, rotate=False,
            xlabelpad=None, ylabelpad=None, xlim=None, ylim=None,
            xloc=None, xpos=None, logy=False):
    bp = ax.boxplot(data.T, vert=vert, positions=xpos)
    # bp = ax.boxplot(data)
    plt.setp(bp['boxes'], color='#1F4A7D', lw=0.5)
    plt.setp(bp['medians'], color='#1F4A7D', lw=1.0)
    plt.setp(bp['whiskers'], color='#1F4A7D', lw=0.5)
    plt.setp(bp['caps'], color='#1F4A7D', lw=0.5)
    plt.setp(bp['fliers'], color='#1F4A7D', marker='+', markersize=3.0, markeredgewidth=0.5)
    # 348ABD : blue
    # 7A68A6 : purple
    # A60628 : red
    # 467821 : green
    # CF4457 : pink
    # 188487 : turquoise
    # E24A33 : orange
    #
    # Fill the boxes
    print yl
    for name, box, median in zip(names, bp['boxes'], bp['medians']):
        IQR = max(box.get_ydata()) - min(box.get_ydata())
        print '\t%s: median=%.2f, IQR=%.2f' % (name, median.get_ydata()[0], IQR)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX,boxY)
        boxPolygon = Polygon(boxCoords, facecolor='#1F4A7D', alpha=0.25)
        ax.add_patch(boxPolygon)
        # Now draw the median lines back over what we just filled in
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(median.get_xdata()[j])
            medianY.append(median.get_ydata()[j])
            ax.plot(medianX, medianY, '#1F4A7D')
    #
    # if vert:
    #     ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
    #                   alpha=0.5)
    # else:
    #     ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
    #                   alpha=0.5)
    # Hide these grid behind plot objects
    # ax.set_axisbelow(True)
    ax.set_xlabel(xl, labelpad=xlabelpad)
    if logy:
        ax.set_yscale('log')
    if xpos is None:
        xpos = np.arange(1, int(len(names)) + 1)
    # have to loop over data to build averages due to possibly different data shapes
    assert len(data.shape) == 1 or len(data.shape) == 2
    if len(data.shape) == 1:
        averages = [np.average(data)]
    else:
        averages = np.zeros((len(data)))
        for i in range(data.shape[0]):
            averages[i] = np.average(data[i])
    if vert:
        # Overplot the sample averages
        ax.plot(xpos, averages, linestyle='', color='#A60628', marker='*', markeredgewidth=0.0)
        # Set the axes ranges and axes labels
        ax.set_xlim(0.5, xpos[-1] + 0.5)
        xticklabels = plt.setp(ax, xticklabels=names)
        if xloc is not None:
            ax.xaxis.set_major_locator(xloc)
        if rotate:
            plt.setp(xticklabels, fontsize=8, rotation=90)
    else:
        ax.plot(averages, xpos, linestyle='', color='#A60628', marker='*', markeredgewidth=0.0)
        ax.set_ylim(0.5, xpos[-1] + 0.5)
        yticklabels = plt.setp(ax, yticklabels=names)
        if xloc is not None:
            ax.yaxis.set_major_locator(xloc)
        if rotate:
            plt.setp(yticklabels, fontsize=8, rotation=90)
    ax.set_ylabel(yl, labelpad=ylabelpad)
    # Set the axes ranges and axes labels
    if xlim is not None:
        if len(xlim) == 1:
            ax.set_xlim(left=xlim[0])
        else:
            ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        if len(ylim) == 1:
            ax.set_ylim(bottom=ylim[0])
        else:
            ax.set_ylim(ylim[0], ylim[1])


def qplot(ax, data, names, xl, yl, vert=True):
    bp = ax.boxplot(data.T, vert=vert, sym='')
    for artist in bp['whiskers'] + bp['caps']:
        artist.remove()
    del bp['whiskers'], bp['caps']
    # import pdb
    # pdb.set_trace()
    #
    # bp = ax.boxplot(data)
    # plt.setp(bp['boxes'], color='black')
    # plt.setp(bp['whiskers'], color='black')
    # plt.setp(bp['fliers'], color='red', marker='+')
    if vert:
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)
    else:
        ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)
    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    n = np.arange(int(len(names)))
    medians = np.median(data, 1)
    if vert:
        # Overplot the sample averages
        # ax.plot(n+1, averages, color='w', marker='*', markeredgecolor='k')
        # Connect the medians with a line
        ax.plot(n+1, medians, color='k', linestyle='-')
        # Set the axes ranges and axes labels
        ax.set_xlim(0.5, n[-1]+1.5)
        tickNames = plt.setp(ax, xticklabels=names)
    else:
        # ax.plot(averages, n+1, color='w', marker='*', markeredgecolor='k')
        ax.plot(medians, n+1, color='k', linestyle='-')
        ax.set_ylim(0.5, n[-1]+1.5)
        tickNames = plt.setp(ax, yticklabels=names)
    plt.setp(tickNames, fontsize=9)


def tubeplot(ax, data, names, xl, yl, xticklabels=None, rotate=False,
             xloc=None, xlabelpad=None, ylabelpad=None, xlim=None,
             ylim=None, x=None):
    if x is None:
        x = np.arange(int(len(names)))
    # medians2 = np.median(data, 1)
    medians = np.percentile(data, 50, axis=1)
    lower_quartiles = np.percentile(data, 25, axis=1)
    upper_quartiles = np.percentile(data, 75, axis=1)

    # ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
    #               alpha=0.5)
    # Hide these grid behind plot objects
    # ax.set_axisbelow(True)
    ax.set_xlabel(xl, labelpad=xlabelpad)
    ax.set_ylabel(yl, labelpad=ylabelpad)

    ax.fill_between(x, lower_quartiles, upper_quartiles, alpha=0.25, color='#1F4A7D', lw=0.0)
    ax.plot(x, medians, color='#1F4A7D')
    # ax.plot(x, medians2, color='r', linestyle='-')

    # Set the axes ranges and axes labels
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if xloc is not None:
        # ax.xaxis.set_major_locator(IndexLocator(10, 0))
        ax.xaxis.set_major_locator(xloc)
    if xticklabels is not None:
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        xticklabels = plt.setp(ax, xticklabels=xticklabels)
    if rotate:
        plt.setp(xticklabels, fontsize=8, rotation=90)


def logplots(x, y):
    fig, ax = plt.subplots(2, 2)
    ax[0,1].scatter(x, y, color='k')
    ax[0,1].set_yscale('log')

    ax[1,0].scatter(x, y, color='k')
    ax[1,0].set_xscale('log')

    ax[1,1].scatter(x, y, color='k')
    ax[1,1].set_xscale('log')
    ax[1,1].set_yscale('log')

    ax[0,0].scatter(x, y, color='k')


def inverse(v, v_min, v_max):
    return (-1.0 * v) + v_min + v_max


def downstream(transformation, x, y, base='e', invert=False):
    if base == 'e':
        log = np.log
    elif base == '10':
        log = np.log10
    elif base == '2':
        log = np.log2
    else:
        raise RuntimeError('unsupported base: %s' % base)

    if invert:
        y = inverse(y, np.min(y), np.max(y))

    if transformation == 'lin-lin':
        x_t, y_t = x, y
    elif transformation == 'log-lin':
        x_t, y_t = x, log(y)
    elif transformation == 'sqrt-lin':
        x_t, y_t = x, np.sqrt(y)
    elif transformation == 'reciprocal':
        x_t, y_t = x, 1.0 / y
    elif transformation == 'lin-log':
        x_t, y_t = log(x), y
    elif transformation == 'log-log':
        x_t, y_t = log(x), log(y)
    else:
        raise RuntimeError('unknown transformation: %s' % transformation)

    return x_t, y_t


def upstream(transformation, c0, c1, x, base='e'):
    if base == 'e':
        log = np.log
        b = np.e
    elif base == '10':
        log = np.log10
        b = 10
    elif base == '2':
        log = np.log2
        b = 2
    else:
        raise RuntimeError('unsupported base: %s' % base)

    if transformation == 'lin-lin':
        y_t = c0 + c1 * x
    elif transformation == 'log-lin':
        y_t = b**(c0 + c1 * x)
    elif transformation == 'sqrt-lin':
        y_t = (c0 + c1 * x)**2
    elif transformation == 'reciprocal':
        y_t = 1.0 / (c0 + c1 * x)
    elif transformation == 'lin-log':
        y_t = c0 + c1 * log(x)
    elif transformation == 'log-log':
        y_t = b**(c0 + c1 * log(x))
        # y_t = b**c0 * x**c1   # different notation, same result
    else:
        raise RuntimeError('unknown transformation: %s' % transformation)

    return y_t


def fitting_plot(ax, x, y, xl, yl, tests, xlim=None, ylim=None,
                 xlabelpad=None, ylabelpad=None, extra_range=0):
    # Set the axes ranges and axes labels
    ax.set_xlabel(xl, labelpad=xlabelpad)
    ax.set_ylabel(yl, labelpad=ylabelpad)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1] + extra_range)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    # Plot medians
    medians = []
    for v in y:
        medians.append(np.median(v))
    y = np.array(medians)
    ax.scatter(x, y, color='k')
    # Test and plot different growth functions
    xpoints = np.linspace(x[0], x[-1] + extra_range, 100)
    residuals = OrderedDict()
    styles = ((None, None), (8, 1), (6, 1), (4, 1), (2, 1))
    for transformation, style in zip(tests, styles):
        # invert if data is decreasing
        invert = y[0] > y[-1]
        # transformation
        x_t, y_t = downstream(transformation, x, y, invert=invert)
        # fitting
        p = np.polyfit(x_t, y_t, 1)[::-1]
        # back-transformation
        estimator = upstream(transformation, p[0], p[1], xpoints)
        y_p = upstream(transformation, p[0], p[1], x)
        if invert:
            estimator = inverse(estimator, np.min(y), np.max(y))
            y_p = inverse(y_p, np.min(y), np.max(y))
        # goodness-of-fit
        # http://de.wikipedia.org/wiki/Bestimmtheitsma%C3%9F
        # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample
        r = (1 / len(y - 1)) * np.sum(((y - np.mean(y)) / np.std(y)) *
                                      ((y_p - np.mean(y_p)) / np.std(y_p)))
        residuals[transformation] = (p, r**2)
        ax.plot(xpoints, estimator, dashes=style, label=transformation, lw=0.75)
    if len(xl) == 0:
        plt.setp(ax.get_xticklabels(), visible=False)
    print yl
    for k, v in residuals.items():
        print '\t%s: %s' % (k, v)
    return residuals


def autolabel(ax, rects, labels):
    # attach some text labels
    for rect, label in zip(rects, labels):
        xpos = rect.get_x() + rect.get_width() / 2.0
        bottom = ax.get_ylim()[0]
        ypos = max(0, bottom + bottom * 0.01)
        ax.text(xpos, ypos, label, ha='left', va='center',
                color=PRIMD, fontsize=6, rotation='vertical',
                rotation_mode='anchor')


def residuals_plot(ax, names, residuals, xl, yl, ylim=None, xticks=None,
                   xlabelpad=None, ylabelpad=None):
    x = np.arange(len(names))
    y = []
    for name in names:
        y.append(residuals[name][1])

    # Set the axes ranges and axes labels
    ax.set_xlabel(xl, labelpad=xlabelpad)
    ax.set_ylabel(yl, labelpad=ylabelpad)
    if ylim is None:
        y_min = min(y)
        ax.set_ylim(y_min - y_min * 0.15, 1)
    else:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(-0.5, x[-1] + 0.5)
    ax.xaxis.set_major_locator(FixedLocator(x))
    ax.grid(False, which='major', axis='x')

    bars = ax.bar(x, y, align='center', width=0.8, facecolor=PRIM+(0.5,), edgecolor=EC)
    # labels = ['$\mathsf{%s: R}^2 = %.3f$' % (
    #         name.replace('-', '\!\operatorname{-}\!'), r) for name, r in zip(names, y)]
    labels = ['$\mathsf{R}^2 = %.3f$' %r for r in y]
    autolabel(ax, bars, labels)
    if xticks is None:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xticklabels(xticks, rotation='vertical')


if __name__ == '__main__':
    st = []
    for dir in sys.argv[1:]:
        if os.path.isdir(dir):
            st.append(stats_raw(*analyze(dir)))

    # sim_steps_mean, sim_steps_std, \
    #     fitness_mean, fitness_std, \
    #     fitness_bkc_mean, fitness_bkc_std, \
    #     difference_mean, difference_std, \
    #     messages_mean, messages_std, \
    #     m_s_mean, m_s_std = \
    #     [np.array([s[i] for s in st]) for i in range(len(st[0]))]

    sim_steps, fitness, bkc, difference, messages, pendings, objcalls, o_m, o_n, o_q = \
        [np.array([s[i] for s in st]) for i in range(len(st[0]))]
    # messages (per agent)
    mpa = messages / o_m
    # messages (per simstep)
    mps = messages / sim_steps
    # messages (per agent per simstep)
    mpaps = messages / o_m / sim_steps

    # objcalls (per agent)
    opa = objcalls / o_m
    # objcalls (per simstep)
    ops = objcalls / sim_steps
    # objcalls (per agent per simstep)
    opaps = objcalls / o_m / sim_steps

    # Normierte Laufzeiteffizienz
    eff_steps = (1 - bkc) / sim_steps
    # Normierte Kommunikationseffizienz
    eff_msgs = o_m * (1 - bkc) / messages
    # Normierte Berechnungseffizienz
    eff_calc = o_m * (1 - bkc) / objcalls

    # print '--------------------------'
    # print 'overall means (avg, std):'
    # print 'sim steps:', sim_steps.mean(1).mean(), sim_steps.std(1).mean()
    # print 'fitness:', fitness.mean(1).mean(), fitness.std(1).mean()
    # print 'fitness (bkc):', bkc.mean(1).mean(), bkc.std(1).mean()
    # print 'difference:', difference.mean(1).mean(), difference.std(1).mean()
    # print 'messages/agent/sim step:', mpaps.mean(1).mean(), mpaps.std(1).mean()
    # print 'messages/agent:', mpa.mean(1).mean(), mpa.std(1).mean()
    # print 'messages:', messages.mean(1).mean(), messages.std(1).mean()
    # print 'pending messages:', pendings.mean(1).mean(), pendings.std(1).mean()
    # print 'objective calls:', objcalls.mean(1).mean(), objcalls.std(1).mean()
    # print '--------------------------'
    # print






    # # msg_max
    # x = [1, 2, 5, 7, 10][:int(len(st) / 2)]

    # # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # fig.suptitle('influence of message delay ($\\phi=0.5$)')
    # # fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08)
    # # plot(ax[0], None, 'fitness', x, fitness_mean[2::2], err(fitness_mean[2::2], fitness_std[2::2]))
    # # plot(ax[1], None, 'sim_steps', x, sim_steps_mean[2::2], err(sim_steps_mean[2::2], sim_steps_std[2::2]))
    # # # plot(ax[2], None, 'difference', x, difference_mean[2::2], err(difference_mean[2::2], difference_std[2::2]))
    # # plot(ax[2], 'message delay (max)', 'messages', x, messages_mean[2::2], err(messages_mean[2::2], messages_std[2::2]))

    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # fig.suptitle('influence of message delay')
    # fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.98, hspace=0.1)
    # plot(ax[0], None, 'fitness', x, fitness_mean[3::2], err(fitness_mean[3::2], fitness_std[3::2]))
    # plot(ax[1], None, 'sim_steps', x, sim_steps_mean[3::2], err(sim_steps_mean[3::2], sim_steps_std[3::2]))
    # # plot(ax[2], None, 'difference', x, difference_mean[3::2], err(difference_mean[3::2], difference_std[3::2]))
    # plot(ax[2], 'message delay (max)', 'messages', x, messages_mean[3::2], err(messages_mean[3::2], messages_std[3::2]))

    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.suptitle('influence of message delay')
    # plot(ax, 'message delay (max)', 'messages $\cdot$ simulation steps', x, mpa_mean[3::2], err(mpa_mean[3::2], mpa_std[3::2]))
    # fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.98, hspace=0.1)


    # # opt_q
    # x = [8, 16, 32, 48, 96][:int(len(st))]

    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # fig.suptitle('influence of planning horizon')
    # fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.98, hspace=0.1)
    # plot(ax[0], None, 'fitness', x, fitness_mean, err(fitness_mean, fitness_std))
    # plot(ax[1], None, 'simulation steps', x, sim_steps_mean, err(sim_steps_mean, sim_steps_std))
    # # plot(ax[2], None, 'difference', x, difference_mean, err(difference_mean, difference_std))
    # plot(ax[2], 'planning horizon', 'messages', x, messages_mean, err(messages_mean, messages_std))


    # opt_m
    # x = [10, 25, 30, 50, 100][:int(len(st))]

    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # fig.suptitle('influence of population size')
    # fig.subplots_adjust(left=0.12, right=0.97, bottom=0.08, top=0.98, hspace=0.1)
    # plot(ax[0], None, 'fitness', x, fitness_mean, err(fitness_mean, fitness_std))
    # plot(ax[1], None, 'simulation steps', x, sim_steps_mean, err(sim_steps_mean, sim_steps_std))
    # # plot(ax[2], None, 'difference', x, difference_mean, err(difference_mean, difference_std))
    # plot(ax[2], 'population size', 'messages', x, messages_mean, err(messages_mean, messages_std))

    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.suptitle('influence of message delay')
    # plot(ax, 'population size', 'messages $\cdot$ simulation steps', x, mpa_mean, err(mpa_mean, mpa_std))
    # fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.98, hspace=0.1)

    # opt_n
    # x = [20, 200, 2000, 20000, 200000][:int(len(st))]
    # # names = [20, 200, 2000, 20000, 'SVDD-model'][:int(len(st))]

    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # fig.suptitle('influence of search space size')
    # fig.subplots_adjust(left=0.12, right=0.97, bottom=0.1, top=0.98, hspace=0.1)
    # # plot(ax[0], None, 'fitness', x, fitness_mean, err(fitness_mean, fitness_std), logx=True)
    # plot(ax[0], None, 'fitness', x, fitness_bkc_mean, err(fitness_bkc_mean, fitness_bkc_std), logx=True)
    # plot(ax[1], None, 'simulation steps', x, sim_steps_mean, err(sim_steps_mean, sim_steps_std), logx=True)
    # # plot(ax[2], None, 'difference', x, difference_mean, err(difference_mean, difference_std), logx=True)
    # plot(ax[2], 'size of local search spaces', 'messages', x, messages_mean, err(messages_mean, messages_std), logx=True)

    # networks
    # names = ['ring', 'small world\n$\phi=0.1$', 'small world\n$\phi=0.5$',
    #         'small world\n$\phi=1.0$', 'small world\n$\phi=2.0$',
    #         'small world\n$\phi=4.0$'][:int(len(st))]              # ignore the 'mesh' topology for now
    # x = np.arange(len(names))

    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # fig.suptitle('influence of search space size')
    # plot(ax[0], None, 'fitness', x, fitness_mean, err(fitness_mean, fitness_std), xticknames=names)
    # plot(ax[1], None, 'simulation steps', x, sim_steps_mean, err(sim_steps_mean, sim_steps_std), xticknames=names)
    # # plot(ax[2], None, 'difference', x, difference_mean, err(difference_mean, difference_std), xticknames=names)
    # plot(ax[2], 'network topology', 'messages', x, messages_mean, err(messages_mean, messages_std), xticknames=names, xlabelpad=-8)
    # fig.autofmt_xdate()
    # fig.subplots_adjust(left=0.12, right=0.97, bottom=0.17, top=0.98, hspace=0.1)

    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.suptitle('influence of message delay')
    # plot(ax, 'network topology', 'messages $\cdot$ simulation steps', x, mpa_mean, err(mpa_mean, mpa_std), xticknames=names, xlabelpad=-8)
    # fig.autofmt_xdate()
    # fig.subplots_adjust(left=0.09, right=0.99, bottom=0.26, top=0.98, hspace=0.1)









    # ##########################################################################
    # # opt_n vs. error[0, 1]
    # # python analyze.py ../data/2013-03-26_CHP,* ../data/2013-03-26_SVSM-CHP,n00200_91508f017113/
    # names = [20, 200, 2000, 20000, 'SVDD-\nmodel'][:int(len(st))]
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # fig.suptitle('opt_n vs. error[0, 1]')
    # boxplot(ax, bkc, names, 'size of local search spaces', 'error [0, 1]')

    # opt_n vs. simsteps
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False)
    # fig.subplots_adjust(left=0.1, right=0.96, bottom=0.13, top=0.97)
    # # fig.suptitle('opt_n vs. simsteps')
    # boxplot(ax, sim_steps, names, 'simulation steps', 'size of local search spaces', vert=False)

    # # opt_n vs. messages per agent per simstep
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # fig.suptitle('opt_n vs. messages per agent per simstep')
    # boxplot(ax, mpaps, names, 'size of local search spaces', 'messages per agent per simstep')

    # # opt_n vs. messages per agent
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # fig.suptitle('opt_n vs. messages per agent')
    # boxplot(ax, mpa, names, 'size of local search spaces', 'messages per agent')

    # # opt_n vs. messages
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.97)
    # # fig.suptitle('opt_n vs. messages')
    # boxplot(ax, messages, names, 'size of local search spaces', 'messages')

    ###########################################################################
    # # bandwidth vs. error[0, 1]
    # import itertools
    # names = ([0.1, 0.5] + [float(bw) for bw in itertools.chain(range(1, 21), range(25, 51, 5), range(60, 101, 10))])[:int(len(st))]
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.08, right=0.97, bottom=0.1, top=0.94)
    # fig.suptitle('bandwidth vs. error[0, 1]')
    # boxplot(ax, bkc, names, 'bandwidth parameter', 'error[0, 1]')

    ###########################################################################
    # # opt_m vs. error[0, 1]
    # names = [50, 100, 200, 500][:int(len(st))]
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # fig.suptitle('opt_n vs. error[0, 1]')
    # boxplot(ax, bkc, names, 'population size', 'error[0, 1]')

    # # opt_m vs. simsteps
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # fig.suptitle('opt_n vs. simsteps')
    # boxplot(ax, sim_steps, names, 'population size', 'simulation steps')

    # # opt_m vs. messages per agent per simstep
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # fig.suptitle('opt_n vs. messages per agent per simstep')
    # boxplot(ax, mpaps, names, 'population size', 'messages per agent per simstep')


    # x = [50, 100, 200, 500][:int(len(st))]
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.suptitle('influence of population size')
    # fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.98, hspace=0.1)
    # plot(ax, 'population size', 'simulation steps', x, sim_steps.mean(1), err(sim_steps.mean(1), sim_steps.std(1)))

    ###########################################################################
    # # agent_delay vs. error [0, 1]
    # names = [x for x in range(1, 10)][:int(len(st))]
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # boxplot(ax, bkc, names, 'agent delay', 'error [0, 1]')

    # # agent_delay vs. simsteps
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # boxplot(ax, sim_steps, names, 'agent delay', 'simulation steps')

    # # # agent_delay vs. messages per agent per simstep
    # # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # boxplot(ax, mpaps, names, 'agent delay', 'messages per agent per simstep')

    # # agent_delay vs. messages per agent
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # boxplot(ax, mpa, names, 'agent delay', 'messages per agent')

    # # # agent_delay vs. messages
    # # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.97)
    # # boxplot(ax, messages, names, 'agent delay', 'messages')


    ###########################################################################
    # # p_accept vs. error [0, 1]
    # names = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0][:int(len(st))]
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # boxplot(ax, bkc, names, '$p_{accept}$', 'error [0, 1]')

    # # agent_delay vs. simsteps
    # # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False)
    # # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # boxplot(ax, sim_steps, names, '$p_{accept}$', 'simulation steps')

    # # # agent_delay vs. messages per agent per simstep
    # # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # boxplot(ax, messages, names, '$p_{accept}$', 'messages per agent per simstep')

    # # agent_delay vs. messages per agent
    # # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # # boxplot(ax, mpa, names, '$p_{accept}$', 'messages per agent')

    # # # agent_delay vs. messages
    # # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # # fig.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.97)
    # # boxplot(ax, messages, names, '$p_{accept}$', 'messages')

    ###########################################################################
    # # stigspace vs. error [0, 1]
    # names = ['smallworld', 'stigspace'][:int(len(st))]
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.97)
    # boxplot(ax, bkc, names, 'approach', 'error [0, 1]')









    ###########################################################################
    # SC scenarios
    # # -------------------------------------------------------------------------
    # # instance h
    # names = ['h%03d' % d for d in range(1, len(st) + 1)]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # tubeplot(ax[0], bkc, names, '', 'error [0, 1]')
    # tubeplot(ax[1], sim_steps, names, '', 'simulation steps')
    # tubeplot(ax[2], mpaps, names, 'instance', 'messages per agent per simulation step')

    # # -------------------------------------------------------------------------
    # # Single plots for instance h
    # #
    # names = ['h%03d' % d for d in range(1, len(st) + 1)]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # xloc = FixedLocator([0] + [(10 * x)-1 for x in range(1, 11)])

    # # instance h vs. error [0, 1]
    # tubeplot(ax, bkc, names, 'Instanz', 'Fehler', xticklabels=[names[0]]+names[9::10], rotate=True, xloc=xloc)
    # ax.set_ylim(top=0.02)

    # # instance h vs. simsteps
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # tubeplot(ax, sim_steps, names, 'Instanz', 'Simulationsdauer', xticklabels=[names[0]]+names[9::10], rotate=True, xloc=xloc)
    # ax.set_ylim(10, 30)

    # # # instance h vs. mpaps
    # # tubeplot(ax, mpaps, names, 'Instanz', 'Nachrichten pro Agent und Zeitschritt', xticklabels=[names[0]]+names[9::10], rotate=True, xloc=xloc)
    # # ax.set_ylim(0, o_m.max())
    # # import pdb
    # # pdb.set_trace()

    # # instance h vs. opaps
    # tubeplot(ax, opaps, names, 'Instanz', 'Evaluationen pro Agent und Zeitschritt', xticklabels=[names[0]]+names[9::10], rotate=True, xloc=xloc)
    # # ax.set_ylim(0, o_m.max())

    # # -------------------------------------------------------------------------
    # # agentdelay
    # # python analyze.py ../data/2013-11-02_SC,agentdelay/2013-11-0*
    # x = range(1, len(st) + 1)
    # names = ['%d' % d for d in x]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.01))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=14)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=14)
    # boxplot(ax[3], objcalls, names, 'Reaktionszeit', 'Funktionsaufrufe', ylabelpad=8)

    # # -------------------------------------------------------------------------
    # # msgdelay
    # # python analyze.py ../data/2013-11-02_SC,msgdelay/*
    # x = range(1, len(st) + 1)
    # names = ['%d' % d for d in x]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # xloc = FixedLocator(x)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.01))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=14)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=14)
    # boxplot(ax[3], objcalls, names, r"""Nachrichtenverz\"{o}gerung""", 'Funktionsaufrufe', ylabelpad=8)

    # # -------------------------------------------------------------------------
    # # network
    # # python analyze.py ../data/2013-11-02_SC,network/0*
    # names = ['Liste', 'Ring', 'Small-World', 'Vollvernetzt', 'Gitter'][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # xloc = FixedLocator(np.arange(len(names)))
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.01))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=20)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=14)
    # boxplot(ax[3], objcalls, names, 'Topologie', 'Funktionsaufrufe', rotate=True, ylabelpad=8)

    # # -------------------------------------------------------------------------
    # # phi
    # # python analyze.py ../data/2013-11-02_SC,phi/*
    # names = ['%.01f' % d for d in (0.0, 0.1, 0.5, 1.0, 2.0, 4.0)][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # xloc = FixedLocator(np.arange(1, len(names) + 1))
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.01))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=20)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=8)
    # boxplot(ax[3], objcalls, names, r"""Small-World $\phi$""", 'Funktionsaufrufe', xloc=xloc, ylabelpad=8)
    # # ax[3].xaxis.get_major_ticks()[0].label.set_rotation('vertical')

    # # -------------------------------------------------------------------------
    # # network: mesh vs. smallworld
    # # python analyze.py ../data/2013-11-04_SC,mesh/*
    # names = ['Small-World,\n$m=10$', 'Gitter,\n$m=10$', 'Small-World,\n$m=20$', 'Gitter,\n$m=20$', 'Small-World,\n$m=50$', 'Gitter,\n$m=50$'][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # xloc = FixedLocator(np.arange(len(names)))
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.01), ylabelpad=8)
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=16)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=10)
    # boxplot(ax[3], objcalls, names, 'Topologie (Small-World vs. Gitter)', 'Funktionsaufrufe', rotate=True, ylabelpad=4)

    # # -------------------------------------------------------------------------
    # # m
    # # python analyze.py ../data/2013-11-02_SC,m/*
    # names = ['%d' % d for d in (10, 20, 30, 40, 50, 100, 500)][:len(st)]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz')
    # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz', ylabelpad=12)
    # boxplot(ax[2], eff_calc, names, r"""Verbundgr\"{o}\ss{}e""", 'Berechnungseffizienz')
    # # -------------------------------------------------------------------------
    # # m (raw)
    # names = ['%d' % d for d in (10, 20, 30, 40, 50, 100, 500)][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.01))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=12)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=14, logy=True)
    # boxplot(ax[3], objcalls, names, r"""Verbundgr\"{o}\ss{}e""", 'Funktionsaufrufe', ylabelpad=14, logy=True)

    # # -------------------------------------------------------------------------
    # # n
    # # python analyze.py ../data/2013-11-02_SC,n/*
    # names = ['%d' % d for d in (5, 10, 100, 1000, 10000)][:len(st)]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz', ylabelpad=8)
    # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz', ylabelpad=8)
    # boxplot(ax[2], eff_calc, names, r"""Suchraumgr\"{o}\ss{}e""", 'Berechnungseffizienz', ylim=(-0.0001,))
    # # -------------------------------------------------------------------------
    # # n (raw)
    # names = ['%d' % d for d in (5, 10, 100, 1000, 10000)][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.01))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=20)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=14)
    # boxplot(ax[3], objcalls, names, r"""Suchraumgr\"{o}\ss{}e""", 'Funktionsaufrufe', logy=True, ylabelpad=16)

    # # -------------------------------------------------------------------------
    # # q
    # # python analyze.py ../data/2013-11-02_SC,q/*
    # names = ['%d' % d for d in (5, 10, 20, 30, 40, 50, 100)][:len(st)]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz', ylabelpad=8)
    # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz', ylabelpad=8)
    # boxplot(ax[2], eff_calc, names, 'Planungshorizont', 'Berechnungseffizienz')
    # # -------------------------------------------------------------------------
    # # q (raw)
    # names = ['%d' % d for d in (5, 10, 20, 30, 40, 50, 100)][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.015))
    # ylim = (sim_steps.min() - (sim_steps.min() * 0.05), sim_steps.max() + (sim_steps.max() * 0.05))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylim=ylim, ylabelpad=20)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=14)
    # boxplot(ax[3], objcalls, names, 'Planungshorizont', 'Funktionsaufrufe', ylabelpad=8)

    # # -------------------------------------------------------------------------
    # # p static
    # # python analyze.py ../data/2013-11-12_SC,p_static/*
    # # names = ['%0.1f' % (p / 10.0) for p in range(10)][:len(st)]
    # # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz', ylim=(-0.0004,), ylabelpad=13)
    # # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz', ylim=(-0.0003,), ylabelpad=13)
    # # boxplot(ax[2], eff_calc, names, 'Gesinnungswahl ($p_{\mathrm{static}}$)', 'Berechnungseffizienz', ylim=(-0.0001,))
    # # -------------------------------------------------------------------------
    # # p static (raw)
    # names = ['%0.1f' % (p / 10.0) for p in range(10)][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.005, 1.01), ylabelpad=14)
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=16)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=10)
    # boxplot(ax[3], objcalls, names, 'Gesinnungswahl ($\psi_{\mathrm{stat}}$)', 'Funktionsaufrufe')

    # # -------------------------------------------------------------------------
    # # p dynamic
    # # python analyze.py ../data/2013-11-12_SC,p_dynamic/*
    # # names = ['%0.1f' % (p / 10.0) for p in range(10)][:len(st)]
    # # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz', ylabelpad=8)
    # # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz', ylabelpad=8)
    # # boxplot(ax[2], eff_calc, names, 'Gesinnungswahl ($p_{\mathrm{dynamic}}$)', 'Berechnungseffizienz')
    # # -------------------------------------------------------------------------
    # # p dynamic (raw)
    # names = ['%0.1f' % (p / 10.0) for p in range(10)][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.003,), ylabelpad=14)
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=16)
    # boxplot(ax[2], messages, names, '', 'Nachrichten')
    # boxplot(ax[3], objcalls, names, 'Gesinnungswahl ($\psi_{\mathrm{dyn}}$)', 'Funktionsaufrufe', ylabelpad=4)
    # # -------------------------------------------------------------------------
    # # p dynamic (raw), only error
    # names = ['%0.1f' % (p / 10.0) for p in range(10)][:len(st)]
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(left=0.13, right=0.88, bottom=0.23, top=0.85)
    # boxplot(ax, bkc, names, 'Gesinnungswahl ($\psi_{\mathrm{dyn}}$)', 'Fehler', ylim=(-0.0001, 0.01), ylabelpad=8)


    ###########################################################################
    # Energy scenarios

    # # -------------------------------------------------------------------------
    # # Suchraum
    # # python analyze.py ../data/2013-03-26_CHP,n00200_91508f017113/ ../data/2013-03-26_SVSM-CHP,n00200_91508f017113/
    # names = ['$\\sigma^{\mathrm{enum}}$', '$\\sigma^{\mathrm{SVM}}$']
    # # # ACHTUNG: für die Effizienzauswertung in der Diss wurde "bkc_thres = 0.01" verwendet.
    # # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz', ylabelpad=6)
    # # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz', ylabelpad=6)
    # # boxplot(ax[2], eff_calc, names, 'Suchraummodell', 'Berechnungseffizienz', ylim=(-0.0001,))
    # # -------------------------------------------------------------------------
    # # Suchraum (raw)
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0001, 0.02), ylabelpad=8)
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=16)
    # boxplot(ax[2], messages, names, '', 'Nachrichten')
    # boxplot(ax[3], objcalls, names, 'Suchraummodell', 'Funktionsaufrufe', ylabelpad=18)
    # ax[3].set_yscale('log')

    # # -------------------------------------------------------------------------
    # # m
    # # python analyze.py ../data/2013-11-30_CHP,m/*
    # names = ['%d' % d for d in (10, 20, 30, 40, 50, 100, 500)][:len(st)]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz')
    # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz')
    # boxplot(ax[2], eff_calc, names, r"""Verbundgr\"{o}\ss{}e""", 'Berechnungseffizienz')
    # # -------------------------------------------------------------------------
    # # m (raw)
    # names = ['%d' % d for d in (10, 20, 30, 40, 50, 100, 500)][:len(st)]
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], bkc, names, '', 'Fehler', ylim=(-0.0005, 0.07))
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer', ylabelpad=8)
    # boxplot(ax[2], messages, names, '', 'Nachrichten', ylabelpad=10, logy=True)
    # boxplot(ax[3], objcalls, names, r"""Verbundgr\"{o}\ss{}e""", 'Funktionsaufrufe', ylabelpad=10, logy=True)


    ###########################################################################
    # Regression
    # # http://stackoverflow.com/questions/3433486/how-to-do-exponential-and-logarithmic-curve-fitting-in-python-i-found-only-poly
    # # http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    # # http://zunzun.com/
    # # http://stattrek.com/regression/linear-transformation.aspx
    # Legend codes:
    # ‘best’  0
    # ‘upper right’   1
    # ‘upper left’    2
    # ‘lower left’    3
    # ‘lower right’   4
    # ‘right’ 5
    # ‘center left’   6
    # ‘center right’  7
    # ‘lower center’  8
    # ‘upper center’  9
    # ‘center’    10
    #
    # tests = ('lin-log', 'log-log', 'lin-lin', 'sqrt-lin', 'log-lin')
    # fig = plt.figure()
    # ax00 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    # ax10 = plt.subplot2grid((3, 3), (1, 0), colspan=2, sharex=ax00)
    # ax20 = plt.subplot2grid((3, 3), (2, 0), colspan=2, sharex=ax00)
    # ax01 = plt.subplot2grid((3, 3), (0, 2))
    # ax11 = plt.subplot2grid((3, 3), (1, 2))
    # ax21 = plt.subplot2grid((3, 3), (2, 2))
    # ax00.set_title('Regressionstests', fontsize='small', color='#555555')
    # ax01.set_title(r"""Bestimmtheitsma\ss{}""", fontsize='small', color='#555555')
    # fig.subplots_adjust(left=0.12, right=0.96, wspace=0.5)

    # # -------------------------------------------------------------------------
    # # phi
    # # python analyze.py ../data/2013-11-02_SC,phi/2013-11-02_SC,phi*
    # x = np.array([0.1, 0.5, 1.0, 2.0, 4.0])[:len(st)]
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, '', 'Nachrichten', tests)
    # r_objcalls = fitting_plot(ax20, x, objcalls, r"""Small-World $\phi$""", 'Funktionsaufrufe', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=1)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '')
    # residuals_plot(ax21, tests, r_objcalls, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # msgdelay
    # # python analyze.py ../data/2013-11-02_SC,msgdelay/*
    # x = np.arange(1, len(st) + 1)
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, '', 'Nachrichten', tests)
    # r_objcalls = fitting_plot(ax20, x, objcalls, r"""Nachrichtenverz\"{o}gerung""", 'Funktionsaufrufe', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=4)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '')
    # residuals_plot(ax21, tests, r_objcalls, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # agentdelay
    # # python analyze.py ../data/2013-11-02_SC,agentdelay/2013-11-0*
    # x = np.arange(1, len(st) + 1)
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, '', 'Nachrichten', tests)
    # r_objcalls = fitting_plot(ax20, x, objcalls, 'Reaktionszeit', 'Funktionsaufrufe', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=4)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '')
    # residuals_plot(ax21, tests, r_objcalls, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # m (raw)
    # # python analyze.py ../data/2013-11-02_SC,m/*
    # fig.subplots_adjust(left=0.14, right=0.96, wspace=0.5)
    # x = np.array([10, 20, 30, 40, 50, 100, 500])[:len(st)]
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, '', 'Nachrichten', tests)
    # r_objcalls = fitting_plot(ax20, x, objcalls, r"""Verbundgr\"{o}\ss{}e""", 'Funktionsaufrufe', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=4)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '')
    # residuals_plot(ax21, tests, r_objcalls, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # n (raw)
    # # python analyze.py ../data/2013-11-02_SC,n/*
    # fig.subplots_adjust(left=0.14, right=0.96, wspace=0.5)
    # x = np.array([5, 10, 100, 1000, 10000])[:len(st)]
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]), ylim=(20, 30))
    # r_messages = fitting_plot(ax10, x, messages, '', 'Nachrichten', tests, ylim=(380, 420))
    # r_objcalls = fitting_plot(ax20, x, objcalls, r"""Suchraumgr\"{o}\ss{}e""", 'Funktionsaufrufe', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=1)
    # residuals_plot(ax01, tests, r_sim_steps, '', '', ylim=(0, 1))
    # residuals_plot(ax11, tests, r_messages, '', '')
    # residuals_plot(ax21, tests, r_objcalls, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # q (raw)
    # # python analyze.py ../data/2013-11-02_SC,q/*
    # x = np.array([5, 10, 20, 30, 40, 50, 100])[:len(st)]
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, '', 'Nachrichten', tests)
    # r_objcalls = fitting_plot(ax20, x, objcalls, 'Planungshorizont', 'Funktionsaufrufe', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=1)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '')
    # residuals_plot(ax21, tests, r_objcalls, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # m (raw) --> sim_steps and objcalls combined
    # # python analyze.py ../data/2013-11-02_SC,m/*
    # fig.subplots_adjust(left=0.14, right=0.96, wspace=0.5)
    # x = np.array([10, 20, 30, 40, 50, 100, 500])[:len(st)]
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]))
    # r_objcalls = fitting_plot(ax10, x, objcalls, '', 'Funktionsaufrufe', tests)
    # r_combined = fitting_plot(ax20, x, objcalls / sim_steps, r"""Verbundgr\"{o}\ss{}e""", '$\\frac{\\text{Funktionsaufrufe}}{\\text{Simulationsdauer}}$', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=4)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_objcalls, '', '')
    # residuals_plot(ax21, tests, r_combined, '', '', xticks=tests)
    # # RESULT: when objcalls is normed with sim_steps, overall complexity is LINEAR!

    # # -------------------------------------------------------------------------
    # # m (CHP raw)
    # # python analyze.py ../data/2013-11-30_CHP,m/*
    # x = np.array([10, 20, 30, 40, 50, 100, 500])[:len(st)]
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulationsdauer', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, '', 'Nachrichten', tests)
    # r_objcalls = fitting_plot(ax20, x, objcalls, r"""Verbundgr\"{o}\ss{}e""", 'Funktionsaufrufe', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=4)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '')
    # residuals_plot(ax21, tests, r_objcalls, '', '', xticks=tests)


    ###########################################################################
    # Stigspace

    # # -------------------------------------------------------------------------
    # # (eff)
    # # python analyze.py ../data/2013-12-07_CHP,stigspace/*
    # names = ['$p=%.1f$, $\mathsf{V}=%d$' % (p, v)
    #             for (p, v) in ((0.0, 1), (0.0, 2), (0.1, 1), (0.1, 2))][:len(st)]
    # # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # # boxplot(ax[0], eff_steps, names, '', 'Laufzeiteffizienz', ylabelpad=12)
    # # boxplot(ax[1], eff_msgs, names, '', 'Kommunikationseffizienz', ylabelpad=12)
    # # boxplot(ax[2], eff_calc, names, 'Suchraummodell', 'Berechnungseffizienz', ylim=(-0.0001,))
    # # -------------------------------------------------------------------------
    # # (raw)
    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # boxplot(ax[0], fitness, names, '', 'Fehler')
    # boxplot(ax[1], sim_steps, names, '', 'Simulationsdauer')
    # boxplot(ax[2], messages, names, '', 'Nachrichten')
    # boxplot(ax[3], objcalls, names, 'Suchraummodell', 'Funktionsaufrufe')


    ###########################################################################
    # MATES 2014

    # # -------------------------------------------------------------------------
    # # phi
    # # python analyze.py ../data/2014-06-24_SC,h-rnd,phi/*
    # names = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0][:len(st)]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.09)
    # # xloc = FixedLocator(np.arange(1, len(names) + 1))
    # boxplot(ax[0], bkc, names, '', 'Error', ylim=(-0.0003, 0.02))
    # boxplot(ax[1], sim_steps, names, '', 'Simulation steps', ylabelpad=18)
    # boxplot(ax[2], messages, names, r"""Link density $\phi$""", 'Messages', ylabelpad=6)

    # # -------------------------------------------------------------------------
    # # msgdelay
    # # python analyze.py ../data/2014-06-25_SC,h-rnd,msgdelay/*
    # x = range(1, len(st) + 1)
    # names = ['%d' % d for d in x]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.09)
    # xloc = FixedLocator(x)
    # boxplot(ax[0], bkc, names, '', 'Error', ylim=(-0.0003, 0.02))
    # boxplot(ax[1], sim_steps, names, '', 'Simulation steps', ylabelpad=14)
    # boxplot(ax[2], messages, names, 'Communication delay $d_{\mathrm{max}}$', 'Messages', ylabelpad=14)

    # # -------------------------------------------------------------------------
    # # agentdelay
    # # python analyze.py ../data/2014-06-24_SC,h-rnd,agentdelay/*
    # x = range(1, len(st) + 1)
    # names = ['%d' % d for d in x]
    # fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    # fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.09)
    # boxplot(ax[0], bkc, names, '', 'Error', ylim=(-0.0003, 0.02))
    # boxplot(ax[1], sim_steps, names, '', 'Simulation steps', ylabelpad=14)
    # boxplot(ax[2], messages, names, 'Reaction delay $r_{\mathrm{max}}$', 'Messages', ylabelpad=14)


    # # -------------------------------------------------------------------------
    # # Regression
    # tests = ('lin-log', 'log-log', 'lin-lin', 'sqrt-lin', 'log-lin')
    # fig = plt.figure()
    # ax00 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    # ax10 = plt.subplot2grid((2, 3), (1, 0), colspan=2, sharex=ax00)
    # ax01 = plt.subplot2grid((2, 3), (0, 2))
    # ax11 = plt.subplot2grid((2, 3), (1, 2))
    # fig.subplots_adjust(left=0.12, right=0.96, bottom=0.15, wspace=0.5)

    # # -------------------------------------------------------------------------
    # # phi
    # #   exclude x=0.0 (won't work in the regression)
    # # python analyze.py ../data/2014-06-24_SC,h-rnd,phi/2014-06-24_SC,h-rnd,phi-?\.[^0]_* ../data/2014-06-24_SC,h-rnd,phi/2014-06-24_SC,h-rnd,phi-[^0]\.?_*
    # x = np.array([0.1, 0.5, 1.0, 2.0, 4.0])[:len(st)]
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulation steps', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, r"""Link density $\phi$""", 'Messages', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=1)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # msgdelay
    # # python analyze.py ../data/2014-06-25_SC,h-rnd,msgdelay/*
    # x = np.arange(1, len(st) + 1)
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulation steps', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, 'Communication delay $d_{\mathrm{max}}$', 'Messages', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=4)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '', xticks=tests)

    # # -------------------------------------------------------------------------
    # # agentdelay
    # # python analyze.py ../data/2014-06-24_SC,h-rnd,agentdelay/*
    # x = np.arange(1, len(st) + 1)
    # r_sim_steps = fitting_plot(ax00, x, sim_steps, '', 'Simulation steps', tests, xlim=(x[0], x[-1]))
    # r_messages = fitting_plot(ax10, x, messages, 'Reaction delay $r_{\mathrm{max}}$', 'Messages', tests)
    # ax00.legend(fontsize='xx-small', handlelength=3.5, loc=4)
    # residuals_plot(ax01, tests, r_sim_steps, '', '')
    # residuals_plot(ax11, tests, r_messages, '', '', xticks=tests)


    plt.show()
