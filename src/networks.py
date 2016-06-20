# coding=utf-8

from __future__ import division

from collections import OrderedDict as dict

from definitions import *


def unconnected(params, network=None):
    return dict({n: list() for n in params[KW_AGENT_IDS]})


def sequence(params, network=None):
    ids = params[KW_AGENT_IDS]
    if network is None:
        network = dict({n: list() for n in ids})
    for i in range(len(ids)):
        for k in range(1, params[KW_NETWORK_K] + 1):
            if i + k < len(ids):
                network[ids[i]].append(ids[i + k])
                network[ids[i + k]].append(ids[i])
    return network


def ring(params, network=None):
    ids = params[KW_AGENT_IDS]
    if network is None:
        network = dict({n: list() for n in ids})
    for i in range(len(ids)):
        for k in range(1, min(len(ids), params[KW_NETWORK_K] + 1)):
            node = (i + k) % len(ids)
            network[ids[i]].append(ids[node])
            network[ids[node]].append(ids[i])
    return network


def stigspace(params, network=None):
    ids = params[KW_AGENT_IDS]
    if network is None:
        network = dict({n: list() for n in ids})
    # Directed ring with k=1
    for i in range(len(ids)):
        network[ids[i]].append(ids[(i + 1) % len(ids)])
    return network


def full(params, network=None):
    ids = params[KW_AGENT_IDS]
    if network is None:
        network = dict()
    for i in range(len(ids)):
        network[ids[i]] = [ids[x] for x in range(len(ids)) if x != i]
    return network


def half(params, network=None):
    ids = params[KW_AGENT_IDS]
    if network is None:
        network = dict({n: list() for n in ids})
    for n1 in ids[:len(ids)/2]:
        for n2 in ids[len(ids)/2:]:
            network[n1].append(n2)
            network[n2].append(n1)
    return network


def mesh_rect(params, network=None):
    from math import floor, sqrt
    ids = params[KW_AGENT_IDS]
    if network is None:
        network = dict()
    s = int(floor(sqrt(len(ids))))
    for i in range(len(ids)):
        d = list()
        if i - s >= 0:
            d.append(ids[i - s])      # Node above i
        if i + s < len(ids):
            d.append(ids[i + s])      # Node below i
        if i % s > 0 and i > 0:
            d.append(ids[i - 1])      # Node left from i
        if (i + 1) % s > 0 and i + 1 < len(ids):
            d.append(ids[i + 1])      # Node right from i
        network[ids[i]] = d
    return network


def random(params, network=None):
    ids, rnd = params[KW_AGENT_IDS], params[KW_RND]
    if network is None:
        sids = list(ids)
        rnd.shuffle(sids)
        par = dict(params)
        par[KW_AGENT_IDS] = sids
        network = sequence(par)
    c = params[KW_NETWORK_C]
    for n in ids:
        if len(network[n]) >= c:
            continue
        cand = list(ids)
        rnd.shuffle(cand)
        for cnd in cand:
            if len(network[n]) >= c:
                break
            if (cnd == n or cnd in network[n] or
                    len(network[cnd]) >= c):
                continue
            network[n].append(cnd)
            network[cnd].append(n)
    return network


def smallworld(params, network=None):
    # First create a k-neighbour-ring
    network = ring(params, network=network)
    # Create len(ids)*k*phi random shortcuts
    ids, k, phi = params[KW_AGENT_IDS], params[KW_NETWORK_K], params[KW_NETWORK_PHI]
    for i in range(int(len(ids) * k * phi)):
        subset = params[KW_RND].sample(ids, 2)
        if not subset[1] in network[subset[0]]:
            network[subset[0]].append(subset[1])
            network[subset[1]].append(subset[0])
    return network

