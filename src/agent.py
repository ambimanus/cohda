# coding=utf-8

import random

import numpy as np

from logger import *


class Agent():
    def __init__(self, aid, opt_w, sol_init, timeout, seed, p_refuse):
        # ActorMixin
        self.aid = aid
        self.timeout = timeout
        self.current_timeout = timeout
        self.rnd = random.Random(seed)
        self.p_refuse = p_refuse
        self.dirty = False
        # PeerMixin
        self.peers = dict()
        # MMMSSPMixin
        self.opt_w = opt_w
        self.sol_init = sol_init
        self.sol = sol_init
        self.sol_f = None
        self.objective = None
        # AgentGossip
        self.sol_counter = 0
        self.gossip_updates = dict()
        self.gossip_updates_inbox = dict()
        self.gossip_storage = dict()
        # AgentGossipBKC
        self.bkc = dict()
        self.bkc_creator = self.aid
        self.bkc_f = None

        self.sim = None

    def init(self, sim):
        self.sim = sim

    def step(self):
        if self.current_timeout <= 1:
            # become active
            if self.objective is not None and self.dirty:
                AGENT(self.aid, 'entering run()')
                self.run()
            else:
                AGENT(self.aid, 'no run() required, going to sleep now.')
            # reset wait counter
            self.current_timeout = self.timeout
        else:
            # wait
            self.current_timeout -= 1

    def notify_peers(self):
        AGENTV(self.aid, 'notifying peers')
        for k in sorted(self.peers.keys()):
            self.sim.msg(self.aid, k, (self.aid,
                                       self.objective,
                                       dict(self.gossip_updates),
                                       dict(self.bkc),
                                       self.bkc_creator))
        AGENTV(self.aid, 'Clearing gossip_updates')
        self.gossip_updates.clear()

    def add_peer(self, aid, peer):
        # Consistency check
        assert aid != self.aid, 'cannot add myself as peer!'
        # Store peer
        if aid not in self.peers.keys():
            self.peers[aid] = peer
            AGENTV(self.aid, 'added', aid, 'as peer')
        else:
            AGENTV(self.aid, aid, 'is already connected')

    def choose_solution(self, local_sol):
        assert self.objective is not None
        if local_sol is None:
            local_sol = 0
        AGENTV(self.aid, 'choose_solution() for local solution:', local_sol)
        # Find best own solution for given local solution
        sol, distance = None, None
        for cand in self.opt_w:
            if not self.is_feasible(cand):
                continue
            d = self.objective(local_sol + cand)
            if distance is None or d < distance:
                distance = d
                sol = cand
        return sol, distance

    def is_feasible(self, sol):
        # The given solution isn't actually checked for feasibility in this
        # simplified implementation.
        # Instead, accept the solution with a given propability:
        masked = False
        if hasattr(sol, 'mask') and sol.mask.all():
            masked = True
        # use 1-rnd here to convert [0.0, 1.0) to (0.0, 1.0]
        if not masked and 1 - self.rnd.random() > self.p_refuse:
            return True
        return False

    def update(self, msg):
        aid, objective, gossip_updates, bkc, bkc_creator = msg
        AGENT(self.aid, 'update from %s' % aid)
        # Agent
        if not aid in self.peers:
            WARNING(self.aid, 'got update from unknown sender', aid)
            # return

        # Update objective
        if (self.objective is None or
                self.objective.instance < objective.instance):
            self.notify(objective)

        # AgentGossip
        AGENT(self.aid, 'receiving gossip_updates_inbox:', gossip_updates)
        for k, v in gossip_updates.items():
            if (k != self.aid and
                    (k not in self.gossip_storage or
                     self.gossip_storage[k][KW_SOL_COUNTER] <
                        v[KW_SOL_COUNTER]) and
                    (k not in self.gossip_updates_inbox or
                     self.gossip_updates_inbox[k][KW_SOL_COUNTER] <
                        v[KW_SOL_COUNTER])):
                AGENTV(self.aid, 'adding', k, 'to gossip_updates_inbox')
                self.gossip_updates_inbox[k] = v
                # Mark myself as dirty
                self.dirty = True
        AGENT(self.aid, 'updated gossip_updates_inbox:',
              self.gossip_updates_inbox)

        # AgentGossipBKC
        bkc_f = self.objective(np.array(bkc.values()))
        AGENTV(self.aid, 'from %s: f(bkc_%d)' % (aid, bkc_creator))

        # For bkc vs. self.bkc, there are different cases to consider:
        # 1) given bkc is created by myself
        #       -> do nothing
        # 2) given bkc is larger and all keys from self.bkc are already in
        #    given bkc:
        #       -> replace self.bkc by bkc
        # 3) there is at least one key in bkc which is not in self.bkc
        #       -> merge both bkc's (TODO: in which direction?)
        # 4) both bkc's have equal keys, but self.bkc has a worse rating
        #       -> replace self.bkc by bkc
        # 5) both bkc's have equal keys and equal rating, but the creator-id of
        #    self.bkc is larger
        #       -> replace self.bkc by bkc
        # 6) every other case
        #    (i.e.: self.bkc is larger and no new keys in given bkc)
        #       -> do nothing
        set_own, set_other = set(self.bkc.keys()), set(bkc.keys())
        in_keys = set_other - set_own
        out_keys = set_own - set_other
        if bkc_creator == self.aid:
            # Case 1: do nothing
            AGENTV(self.aid, 'given bkc was created by myself --> ignore it')
            pass
        elif len(in_keys) > 0:
            assert self.aid not in in_keys  # this should not happen
            if len(out_keys) == 0:
                # Case 2: replace bkc
                AGENTV(self.aid, 'replacing BKC')
                self.bkc = bkc
                self.bkc_f = bkc_f
                self.bkc_creator = bkc_creator
            else:
                # Case 3: merge bkc
                AGENTV(self.aid, 'keys are different, updating BKC')
                # Add every newly discovered item in bkc to self.bkc
                for k in in_keys:
                    self.bkc[k] = bkc[k]
                self.bkc_creator = self.aid
                self.bkc_f = self.objective(np.array(self.bkc.values()))
                AGENTV(self.aid, 'rating new BKC as', self.bkc_f)
            # Mark myself as dirty
            self.dirty = True
        elif set_own == set_other:
            AGENTV(self.aid, 'keys are equal, compare BKC ratings: given bkc_f =',
                   bkc_f, 'by', bkc_creator, ', self.bkc_f =', self.bkc_f,
                   'by', self.bkc_creator)
            if (self.bkc_f > bkc_f or
                    (self.bkc_f == bkc_f and self.bkc_creator > bkc_creator)):
                # Cases 4 + 5: replace bkc
                AGENT(self.aid, 'replacing BKC')
                self.bkc = bkc
                self.bkc_f = bkc_f
                self.bkc_creator = bkc_creator
                # Mark myself as dirty
                self.dirty = True
        else:
            AGENTV(self.aid, 'default case: self.bkc has', self.bkc.keys(),
                   'f =', self.bkc_f, ', given bkc has', bkc.keys(), 'f =',
                   bkc_f)
        AGENTV(self.aid, 'dirty =', self.dirty)

    def notify(self, objective):
        self.objective = objective
        AGENT(self.aid, 'notify() - got new objective')
        # Mark myself as dirty
        self.dirty = True
        # Calculate initial ratings etc.
        current_sol = self.get_current_sol()
        rating = self.objective(current_sol + self.sol)
        self.sol_f = rating
        self.new_bkc(self.sol, rating)
        # Prepare first gossip_updates
        self.gossip_updates[self.aid] = dict()
        self.gossip_updates[self.aid][KW_SOL] = self.sol
        self.gossip_updates[self.aid][KW_SOL_COUNTER] = self.sol_counter
        AGENTV(self.aid, 'initialized gossip_updates:', self.gossip_updates)
        # AgentGossipBKC
        if len(self.bkc) > 0:
            # Recalculate bkc_f
            self.bkc_f = self.objective(np.array(self.bkc.values()))

    def run(self):
        AGENT(self.aid, 'run')
        self.dirty = False
        # update knowledge
        self.gossip_storage.update(self.gossip_updates_inbox)
        self.gossip_updates.update(self.gossip_updates_inbox)
        self.gossip_updates_inbox.clear()
        # optimize
        current_sol = self.get_current_sol()
        AGENT(self.aid, 'overlayed gossip solution:', current_sol)
        sol, distance = self.choose_solution(current_sol)
        if sol is None:
            AGENT(self.aid, 'no feasible solution candidate found!')
            if len(self.gossip_updates) > 0:
                # Do nothing, just publish gossip updates
                AGENT(self.aid, 'sending gossip updates:', self.gossip_updates)
                self.notify_peers()
        else:
            AGENT(self.aid, 'solution candidate:', sol, 'with distance', distance)
            if (self.sol_f is None or
                    (any(sol != self.sol) and
                        (len(self.bkc) < len(self.gossip_storage) or
                         distance < self.bkc_f))):
                # Newly chosen solution will improve BKC rating
                AGENTV(self.aid, 'decision: new solution, new bkc')
                self.set_solution(sol, distance)
            elif (len(self.bkc) > 0 and
                  any(self.bkc[self.aid] != self.sol)):
                # Revert to BKC solution
                AGENTV(self.aid, 'decision: reverting to bkc solution')
                r = self.objective(current_sol + self.bkc[self.aid])
                self.set_solution(self.bkc[self.aid], r, new_bkc=False)
            elif len(self.gossip_updates) > 0:
                # Do nothing, just publish gossip updates
                AGENT(self.aid, 'decision: just sending gossip updates:', self.gossip_updates)
                self.notify_peers()
        AGENTV(self.aid, 'run finished, sol = %s, bkc = %s' % (self.sol, self.bkc))

    def get_current_sol(self):
        # Overlay known solutions from gossiping
        current_sol = np.zeros(len(self.objective.target))
        assert self.aid not in self.gossip_storage
        for k, v in self.gossip_storage.items():
            current_sol += v[KW_SOL]
        return current_sol

    def set_solution(self, sol, rating, new_bkc=True):
        self.sol_counter += 1
        # AgentGossipBKC
        if new_bkc:
            self.new_bkc(sol, rating)
        # AgentGossip
        if self.aid not in self.gossip_updates:
            self.gossip_updates[self.aid] = dict()
        self.gossip_updates[self.aid][KW_SOL] = sol
        self.gossip_updates[self.aid][KW_SOL_COUNTER] = self.sol_counter
        AGENT(self.aid, 'sending gossip updates:', self.gossip_updates)
        # Agent
        self.sol = sol
        self.sol_f = rating
        AGENT(self.aid, 'choosing solution:', self.sol)
        if len(self.peers) > 0:
            self.notify_peers()

    def new_bkc(self, sol, rating):
        d = dict()
        for k in self.gossip_storage:
            d[k] = self.gossip_storage[k][KW_SOL]
        d[self.aid] = sol
        self.bkc_creator = self.aid
        self.bkc = d
        self.bkc_f = rating
        AGENT(self.aid, 'created new BKC with rating', rating)
