# coding=utf-8

import logging
from datetime import datetime as dt

from definitions import *
from logger import *
from stigspace import Stigspace
from util import PBar


class Simulator(object):
    def __init__(self, cfg, agents):
        self.cfg = cfg
        self.sc = cfg.scenario
        self.agents = agents
        self.current_time = 0
        self.messages = []

    def init(self):
        for a in self.agents.values():
            a.init(self)
        speaker = (self.cfg.rnd.choice(self.agents.keys())
                   if self.cfg.random_speaker else 0)
        INFO('Notifying speaker (a%d)' % speaker)
        objective = self.sc[KW_OBJECTIVE]
        if Stigspace.active:
            for a in self.agents.values():
                a.notify(objective)
        else:
            self.agents[speaker].notify(objective)
        objective._reset_call_counter()


    def step(self):
        ts = dt.now()
        progress = None
        counter = 0

        # transfer messages
        for receiver, msg in self.messages:
            self.agents[receiver].update(msg)
            counter += 1
            if progress is None and (dt.now() - ts).seconds >= 1:
                progress = PBar(self.sc[KW_OPT_M] + len(self.messages)).start()
            if progress is not None:
                progress.update(counter)
        self.messages = []

        # activate agents
        for a in self.agents.values():
            a.step()
            counter += 1
            if progress is None and (dt.now() - ts).seconds >= 1:
                progress = PBar(self.sc[KW_OPT_M] + len(self.messages)).start()
            if progress is not None:
                progress.update(counter)
        self.current_time += 1


    def msg(self, sender, receiver, msg):
        msg_counter()
        MSG('a%d ---> a%d' % (sender, receiver))
        self.messages.append((receiver, msg))


    def is_active(self):
        # Check maximal runtime
        # WARNING: This will stop the process in an unconverged state, so a
        #          post-processing step to announce the current bkc in all
        #          agents should be implemented.
        if (self.cfg.max_simulation_steps and
                self.current_time >= self.cfg.max_simulation_steps):
            INFO('Stopping (max simulation steps reached)')
            return False

        # Check messages
        if len(self.messages) > 0:
            return True

        # Check agent activity
        for a in self.agents.values():
            if a.dirty:
                return True

        INFO('Stopping (no activity)')
        return False
