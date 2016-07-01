# coding=utf-8

import signal
from datetime import datetime as dt

from definitions import *
from logger import *
from stigspace import Stigspace
from util import PBar
from model import Working_Memory, Solution_Candidate


# Install signal handler for SIGINT
SIGINT_DETECTED = False


def sigint_detected(signal, frame):
    global SIGINT_DETECTED
    SIGINT_DETECTED = True

signal.signal(signal.SIGINT, sigint_detected)


class Simulator(object):
    def __init__(self, cfg, agents):
        self.cfg = cfg
        self.sc = cfg.scenario
        self.agents = agents
        self.current_time = 0
        self.messages = []
        self.agent_delays = {aid: 0 for aid in self.agents}

    def init(self):
        for a in self.agents.values():
            a.init(self)
        kappa = Working_Memory(self.sc[KW_OBJECTIVE], dict(),
                               Solution_Candidate(None, dict(), float('inf')))
        msg = {'sender': 'mas', 'kappa': kappa}
        if Stigspace.active:
            for a in self.agents.values():
                a.inbox.append(msg)
        else:
            speaker = (self.cfg.rnd.choice(self.agents.keys())
                       if self.cfg.random_speaker else 0)
            INFO('Notifying speaker (a%s)' % speaker)
            self.agents[speaker].inbox.append(msg)
        # objective._reset_call_counter()

    def step(self):
        ts = dt.now()
        progress = None
        counter = 0
        amount = self.sc[KW_OPT_M] + len(self.messages)

        # transfer messages
        delayed_messages = []
        for delay, msg in self.messages:
            if delay <= 1:
                self.agents[msg['receiver']].inbox.append(msg)
            else:
                delayed_messages.append((delay - 1, msg))
            counter += 1
            if progress is None and (dt.now() - ts).seconds >= 1:
                progress = PBar(amount).start()
            if progress is not None:
                progress.update(counter)
        self.messages = delayed_messages

        # activate agents
        for aid in self.agents:
            a = self.agents[aid]
            delay = self.agent_delays[aid]
            if delay <= 1:
                a.step()
                self.agent_delays[aid] = self.agentdelay()
            else:
                self.agent_delays[aid] = delay - 1
            counter += 1
            if progress is None and (dt.now() - ts).seconds >= 1:
                progress = PBar(amount).start()
            if progress is not None:
                progress.update(counter)
        self.current_time += 1

    def msgdelay(self):
        return self.cfg.rnd.randint(self.cfg.msg_delay_min,
                                    self.cfg.msg_delay_max)

    def agentdelay(self):
        return self.cfg.rnd.randint(self.cfg.agent_delay_min,
                                    self.cfg.agent_delay_max)

    def msg(self, msg):
        msg_counter()
        delay = self.msgdelay()
        MSG('a%s --(%d)--> a%s' % (msg['sender'], delay, msg['receiver']))
        self.messages.append((delay, msg))

    def is_active(self):
        # Check maximal runtime
        # WARNING: This will stop the process in an unconverged state, so a
        #          post-processing step to announce the current bkc in all
        #          agents should be implemented.
        if (self.cfg.max_simulation_steps and
                self.current_time >= self.cfg.max_simulation_steps):
            INFO('Stopping (max simulation steps reached)')
            return False
        # Check user interrupt
        if SIGINT_DETECTED:
            INFO('Stopping due to user request (SIGINT detected)')
            return False

        # Check messages
        if len(self.messages) > 0:
            return True

        # Check agent activity
        for a in self.agents.values():
            if len(a.inbox) > 0:
                return True

        INFO('Stopping (no activity)')
        return False
