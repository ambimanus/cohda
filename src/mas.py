# coding=utf-8

import signal
from datetime import datetime as dt

from logger import *
from util import PBar


# Install signal handler for SIGINT
SIGINT_DETECTED = False


def sigint_detected(signal, frame):
    global SIGINT_DETECTED
    SIGINT_DETECTED = True

signal.signal(signal.SIGINT, sigint_detected)


class Mas(object):
    def __init__(self, sc, agents):
        self.aid = 'mas'
        self.sc = sc
        self.agents = agents
        self.current_time = 0
        self.messages = []
        self.agent_delays = {aid: 0 for aid in self.agents}
        self.message_counter = 0
        for a in self.agents.values():
            a.set_mas(self)

    def step(self):
        # transfer messages
        ts = dt.now()
        progress = None
        counter = 0
        amount = len(self.messages)
        delayed_messages = []
        for delay, msg in self.messages:
            if delay <= 1:
                self.agents[msg.receiver].inbox.append(msg)
            else:
                delayed_messages.append((delay - 1, msg))
            counter += 1
            if progress is None and (dt.now() - ts).seconds >= 1:
                progress = PBar(amount).start()
            if progress is not None:
                progress.update(counter)
        if progress is not None:
                progress.finish()
        self.messages = delayed_messages

        # activate agents
        ts = dt.now()
        progress = None
        counter = 0
        amount = len([k for k in self.agents if self.agent_delays[k] > 1])
        for aid in self.agents:
            a = self.agents[aid]
            delay = self.agent_delays[aid]
            if delay <= 1:
                a.step()
                self.agent_delays[aid] = self.agentdelay()
                counter += 1
                if progress is None and (dt.now() - ts).seconds >= 1:
                    progress = PBar(amount).start()
                if progress is not None:
                    progress.update(counter)
            else:
                self.agent_delays[aid] = delay - 1
        if progress is not None:
                progress.finish()

        # Update simulation time
        self.current_time += 1

    def msgdelay(self):
        return self.sc.rnd.randint(self.sc.sim_msg_delay_min,
                                   self.sc.sim_msg_delay_max)

    def agentdelay(self):
        return self.sc.rnd.randint(self.sc.sim_agent_delay_min,
                                   self.sc.sim_agent_delay_max)

    def msg(self, msg):
        self.message_counter += 1
        delay = self.msgdelay()
        MSG('%s --(%d)--> %s' % (msg.sender, delay, msg.receiver))
        self.messages.append((delay, msg))

    def is_active(self):
        # Check maximal runtime
        # WARNING: This will stop the process in an unconverged state, so a
        #          post-processing step to announce the current bkc in all
        #          agents should be implemented.
        if (self.sc.sim_max_steps and
                self.current_time >= self.sc.sim_max_steps):
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


class Message():

    def __init__(self, sender, receiver, payload):
        """Initializes a message that can be sent via the MAS.

        Arguments:
        sender      -- aid of the sender (str)
        receiver    -- aid of the receiver (str)
        payload     -- data to be sent (object)
        """
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
