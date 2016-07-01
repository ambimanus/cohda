# coding=utf-8

import numpy as np

from logger import *
from model import Schedule_Selection, Solution_Candidate, Working_Memory


class Agent():

    def __init__(self, aid, search_space, initial_value=None):
        """Initializes an agent for the COHDA algorithm.

        Arguments:
        aid -- agent identifier (str)
        objective -- one-argument objective function (obj --> float)
        search_space -- search-space (ndarray(2))

        Keyword arguments:
        initial_value -- initially chosen value (ndarray, default None)

        The initial value will be chosen randomly from the search_space, if
        None is given.
        """
        self.aid = aid
        self.neighbors = dict()
        self.inbox = []

        self.search_space = search_space
        if initial_value is None:
            AGENT('Warning: No initial schedule defined, setting random one.')
            initial_value = self.rnd.choice(self.search_space)

        schedule_selection = Schedule_Selection(self.aid, initial_value, 0)
        configuration = {self.aid: schedule_selection}
        solution_candidate = Solution_Candidate(self.aid, configuration,
                                                float('inf'))
        self.kappa = Working_Memory(None, configuration, solution_candidate)

    def init(self, mas):
        """Sets the multi-agent system, used for messaging."""
        self.mas = mas

    def step(self):
        """The agent follows the classical perceive--decide--act behavior."""

        # Termination criterion
        dirty = False

        # Perceive
        for msg in self.inbox:
            sender = msg['sender']
            # Check message
            AGENT(self.aid, 'message from %s' % sender)
            if not sender in self.neighbors:
                INFO(self.aid,
                     'Sender (%s) not in neighbours, assuming system message.'
                     % sender)
            # Update Zeta (the objective)
            if self.kappa.objective is None:
                self.kappa.objective = msg['kappa'].objective
                dirty = True
            # Update Omega (the system configuration)
            omega = msg['kappa'].configuration
            for aid in omega.keys():
                if (aid not in self.kappa.configuration or
                        self.kappa.configuration[aid].v_lambda <
                        omega[aid].v_lambda):
                    self.kappa.configuration[aid] = omega[aid]
                    dirty = True
            # Update gamma (the solution candidate)
            gamma_own = self.kappa.solution_candidate
            gamma_other = msg['kappa'].solution_candidate
            # There are different cases to consider:
            # 1) gamma_other is larger and all keys from gamma_own are already
            #    in gamma_other
            #       -> replace gamma_own by gamma_other
            # 2) there is at least one key in gamma_other which is not in
            #    gamma_own
            #       -> merge both gammas (TODO: in which direction?)
            # 3) both gammas have equal keys, but gamma_own has a worse rating
            #       -> replace gamma_own by gamma_other
            # 4) both gammas have equal keys and equal rating, but the
            #    creator-id of gamma_own is larger
            #       -> replace gamma_own by gamma_other
            # 5) every other case
            #    (e.g., gamma_own is larger and no new keys in gamma_other)
            #       -> do nothing
            keys_own = set(gamma_own.configuration.keys())
            keys_other = set(gamma_other.configuration.keys())
            keys_in = keys_other - keys_own
            keys_out = keys_own - keys_other
            assert self.aid not in keys_in
            if keys_own != keys_other:
                # Need only process incoming items
                if len(keys_in) > 0:
                    if len(keys_out) == 0:
                        # Case 1: gamma_own is a subset of gamma_other
                        self.kappa.solution_candidate = gamma_other.copy()
                    else:
                        # Case 2: gamma_other is different, import new items
                        self.kappa.solution_candidate.configuration.update(
                            {aid: gamma_other.configuration[aid]
                             for aid in keys_in})
                    dirty = True
            else:
                # We are evaluating a minimization problem here, so 'less' is
                # 'better'
                if gamma_own.rating > gamma_other.rating:
                    # Case 3: gamma_other is better
                    self.kappa.solution_candidate = gamma_other.copy()
                    dirty = True
                elif (gamma_own.rating == gamma_other.rating and
                      gamma_own.aid > gamma_other.aid):
                    # Case 4: gammas are equally good, but aid of gamma_other
                    # is smaller
                    self.kappa.solution_candidate = gamma_other.copy()
                    dirty = True
        self.inbox = []

        # Decide
        if dirty:
            # Find best own schedule selection for zeta (the objective) with
            # respect to Omega (the current system configuration)
            solution_others = np.array([self.kappa.configuration[aid].schedule
                                        for aid in self.kappa.configuration
                                        if aid != self.aid]).sum(axis=0)
            schedule_best, rating = None, float('inf')
            for schedule in self.search_space:
                # Check for feasibility
                if not self._is_feasible(schedule):
                    continue
                # Check for rating
                r = self.kappa.objective(solution_others + schedule)
                if schedule is None or r < rating:
                    schedule_best, rating = schedule, r
            # TODO: Use search space model or heuristic to select a schedule?
            # Compare resulting configuration to existing configuration
            configuration = dict(self.kappa.configuration)
            configuration[self.aid] = Schedule_Selection(
                self.aid,
                schedule_best,
                self.kappa.configuration[self.aid].v_lambda + 1)
            if rating < self.kappa.solution_candidate.rating:
                # Resulting configuration is better, a new solution candidate
                # has been found.
                solution_candidate = Solution_Candidate(self.aid,
                                                        configuration,
                                                        rating)
                self.kappa.solution_candidate = solution_candidate
                self.kappa.configuration = configuration
            elif np.any(self.kappa.solution_candidate.configuration[self.aid]
                            .schedule !=
                        self.kappa.configuration[self.aid].schedule):
                # Resulting configuration is not better, revert own schedule to
                # the one from the existing solution candidate.
                # Attention: Although the same schedule had been selected
                # somewhere in the past already, this constitutes a new
                # selection for this agent right now. So the lambda value has
                # to be increased.
                self.kappa.configuration[self.aid] = Schedule_Selection(
                    self.aid,
                    self.kappa.solution_candidate.configuration[self.aid]
                        .schedule,
                    self.kappa.configuration[self.aid].v_lambda + 1)

        # Act
        if dirty:
            for aid in self.neighbors:
                self.mas.msg({'sender': self.aid, 'receiver': aid,
                              'kappa': self.kappa.copy()})

    def set_p_refuse(self, p_refuse, seed):
        import random
        self.rnd = random.Random(seed)
        self.p_refuse = p_refuse

    def _is_feasible(self, sol):
        # If feasibility check not desired, return immediately:
        if not hasattr(self, 'rnd'):
            return True
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
