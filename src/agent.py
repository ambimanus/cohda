# coding=utf-8

import numpy as np

from logger import *
from model import Agent_Contribution, Solution_Candidate, Working_Memory
from mas import Message


class Agent(object):
    """COHDA agent for solving distributed combinatorial optimization problems.

    Let ζ be a global objective function, and let A be a set of agents. Each
    agent a_i ∈ A has a set of possible contribution values S_i, and each agent
    may contribute an arbitrary value θ_i ∈ S_i. The goal is to find a
    combination of values such that ζ is maximized:

        ζ(θ_i, θ_j, θ_k, ...) → MAX

    Usually, a valid solution for the given optimization problem must contain
    *exactly* one value θ_i for every agent a_i ∈ A (no more, no less). An
    agent a_i only knows its own set of possible contribution values S_i. From
    an algorithmic point of view, the difficulty of the problem is given by the
    distributed nature of the system in contrast to the task of finding a
    common allocation of values for a global objective function.

    COHDA is a heuristic approach. To favor an asynchronous exploration of the
    combinatorial search space of the given optimization problem, the
    simulation core (i.e., MAS) must ensure the following aspects:

        - Each agent a_i ∈ A has a defined neighborhood comprising a non-empty
          subset of other agents N_i ⊂ A. An agent a_i may only communicate
          with its neighbors a_j ∈ N_i. This is usually achieved by using an
          overlay communication topology, e.g. a small-world topology.

        - The agent is woken up (by calling its step() method) at random (but
          finite) intervals by the MAS.

        - Regarding communication, message transfer durations in the
          communication network are finite (i.e., no message loss occurs).
          Otherwise, convergence of the approach cannot be guaranteed.

        - [Optional] Message transfer durations may be varied randomly by the
          MAS.

        - When an agent receives a message, the message is stored in the
          agent's inbox until the agent is woken up next time. When the agent
          is woken up, it may process all of its intermediately received
          messages.

    Based on this, the agents coordinate by updating and exchanging their
    working memories κ (see model.py for a definition of κ). In particular, the
    key concept of COHDA is an asynchronous iterative approximate best-response
    behaviour, where each agent a_i ∈ A reacts to updated information from
    other agents by adapting its own contribution θ_i with respect to the
    target objective ζ. The process is initiated by sending a system message
    containing just the objective function to an arbitrary agent. Subsequently,
    each agent will execute a classical perceive--decide--act behavior as
    follows:

        1 (perceive): When an agent a_i becomes active, it first processes all
        received messages from its inbox. For each received message κ_j = (ζ,
        Ω_j, γ_j) from one of its neighbours (say, aj), it imports the contents
        of this message into its own working memory by storing ζ if not already
        known, then updating Ω_i with new contributions from Ω_j and, finally,
        replacing γ_i with γ_j if the latter contains more elements or yields a
        better fitness regarding ζ.

        2 (decide): The agent then searches S_i for the best value θ regarding
        the updated system state Ω_i and ζ. If θ yields a better fitness (with
        respect to Ω_i and ζ) than the current solution candidate, a new
        solution candidate is created. Otherwise, the current solution
        candidate still reflects the best solution the agent is aware of, and
        the agent reverts to its contribution that is stored in the solution
        candidate.

        3 (act): If (and only if) any component of the working memory κ_i has
        been modified in one of the previous steps, the agent publishes κ_i to
        its neighbours.

    Following this behaviour, for each agent a_i, its observed system
    configuration Ω_i as well as solution candidate γ_i are empty at the
    beginning, will be filled successively with the ongoing message exchange
    and will some time later represent valid solutions for the given
    optimization problem. After producing some intermediate solutions, the
    heuristic eventually terminates in a state where for all agents the working
    memories κ are identical. At this point, Ω ∈ γ (which is the same for all
    agents then, so the index can be dropped) is the final solution of the
    heuristic and contains exactly one contribution for each agent. In summary,
    as better solution candidates always prevail over inferior solution
    candidates, the system always terminates in an at least local optimum.
    Moreover, due to the asynchronous search in the solution space, inferior
    local optima are discarded continuously. Therefore, the approach is able to
    find near-optimal solutions of a given problem quite easily.

    """

    def __init__(self, aid, search_space, initial_value=None):
        """Initializes an agent for the COHDA algorithm.

        Arguments:
        aid             -- a -- agent identifier (str)
        search_space    -- S -- search-space (iterable)

        Keyword arguments:
        initial_value   -- θ -- initially chosen value (Value, default None)

        The initial value will be chosen randomly from the search_space, if
        None is given.
        """
        self.aid = aid
        self.neighbors = dict()
        self.inbox = []

        self.search_space = search_space
        if initial_value is None:
            AGENT('Warning: No initial value defined, setting random one.')
            initial_value = self.rnd.choice(self.search_space)

        agent_contribution = Agent_Contribution(self.aid, initial_value, 0)
        configuration = {self.aid: agent_contribution}
        solution_candidate = Solution_Candidate(self.aid, configuration,
                                                float('-inf'))
        self.kappa = Working_Memory(None, configuration, solution_candidate)

    def set_mas(self, mas):
        """Sets the multi-agent system (MAS), used for messaging."""
        self.mas = mas

    def step(self):
        """The agent follows a classical perceive--decide--act behavior."""

        # Termination criterion
        dirty = False

        # Perceive
        for msg in self.inbox:
            # Check message
            AGENT(self.aid, 'message from %s' % msg.sender)
            if msg.sender not in self.neighbors and msg.sender != self.mas.aid:
                ERROR(self.aid, 'Unknown sender (%s)' % msg.sender)
            assert isinstance(msg.payload, Working_Memory)
            # Update Zeta (the objective)
            if self.kappa.objective is None:
                self.kappa.objective = msg.payload.objective
                dirty = True
            # Update Omega (the system configuration)
            omega = msg.payload.configuration
            for aid in omega.keys():
                if (aid not in self.kappa.configuration or
                        self.kappa.configuration[aid].v_lambda <
                        omega[aid].v_lambda):
                    self.kappa.configuration[aid] = omega[aid]
                    dirty = True
            # Update gamma (the solution candidate)
            gamma_own = self.kappa.solution_candidate
            gamma_other = msg.payload.solution_candidate
            # There are different cases to consider:
            # 1) gamma_other is larger and all keys from gamma_own are already
            #    in gamma_other
            #       -> replace gamma_own by gamma_other
            # 2) there is at least one key in gamma_other which is not in
            #    gamma_own
            #       -> merge both gammas (TODO: in which direction?)
            # 3) both gammas have equal keys, but gamma_own has a worse fitness
            #       -> replace gamma_own by gamma_other
            # 4) both gammas have equal keys and equal fitness, but the
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
                # We are evaluating a maximization problem here, so 'larger' is
                # 'better'
                if gamma_own.fitness < gamma_other.fitness:
                    # Case 3: gamma_other is better
                    self.kappa.solution_candidate = gamma_other.copy()
                    dirty = True
                elif (gamma_own.fitness == gamma_other.fitness and
                      gamma_own.aid > gamma_other.aid):
                    # Case 4: gammas are equally good, but aid of gamma_other
                    # is smaller
                    self.kappa.solution_candidate = gamma_other.copy()
                    dirty = True
        # Clear inbox
        self.inbox = []

        # Decide
        if dirty:
            # Find best own value for zeta (the objective) with respect to
            # Omega (the current system configuration)
            # TODO: Use search space model or heuristic to select a value?
            solution_others = [self.kappa.configuration[aid].value
                               for aid in self.kappa.configuration
                               if aid != self.aid]
            value_best, fitness = None, float('-inf')
            for value in self.search_space:
                # Check for fitness
                f = self.kappa.objective(solution_others + [value])
                if value_best is None or f > fitness:
                    value_best, fitness = value, f
            # Compare resulting configuration to existing configuration
            configuration = dict(self.kappa.configuration)
            configuration[self.aid] = Agent_Contribution(
                self.aid,
                value_best,
                self.kappa.configuration[self.aid].v_lambda + 1)
            if f > self.kappa.solution_candidate.fitness:
                # Resulting configuration is better, a new solution candidate
                # has been found.
                solution_candidate = Solution_Candidate(self.aid,
                                                        configuration,
                                                        fitness)
                self.kappa.solution_candidate = solution_candidate
                self.kappa.configuration = configuration
            elif (self.kappa.solution_candidate.configuration[self.aid].value !=
                  self.kappa.configuration[self.aid].value):
                # Resulting configuration is not better, revert own value to
                # the one from the existing solution candidate.
                # Attention: Although the same value had been selected
                # somewhere in the past already, this constitutes a new
                # selection for this agent right now. So the lambda value has
                # to be increased.
                self.kappa.configuration[self.aid] = Agent_Contribution(
                    self.aid,
                    self.kappa.solution_candidate.configuration[self.aid]
                        .value,
                    self.kappa.configuration[self.aid].v_lambda + 1)

        # Act
        if dirty:
            if not hasattr(self, 'mas') or self.mas is None:
                ERROR(self.aid, 'MAS not set, cannot send messages')
            else:
                for aid in self.neighbors:
                    self.mas.msg(Message(self.aid, aid, self.kappa.copy()))
