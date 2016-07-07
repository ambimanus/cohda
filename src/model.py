# coding=utf-8

"""
Data models, derived from the 2016 IJBIC paper (in press).

Each agent a_i maintains a working memory κ_i = (ζ, Ω_i, γ_i) that will be
exchanged with other agents. It comprises the target objective ζ, the believed
current configuration of the whole system Ω_i, and a solution candidate to the
optimization problem γ_i. (Note: The set of feasible contribution values S_i is
regarded as private information, and thus is not part of the working memory,
but is stored in the agent object). In more detail, Ω_i = {ω_j, ω_k, ..., }
denotes the believed set of current agent contributions in the system and thus
not only includes a_i’s own contribution, but also reflects the most up to date
information about other agents’ choices. An agent contribution ω_j = (a_j, θ_j,
λ_j) therein is a tuple comprising the identifier a_j of the agent this
contribution refers to, the actual selected (i.e., contributed) value θ_j ∈ S_j
of a_j at the point of time this tuple has been created, and the corresponding
value λ_j of an internal counting variable of a_j, that is increased whenever
the agent creates a new contribution. With this three-fold information,
contributions can uniquely be attributed to their respective agents, and can be
sorted by their creation order. Note that this does not introduce a globally
synchronized time into the system, because each agent a_i maintains its own
individual counting variable, allowing for sorting contributions with regard to
individual agents only. The remaining part of the working memory is γ_i = (a_x,
Ω, f), where Ω comprises a collection of contributions that a_i encountered
some time in the past. Instead of reflecting the current state of the system
(as in Ω_i ∈ κ_i), Ω ∈ γ_i is the best known combination of agent contributions
with respect to the target objective ζ the agent has encountered so far and
thus forms a solution candidate for the optimization problem. The identifier
a_x therein denotes the agent that originally created this solution candidate,
and f holds the fitness of this solution candidate, as derived from the
objective ζ.

Mapping from the textual representation (above) to the code:

κ --> Working_Memory
    ζ --> objective
    Ω --> configuration
        ω --> Agent_Contribution
            a --> aid
            θ --> value
            λ --> v_lambda
    γ --> Solution_Candidate
        a --> aid
        Ω --> configuration
            ω --> Agent_Contribution
                a --> aid
                θ --> value
                λ --> v_lambda
        f --> fitness


"""


class Agent_Contribution(object):

    def __init__(self, aid, value, v_lambda):
        """Initializes an agent contribution (ω).

        Arguments:
        aid         -- a -- agent identifier (str)
        value       -- θ -- the chosen contributed value (Value)
        v_lamdba    -- λ -- the selection counter (int)
        """
        self.aid = aid
        self.value = value
        self.v_lambda = v_lambda

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (other.aid == self.aid and
                    other.value == self.value and
                    other.v_lambda == self.v_lambda)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Solution_Candidate(object):

    def __init__(self, aid, configuration, fitness):
        """Initializes a solution candidate (γ).

        Arguments:
        aid             -- a -- agent identifier (str)
        configuration   -- Ω -- system configuration
                                    (dict(str: Agent_Contribution))
        fitness         -- f -- fitness of this solution candidate (float)
        """
        self.aid = aid
        self.configuration = dict(configuration)
        self.fitness = fitness

    def copy(self):
        return Solution_Candidate(self.aid, self.configuration, self.fitness)


class Working_Memory(object):

    def __init__(self, objective, configuration, solution_candidate):
        """Initializes an agent's working memory (κ).

        Arguments:
        objective           -- ζ -- one-argument objective function
                                        (obj --> float)
        configuration       -- Ω -- system configuration
                                        (dict(str: Agent_Contribution))
        solution_candidate  -- γ -- initial solution candidate
                                        (Solution_Candidate)
        """
        self.objective = objective
        self.configuration = dict(configuration)
        self.solution_candidate = solution_candidate.copy()

    def copy(self):
        return Working_Memory(self.objective, self.configuration,
                              self.solution_candidate.copy())


class Objective(object):

    def __init__(self, supported_types):
        """Initializes the objective function (ζ).

        Arguments:
        supported types -- a list of types this objective supports in _f(x)
        """
        self.supported_types = supported_types
        self.call_counter = 0

    def __call__(self, x, bypass_counter=False):
        """Allows calling this object like a function.

        On each call, an internal counting variable is increased. This allows
        evaluating system performance later on.

        The actual evaluation of x is delegated to _f(x).

        Arguments:
        x -- the argument to be evaluated (list of objects of supported_types)

        Keyword arguments:
        bypass_counter -- allows bypassing the counter (boolean, default False)

        Returns:
        fitness -- the fitness for x, calculated by _f(x) (float)
        """
        if not isinstance(x, list):
            x = [x]
        for v in x:
            assert any([isinstance(v, t) for t in self.supported_types]), \
                'Unsupported type: %s' % type(v)

        if not bypass_counter:
            self.call_counter += 1

        return self._f(x)

    def _f(self, x):
        """The actual evaluation of the argument x.

        This is a method stub. Has to be implemented by application.

        Arguments:
        x -- the argument to be evaluated (list of Value objects)

        Returns:
        fitness -- the fitness for x (float)
        """
        raise NotImplementedError('Application must implement _f(x).')

    def __repr__(self):
        return '%s (%d calls until now)' % (self.__class__, self.call_counter)


class Value(object):

    def __init__(self, v):
        """Initializes a value that can be contributed by an agent (θ).

        Arguments:
        v -- application-specific value (object)
        """
        self.v = v

    def __eq__(self, other):
        raise NotImplementedError('Subclass must implement __eq__.')

    def __ne__(self, other):
        raise NotImplementedError('Subclass must implement __ne__.')
