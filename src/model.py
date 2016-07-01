# coding=utf-8

import numpy as np


"""
Data models, directly corresponding to the formal model from the 2016 IJBIC
paper (in press).
"""


class Schedule_Selection():

    def __init__(self, aid, schedule, v_lambda):
        """Initializes a schedule selection.

        Arguments:
        aid -- agent identifier (str)
        schedule -- the selected schedule (ndarray)
        v_lamdba -- the selection counter (int)
        """
        self.aid = aid
        self.schedule = schedule
        self.v_lambda = v_lambda

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (other.v_lambda == self.v_lambda and
                    np.all(other.schedule == self.schedule))
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Solution_Candidate():

    def __init__(self, aid, configuration, rating):
        """Initializes a solution candidate.

        Arguments:
        aid -- agent identifier (str)
        configuration -- system configuration (dict(str: schedule_selection))
        rating -- rating of this solution candidate (float)
        """
        self.aid = aid
        self.configuration = dict(configuration)
        self.rating = rating

    def copy(self):
        return Solution_Candidate(self.aid, self.configuration, self.rating)

    def keys(self):
        return self.configuration.keys()

    def get(aid):
        return self.configuration[aid]


class Working_Memory():

    def __init__(self, objective, configuration, solution_candidate):
        """Initializes an agent's working memory.

        Arguments:
        objective -- one-argument objective function (obj --> float)
        configuration -- system configuration (dict(str: schedule_selection))
        solution_candidate -- initial solution candidate (Solution_Candidate)
        """
        self.objective = objective
        self.configuration = dict(configuration)
        self.solution_candidate = solution_candidate.copy()

    def copy(self):
        return Working_Memory(self.objective, self.configuration,
                              self.solution_candidate.copy())
