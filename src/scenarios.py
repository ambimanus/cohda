# coding=utf-8

from __future__ import division

import sys
import os
import random

import numpy as np

from model import Objective, Value
from util import autoassign, import_object, current_method, base10toN


# SC problem instance [Pisinger EJOR, 83, 1995],
# capacity generation from [Han, 2010]
class SC(object):

    @autoassign
    def __init__(self,
                 sim_seed=0,
                 sim_max_steps=None,
                 sim_msg_delay_min=1,
                 sim_msg_delay_max=2,
                 sim_agent_delay_min=1,
                 sim_agent_delay_max=5,
                 sim_initial_solution='worst',  # initial solution: random or worst?
                 sim_initial_agent=0,           # which agent is initiator?
                 log_to_file=False,
                 log_basepath='../data',
                 log_title=None,
                 network_module='networks',
                 network_type='smallworld',
                 network_c=3,
                 network_k=1,
                 network_phi=0.5,
                 opt_m=10,                      # number of classes
                 opt_n=5,                       # number of items in each class
                 opt_q=5,                       # number of dimensions
                 opt_s=100,                     # capacity divider (no. of instances)
                 opt_h=5,                       # capacity selector (no. of instance)
                 ):
        self.rnd = random.Random(sim_seed)
        np.random.seed(int(random.getrandbits(32)))

        self.sim_msg_delay_min = max(1, self.sim_msg_delay_min)
        self.sim_msg_delay_max = max(self.sim_msg_delay_min,
                                     self.sim_msg_delay_max)
        self.sim_agent_delay_min = max(1, self.sim_agent_delay_min)
        self.sim_agent_delay_max = max(self.sim_agent_delay_min,
                                       self.sim_agent_delay_max)

        self.agent_ids = [i for i in range(opt_m)]
        self.network = import_object(network_module, network_type)(self)

        opt_w = _weights(opt_m, opt_n, opt_q)
        self.agent_search_spaces = [[Ndarray(v) for v in opt_w[i]]
                                    for i in range(opt_m)]
        self.objective = Manhattan(np.array(
            [_capacity(i, opt_h, opt_s, opt_m, opt_w)
             for i in range(1, opt_q + 1)]))

        self.sol_fitness_max, self.sol_fitness_min, idx = _bounds_bruteforce(
            opt_h, opt_n, opt_m, self.agent_search_spaces, self.objective)
        self.sol_fitness_avg = _sol_avg(self.rnd, self.agent_search_spaces,
                                        opt_m, self.objective)

        if sim_initial_solution == 'random':
            self.agent_initial_values = [self.rnd.choice(aw)
                                         for aw in self.agent_search_spaces]
        elif sim_initial_solution == 'worst':
            self.agent_initial_values = [self.agent_search_spaces[i][j]
                                         for i, j in enumerate(idx)]
        else:
            raise RuntimeError('Unsupported sim_initial_solution "%s"'
                               % sim_initial_solution)


class Manhattan(Objective):
    """Objective function for the inverse manhattan distance (1-norm).

    In order to be a maximizing objective, we calculate the *inverse* manhattan
    distance, i.e., 1/d where d is the manhattan distance.

    Objects of this type can be called like a function. The only supported
    argument is a list of Ndarray objects, where each Ndarray.v object has the
    same shape as the predefined target. This way, this objective does not only
    support the euclidean space (i.e., euclidean vectors of arbitrary length),
    but generally any vector space:

    Let s = (p, q, ...) be the shape of our target. First, the given argument
    list of length m (where each contained value has also shape s) is converted
    into a ndarray of shape (m, p, q, ...). Second, the sum over the m elements
    along axis 0 is calculated, to obtain an array of shape s again. Let v be
    the resulting array. The manhattan distance is then calculated as the total
    sum (over all dimensions) of the element-wise absolute differences between
    target and v.
    """

    def __init__(self, target):
        """Initializes the objective function (ζ).

        Arguments:
        target -- target vector (ndarray)
        """
        Objective.__init__(self, [Ndarray])
        self.target = target

    def _f(self, x):
        """Calculates the manhattan distance (1-norm) between x and target.

        Arguments:
        x -- the argument to be evaluated (list of Schedule objects)

        Returns:
        fitness -- the fitness for x (float)
        """
        ts = self.target.shape
        for schedule in x:
            assert schedule.v.shape == ts, ('Unsupported shape: %s'
                                            % schedule.v.shape)
        if len(x) == 0:
            x = np.zeros(self.target.shape)
        else:
            x = np.array([schedule.v for schedule in x]).sum(0)
        diff = np.abs(self.target - x)
        # We are maximizing, so return the inverse distance
        return 1.0 / np.sum(diff)

    def __repr__(self):
        return '%s, target: %s' (Objective.__repr__(self), self.target)


class Ndarray(Value):

    def __init__(self, v):
        assert isinstance(v, np.ndarray), 'Unsupported type: %s' % type(v)
        Value.__init__(self, v)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.all(other.v == self.v)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


# [Han, 2009] eq. (13)-(16),(20)-(24)
#   use L(10*i-1, 10*i) (see table 2)
#   use S(10) or S(k+5) (see table 2)
def _weights(opt_m, opt_n, opt_q, inversed=False, dim_dependent=True):
    # Generate profits (opt_m * opt_n)
    p = np.empty((opt_m, opt_n))
    for i in range(opt_m):
        for j in range(opt_n):
            # table 2 --> eq. (16) --> eq. (13)
            p_i_min, p_i_max = 10 * i, 10 * (i + 1)
            p[i][j] = (j / (opt_n - 1)) * (p_i_max - p_i_min) + p_i_min

    # Repeat profits in q dimensions (makes the following calculations easier)
    p = p.repeat(opt_q).reshape(opt_m, opt_n, opt_q)

    # Find maximal profit in each class
    p_max = p.max(axis=1)
    # Shape to n*q dimensions (makes the following calculations easier)
    p_max = p_max.repeat(opt_n).reshape((opt_m, opt_n, opt_q))

    # Calculate delta-value for eq. (20)
    if not dim_dependent:
        # table 2 --> eq. (20)
        d = np.array([10] * opt_q)
    else:
        # table 2 --> eq. (24) --> eq. (20)
        d = np.arange(1, opt_q + 1) + 5

    # Minor error in [Han, 2009]. Reverse delta-values to correct.
    # d = d[::-1]

    # Derive weights as strongly correlated to profits
    if not inversed:
        # eq. (20)
        w = (p + (p_max / d))
    else:
        # eq. (22)
        w = (p_max - (p / d))

    # Return a copy of the array, because we need contigous data!
    return np.array(w.astype(int))


# [Han, 2009] eq. (26)
def _capacity(k, h, S, opt_m, w):
    # Filter k_th dimension in source array:
    # - Transpose, so that the dimension index is the most outer one
    # - Select k_th dimension
    # - Re-transpose, so that the array comprises m classes with n entries
    dim_k = w.T[k - 1].T

    # Sum minimal and maximal elements of each class, respectively.
    w_max_k = sum([np.max(dim_k[i]) for i in range(opt_m)])
    w_min_k = sum([np.min(dim_k[i]) for i in range(opt_m)])

    # Calculate capacity according to eq. (26)
    return int(((h / (S + 1)) * (w_max_k - w_min_k)) + w_min_k)


def _bounds_bruteforce(h, n, m, w, obj):
    path = os.path.dirname(os.path.realpath(__file__))
    fn = '%s/../sc_data/bruteforce_h%03d.npy' % (path, h)
    if os.path.exists(fn):
        d = np.load(fn)
    else:
        print 'Bruteforce results file not found, calculating now.'
        d = bruteforce(m, n, w, obj)
        print 'Storing results as %s' % fn
        np.save(fn, d)
    imin, imax = np.argmin(d), np.argmax(d)
    vmin, vmax = d.min(), d.max()
    conv = base10toN(imin, n)
    prefix = '0' * (m - len(conv))
    idx_min = [int(s) for s in prefix + conv]

    return vmax, vmin, idx_min


def _sol_avg(rnd, opt_w, opt_m, objective):
    lim = len(opt_w) * len(objective.target)
    d = np.empty((lim,))
    for k in range(lim):
        # Choose random j for each class
        idx = [rnd.randint(0, len(objective.target) - 1) for j in range(opt_m)]
        # Select according weight in each class
        w = [opt_w[i][j] for i, j in enumerate(idx)]
        # Sum weights and calculate fitness (result is a scalar value!)
        d[k] = objective(w, bypass_counter=True)

    # Return average values over 'lim' runs
    return np.mean(d)


def ids(m, n):
    l = np.zeros(m, dtype=np.int32)
    yield l
    for i in xrange(n**m - 1):
        for j in xrange(m - 1, -1, -1):
            if l[j] == n - 1:
                l[j] = 0
            else:
                l[j] += 1
                break
        yield l


def bruteforce(m, n, w, obj, progress=True, thres=None):
    # Install signal handler for SIGINT
    import signal

    def sigint_detected(signal, frame):
        print 'Stopping due to user request (SIGINT detected).'
        sys.exit(1)

    signal.signal(signal.SIGINT, sigint_detected)
    print 'Cancel with Ctrl-C (i.e., SIGINT).'

    # print
    # print 'brute force'

    import time
    tm = time.time()
    counter, min_counter, max_counter = 0, 0, 0
    s_min, s_max = None, None
    r_min, r_max = None, None

    if thres is None:
        SIZE = n**m
    else:
        SIZE = thres
    # selections = np.empty((SIZE,m))
    results = np.empty((SIZE,))

    if progress:
        import util
        progress = util.PBar(SIZE).start()
    for i in ids(m, n):
        s = [w[i][j] for i, j in enumerate(i)]
        r = obj(s)
        # selections[counter] = i
        results[counter] = r
        # if r_min is None or r < r_min:
        #     s_min = s
        #     r_min = r
        #     min_counter = counter
        #     min_ids = list(i)
        # if r_max is None or r > r_max:
        #     s_max = s
        #     r_max = r
        #     max_counter = counter
        #     max_ids = list(i)
        counter += 1
        if counter >= SIZE:
            break
        if progress:
            progress.update(counter)
    if progress:
        progress.finish()
    tm = time.time() - tm

    print '%d solutions calculated in %d seconds' % (counter, tm)
    # print
    # print 'Optimal Solution'
    # print '----------------------------------'
    # print '  solution number:', min_counter
    # print '  residual error:', r_min
    # print '  solution ids:', min_ids
    # print '  solution:', s_min
    # print
    # print 'Worst Solution'
    # print '----------------------------------'
    # print '  solution number:', max_counter
    # print '  residual error:', r_max
    # print '  solution ids:', max_ids
    # print '  solution:', s_max

    return results


if __name__ == '__main__':
    import random
    seed = 0
    sc = SC(random.Random(seed), seed)

    print sc
