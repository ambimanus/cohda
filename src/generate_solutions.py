import random
import scenarios
import numpy as np

seed = 0

for h in range(1, 101):
    print 'h=%03d' % h
    sc = scenarios.SC(random.Random(seed), opt_h=h)
    results = scenarios.bruteforce(sc, progress=False)
    np.save('bruteforce_h%03d' % h, results)
