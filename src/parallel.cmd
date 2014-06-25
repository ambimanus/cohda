
# GNU parallel:
seq $(python parallel.py count) | parallel "python parallel.py"
