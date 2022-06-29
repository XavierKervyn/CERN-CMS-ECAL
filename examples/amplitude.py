# imports
from ecalautoanalysis import *
import numpy as np

# included runs

#included_runs = [15680]
#included_runs = list(np.arange(15728, 15773+1))
"""
included_runs = list(np.arange(15674, 15723+1))

included_runs.remove(15676)
included_runs.remove(15697)
included_runs.remove(15711)
included_runs.remove(15712)
included_runs.remove(15718)
"""

#included_runs = [15774]

included_runs = list(np.arange(15776, 15821+1))

#included_runs = [15777, 15778, 15779, 15780]

# list of boards connected

#letters = ['C']
letters = ['B', 'D', 'E'] 

# instances

a = Amplitude(included_runs, letters, checked=True)

# extra arguments

ref_channel_ = 'C2'
all_channels_ = False
variation_ = 'spill'
spill_i_ = 3

# methods (uncomment)

#a.spill_variation()
#a.hist()
#a.run_variation('run 4 boards 29 June 2022')
#a.run_colormesh()
a.resolution('test_3_boards')

