# imports
from ecalautoanalysis import *
import numpy as np

# included runs

#included_runs = [15777]
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

#included_runs = list(np.arange(15776, 15821+1))
#included_runs = [15777, 15778, 15779, 15780]

# Four boards
included_runs = list(np.arange(15832, 15881+1))
included_runs.remove(15841)
included_runs.remove(15842)
included_runs.remove(15860)
included_runs.remove(15866)

# list of boards connected

#letters = ['C']
#letters = ['B', 'D', 'E']
letters = ['A', 'B', 'D', 'E'] 

# instances

a = Amplitude(included_runs, letters, checked=True)

# extra arguments

spill_i_ = 3

# methods (uncomment)

#print("\n----- Spill variation -----\n")
#a.spill_variation()

#print("\n----- Histograms -----\n")
#a.hist()

#print("\n----- Histograms (spill) -----\n")                                                    
#a.hist(variation='spill', spill_i=spill_i_)

print("\n----- Run variation -----\n")
a.run_variation('4boards')

#print("\n----- Run colormesh -----\n")
#a.run_colormesh()

print("\n----- Resolution -----\n")
a.resolution('4boards')

print("\n----- Resolution all -----\n")
a.resolution_all('4boards')

