# imports
from ecalautoanalysis import *
import numpy as np

# included runs

#included_runs = [15680, 15681, 15682]
#included_runs = list(np.arange(15728, 15773+1))
"""
included_runs = list(np.arange(15674, 15723+1))

included_runs.remove(15676)
included_runs.remove(15697)
included_runs.remove(15711)
included_runs.remove(15712)
included_runs.remove(15718)
"""

#included_runs = list(np.arange(15776, 15821+1))

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

t = Time_Delta(included_runs, letters, checked=True)

# extra arguments

ref_channel_ = 'A1'
all_channels_ = True
spill_i_ = 3

# methods (uncomment)

#print("\n----- Spill variation -----\n")
#t.spill_variation(ref_channel_, all_channels_)

#print("\n----- Histograms (synchronise) -----\n")
#t.hist(ref_channel_, all_channels_, fit_option='synchronise')

#print("\n----- Histograms (None) -----\n")
#t.hist(ref_channel_, all_channels_, fit_option=None)

#print("\n----- Histograms (gaussians) -----\n")
#t.hist(ref_channel_, all_channels_, fit_option='gaussians')

#print("\n----- Histograms (spill) -----\n")
#t.hist(ref_channel=ref_channel_, all_channels=all_channels_, variation='spill', spill_i=spill_i_)

#print("\n----- Run variation -----\n")
#t.run_variation(ref_channel_, all_channels_, file_title='4board')

#print("\n----- run colormesh -----\n")
#t.run_colormesh(ref_channel_)

print("\n----- Resolution -----\n")
t.resolution(ref_channel_, '4boards')

print("\n----- Resolution all -----\n")
t.resolution_all(ref_channel_, '4boards')

