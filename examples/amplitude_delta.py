# imports
from ecalautoanalysis import *
import numpy as np

# included runs

#included_runs = [15680]
#included_runs = list(np.arange(15728, 15773+1))
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

ad = Amplitude_Delta(included_runs, letters, checked=True)

# extra arguments

ref_channel_ = 'D3'
all_channels_ = True
variation_ = 'spill'
spill_i_ = 3

# methods

print("\n----- Spill variation -----\n")
ad.spill_variation(ref_channel_, all_channels_)
print("\n----- Histograms (synchronise) -----\n")
ad.hist(ref_channel_, all_channels_)
print("\n----- Histograms (spill) -----\n")
ad.hist(ref_channel=ref_channel_, all_channels=all_channels_, variation='spill', spill_i=spill_i_)
print("\n----- Run variation -----\n")
ad.run_variation(ref_channel_, all_channels_, '4boards')
print("\n----- Run colormesh -----\n")
ad.run_colormesh(ref_channel_)
