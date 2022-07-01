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

included_runs = list(np.arange(15776, 15821+1))

# list of boards connected

#letters = ['C']
letters = ['B', 'D', 'E'] 

# instances

t = Time_Delta(included_runs, letters, checked=True)

# extra arguments

ref_channel_ = 'D3'
all_channels_ = True
variation_ = 'run'
spill_i_ = 3

# methods (uncomment)

#t.spill_variation(ref_channel_, all_channels_)
#t.hist(ref_channel_, all_channels_, fit_option='synchronise')
#t.hist(ref_channel_, all_channels_, fit_option=None)
#t.hist(ref_channel_, all_channels_, fit_option='gaussians')
#t.hist(ref_channel=ref_channel_, all_channels=all_channels_, variation=variation_, spill_i=spill_i_) #! variation='spill'
#t.run_variation(ref_channel_, all_channels_, file_title='run 3 boards 1 July 2022')
#t.run_colormesh(ref_t_)
channel.resolution(ref_channel_, 'plot rapport')

