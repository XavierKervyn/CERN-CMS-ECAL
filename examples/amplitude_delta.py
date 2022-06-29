# imports
from ecalautoanalysis import *
import numpy as np

# included runs

#included_runs = [15680]
#included_runs = list(np.arange(15728, 15773+1))

# list of boards connected

letters = ['C'] 

# instances

ad = Amplitude_Delta(included_runs, letters)

# extra arguments

ref_channel_ = 'C2'
all_channels_ = False
variation_ = 'spill'
spill_i_ = 3

# methods

#ad.spill_variation(ref_channel_, all_channels_)
#ad.hist(ref_channel_, all_channels_)
#ad.hist(ref_channel=ref_channel_, all_channels=all_channels_, variation=variation_, spill_i=spill_i_) #! variation='spill'
#ad.run_variation(ref_channel_, all_channels_, 'test code 280622')
#ad.run_colormesh(ref_channel_)
