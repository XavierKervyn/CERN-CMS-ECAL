# imports
from ecalautoanalysis import *
import numpy as np

# included runs

#included_runs = [15680]
#included_runs = list(np.arange(15728, 15773+1))

# list of boards connected

letters = ['A'] 

# instances

t = Time_Delta(included_runs, letters)

# extra arguments

ref_channel_ = 'C2'
all_channels_ = False
variation_ = 'spill'
spill_i_ = 3

# methods (uncomment)

#t.spill_variation(ref_channel_, all_channels_)
#t.hist(ref_channel_, all_channels_, fit_option='synchronise')
#t.hist(ref_channel_, all_channels_, fit_option=None)
#t.hist(ref_channel_, all_channels_, fit_option='gaussians', nb_fits=11)
#t.hist(ref_channel=ref_channel_, all_channels=all_channels_, variation=variation_, spill_i=spill_i_) #! variation='spill'
#t.run_variation(ref_channel_, all_channels_, file_title='run 4 boards 29 June 2022')
#t.run_colormesh(ref_channel_)
t.resolution(ref_channel_, 'run 4 boards 29 June 2022')

