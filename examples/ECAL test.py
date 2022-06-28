# imports
from ecalautoanalysis import *
import numpy as np

# ---- CHANGE HERE ----
included_runs = [15680, 15681]
#included_runs = list(np.arange(15674, 15723+1))
"""
included_runs.remove(15676)
included_runs.remove(15697)
included_runs.remove(15711)
included_runs.remove(15712)
included_runs.remove(15718)
"""

#Jun. 2022: only this board is working
letters = ['C'] 

# various instances
t = Time_Delta(included_runs, letters)
a = Amplitude(included_runs, letters)
ad = Amplitude_Delta(included_runs, letters)

# ---- CHANGE HERE ----
ref_channel_ = 'C2'
all_channels_ = False
variation_ = 'spill'
spill_i_ = 3
# ---- END OF CHANGES ----
"""
t.spill_variation(ref_channel_, all_channels_)
t.hist(ref_channel_, all_channels_, fit_option='synchronise')
t.hist(ref_channel_, all_channels_, fit_option=None)
#t.hist(ref_channel_, all_channels_, fit_option='gaussians', nb_fits=11)
t.hist(ref_channel=ref_channel_, all_channels=all_channels_, variation=variation_, spill_i=spill_i_) #! variation='spill'
"""
t.run_variation(ref_channel_, all_channels_, file_title='test code 280622')
"""
t.run_colormesh(ref_channel_)
t.resolution(ref_channel_, 'test code 280622')

a.spill_variation()
a.hist()
"""
a.run_variation('test code 280622')
"""
a.run_colormesh()

a.resolution('test code 280622')
ad.spill_variation(ref_channel_, all_channels_)
ad.hist(ref_channel_, all_channels_)
ad.hist(ref_channel=ref_channel_, all_channels=all_channels_, variation=variation_, spill_i=spill_i_) #! variation='spill'
"""
ad.run_variation(ref_channel_, all_channels_, 'test code 280622')
"""
ad.run_colormesh(ref_channel_)
"""
