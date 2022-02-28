import numpy as np

def subsample(X, sub=5):
  '''
  Subsamples by averaging every adjacent *sub* samples provided by the parameter
  Resulting length is datapoint length/sub
  '''

  time_length = X.shape[1]
  if time_length % sub != 0:
    raise Exception('Pick a sub that cleanly divises')

  sub_shape = (X.shape[0], X.shape[1]//sub, X.shape[2])
  out = np.zeros(sub_shape)

  for dp in range(0, X.shape[0]): #loop through each datapoint
    out_idx = 0
    for ts in range(0, time_length, sub): #loop through each sub partition
      out[dp][out_idx][:] = np.sum(X[dp][ts:ts+sub][:], axis=0) / sub
      out_idx = out_idx + 1
  
  return out
