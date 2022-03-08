from re import X
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

def normalize(x):
    xNorm = np.zeros_like(x)
    #Get max and mins across all channels through trials and time bins
    trainMaxofChannels = np.max(x, axis=(-1, -3))
    trainMaxofChannels = trainMaxofChannels.reshape((22,1))
    trainMinofChannels = np.min(x, axis=(-1, -3))
    trainMinofChannels = trainMinofChannels.reshape((22,1))
    minMaxofChannels = trainMaxofChannels - trainMinofChannels
    # Use prevoius Values to calculate Min Max Normalization
    # Normalizing across each trial
    for i in range(x.shape[0]):
        xNorm[i] = (x[i] - trainMinofChannels)/(trainMaxofChannels - trainMinofChannels)
    return xNorm

def standardize(x):
    xStand = np.zeros_like(x)
    #Get Mean and StDev across all channels through trials and time bins
    trainChannelMean = np.mean(X, axis=(-1, -3))
    trainChannelMean = trainChannelMean.reshape((22,1))
    trainChannelStd = np.std(x, axis=(-1, -3))
    trainChannelStd = trainChannelStd.reshape((22,1))
    # Use prevoius Values to standardize
    # Standardize across each trial
    for i in range(x.shape[0]):
        xStand[i] = (x[i] - trainChannelMean)/trainChannelStd
    return xStand
