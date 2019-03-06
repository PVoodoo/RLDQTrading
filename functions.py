# some common functions, maybe getNextPositionState should be somewhere else... actually so small so no reimplememention
# Programming marko.rantala@pvoodoo.com
# v1.0.0.1 20190305
##############################
# own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################

import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler  # save this scaler to model directory
from sklearn.externals import joblib
import constant

scaler = MinMaxScaler(feature_range=(0.1, 1))  
Debug=True
#constant.MAXCONTRACTS = 1 # maybe better place needed for this, Setup, same as for slippage and so on.. 
#constant.COMMISSION = 0.0  # Like one TickSize min should be right, see TickSize from InputFile, written there for your info
# Slippage add to commission so maybe 2 TickSize together would be quite common. 

def make_timesteps_old(a, timesteps):  ## lets do some additional timesteps backwards, this is not used any more, see next
        array = []
        for j in np.arange(len(a)):
            unit = []
            for i in range(timesteps):
                unit.append(np.roll(a, i, axis=0)[j])  # why this failed in one special case ? ... 
            array.append(unit)
        return np.array(array[timesteps-1:])      ## see, the new array is no more full length if timesteps over 1, price vector need to be shortened as well

        
# fast version from make_timesteps as that found from  stackoverflow...
def make_timesteps(inArr, L = 2):
    # INPUTS :
    # a : Input array
    # L : Length along rows to be cut to create per subarray
    
    #workaround to length 1 , to add 3D, added by mrr, and no need to cut
    if L == 1:
        return inArr.reshape(inArr.shape[0], 1, inArr.shape[1])

    # Append the last row to the start. It just helps in keeping a view output.
    a = np.vstack(( inArr[-L+1:], inArr ))

    # Store shape and strides info
    m,n = a.shape
    s0,s1 = a.strides

    # Length of 3D output array along its axis=0
    nd0 = m - L + 1

    strided = np.lib.stride_tricks.as_strided    
    return strided(a[L-1:], shape=(nd0,L,n), strides=(s0,-s0,s1))[L-1:]   
        
        
# prints formatted price, not used
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data, "prices and features" from a fixed file, scaling done for features
# scaling is dumpped to model directory if new_scaling=true or tried to read from there if using existing one...
def getStockDataVec(key, timesteps, model_name=""):

    data = np.genfromtxt("data/"+ key + ".csv",delimiter=';',skip_header=1,  dtype="float_")[:,1:] # date removed 
    if Debug:
        print("func Datashape: ", data.shape)
    prices = data[:,0]   # column 0 after date removed
    features = data[:,1:] # price removed, column 1
    
    scaler = MinMaxScaler(feature_range=(0.1, 1)) 
    if model_name=="":  # expect training
        scaler.fit(features)   # this need to be put to Dim 2 not to dim 3 values next
        # save scaler for later usage, if out of sample set read together with model .h5
        joblib.dump(scaler, "models/"  +key+ "_skaler.sav")
    else: # use existing scaler, datafile can be different but model name should define model and scaling 
        scaler = joblib.load("models/"  +model_name.split("_")[0]+ "_skaler.sav") # split to cut number of epochs away ...
    
    scaledFeatures = scaler.transform(features)
    
    features3D = make_timesteps(scaledFeatures, timesteps) # non full feature sets removed 
    prices = prices[timesteps-1:]  # cut as well
    
    assert features3D.shape[0] != prices.shape, "Shape error"
    
    if Debug:
        print("func Features shape: ",features3D.shape, prices.shape)
    
	#vec = []
	#lines = open("data/" + key + ".csv", "r").read().splitlines()

	#for line in lines[1:]:
	#	vec.append(float(line.split(",")[4]))
    return prices, features3D

# returns the sigmoid, no need here any more
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an state representation at time t, which is actually a feature set at n, this is a market data
def getState(data, t):
	
    return data[t].reshape(1, data[t].shape[0], data[t].shape[1])  # actually the whole initial data should have been transformed to 4 D initially, now some extra reshaping here


# here we could define important info for PnL calculations, slippage, commissions (slippage and commission can be a single one) , restrictions (how many contracts and so on)  
# max stoploss, target...  
# let's make a simple one first, only 1 contract allowed by buy/sell and so on
# and should we take a reverse position from first reverse, now only make a flat ... 
# immediate reward for augmented calculations
# or modify this that way that no short selling or what ever 

# position state [Flat,Long,Short, Pnl]
def getNextPositionState(action, position_state, prev_price, price):  # or is it price , next_price
    
    
    price_diff = price - prev_price
    immediate_reward = 0.0
    full_pnl = 0.0
    comission_count = 0
    
    # make some type cast not to compare floats
    # F = int(position_state[0])  # Flat, either 0 or 1
    L = int(position_state[1])  # Long, how many contracts, stock_count or ..
    S = int(position_state[2])  # Short, how many ..
    
    #prev_state = position_state[1] 
    
    if L > 0:
        immediate_reward = position_state[1]*price_diff
    
        
    if S > 0:
        immediate_reward = -1.0*position_state[2]*price_diff  
        
    # position_state[3] += immediate_reward  # full PnL , after action

    if action == 1:  # buy
        if S >= 1:  # sell opposite if exit or buy a new one 
            position_state[2] -= 1   # sell 
            comission_count += 1
        elif L < constant.MAXCONTRACTS:
            position_state[1] += 1 
            comission_count += 1
            
    if action == 2:  # sell
        if L >= 1:  # sell opposite if exit or buy a new one 
            position_state[1] -= 1
            comission_count += 1
        elif S < constant.MAXCONTRACTS:
            position_state[2] += 1 
            comission_count += 1    

    position_state[3] = position_state[3] + immediate_reward - comission_count*constant.COMMISSION     #fullPNL , comission_count is max 1 here now but if diff turn policy implemented...  
      
    if position_state[1] > np.finfo(float).eps or position_state[2] > np.finfo(float).eps:   # should I compare to  double.epsilon and not to 0, whats that in python...? let's find out..   
        position_state[0] = 0
    else:
        position_state[0] = 1
        full_pnl = position_state[3] # this is where we return full pnl from previous trade !, it is already calculated here 
        position_state[3] = 0.0
     
    #if  position_state[0] == 1:  # next two line moved to previous else: as we know now that position_state is 1 (or )
     #   full_pnl = position_state[3]   # this is where we return full pnl from previous trade !, it is already calculated here 
      #  position_state[3] = 0.0
        
    #if prev_state == 1 and position_state[1] > 0 and position_state[3] == 0:
    #    print(price_diff, immediate_reward, full_pnl, comission_count )
    
    return  position_state, immediate_reward, full_pnl
            
            
            
            
        