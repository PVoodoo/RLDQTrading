# some common functions, maybe getNextPositionState and run_flat should be somewhere else... in PVAgent, actually so small so no reimplememention
# Programming marko.rantala@pvoodoo.com
# v1.0.0.1 20190305
# v1.0.0.2 20190307 eod
# v1.0.1.0 20190310 Start of 
##############################
# own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################

import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler  # save this scaler to model directory
from sklearn.externals import joblib
import constant

Debug=True

scaler = MinMaxScaler(feature_range=(0.1, 1))  

#constant.MAXCONTRACTS = 1 # maybe better place needed for this, Setup, same as for slippage and so on..   -> constant
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
def getStockDataVec(key, timesteps, model_name="", dayTrading=False):

    data = np.genfromtxt("data/"+ key + ".csv",delimiter=';',skip_header=1,  dtype="float_")[:,1:] # date removed 
    if Debug:
        print("func Datashape: ", data.shape)
    prices = data[:,0]   # column 0 after date removed
    
    #eod = data[:,1] # column 1 after price (or some not used feature if not )
    startFeatures = 1  # normal mode
    if dayTrading:
        eod = data[:,1] # column 1 after price (or some not used feature if not )
        startFeatures = 2
    else:
        eod = np.zeros(len(data), dtype=int)
    eod[len(data)-1] = 1  # only last is marked, so take flat position there , maybe last DayTrading data should be marked as well! yes, if not session boundary, just remove ident here, or add now if not needed 
        
    features = data[:,startFeatures:] # price removed, column 1 or 2 based to dayTrading
    
    scaler = MinMaxScaler(feature_range=(0.1, 1)) 
    if model_name=="":  # expect training
        scaler.fit(features)   # this need to be put to Dim 2 not to dim 3 values next
        # save scaler for later usage, if out of sample set read together with model .h5
        joblib.dump(scaler, "models/"  +key+ "_skaler.sav")
    else: # use existing scaler, datafile can be different but model name should define model and scaling 
        scaler = joblib.load("models/"  +model_name.split("_")[0]+ "_skaler.sav") # split to cut number of epochs away ...
        # if this fails, a new fit could be thought here too
        
    scaledFeatures = scaler.transform(features)
    
    features3D = make_timesteps(scaledFeatures, timesteps) # non full feature sets removed 
    prices = prices[timesteps-1:]  # cut as well
    eod = eod[timesteps-1:] ## 
    
    assert features3D.shape[0] != prices.shape, "Shape error"
    
    if Debug:
        print("func Features shape: ",features3D.shape, prices.shape)
    
	#vec = []
	#lines = open("data/" + key + ".csv", "r").read().splitlines()

	#for line in lines[1:]:
	#	vec.append(float(line.split(",")[4]))
    return prices, features3D, eod

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
def getNextPositionStateWrong(action, position_state, prev_price, price, eod):  # or think it like current  price and  next_price  !!!!!!!!!!!!!!!!!!!!!!
    
    
    price_diff = price - prev_price
    immediate_reward = 0.0
    full_pnl = 0.0
    comission_count = 0
    
        
    # make some type cast not to compare floats
    #F = int(position_state[0])  # Flat, either 0 or 1
    L = int(position_state[1])  # Long, how many contracts, stock_count or ..
    S = int(position_state[2])  # Short, how many ..
    
    #prev_state = position_state[1] 
    
    if L > 0:
        immediate_reward = position_state[1]*price_diff
        
    if S > 0:
        immediate_reward = -1.0*position_state[2]*price_diff  
        
    if eod == 1:   # no new positions taken, although
        return run_flat(position_state, immediate_reward)                            # exit here tooo 
        
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
     
    #if  position_state[0] == 1:  # next two line moved to previous else: as we know now that position_state is 1 = Flat(or )
     #   full_pnl = position_state[3]   # this is where we return full pnl from previous trade !, it is already calculated here 
      #  position_state[3] = 0.0
        
    #if prev_state == 1 and position_state[1] > 0 and position_state[3] == 0:
    #    print(price_diff, immediate_reward, full_pnl, comission_count )
    
    return  position_state, immediate_reward, full_pnl  # realize that position_state is not a new allocation, it points to prev np array where values has been changed 


def run_flat(position_state, immediate_reward):

    full_pnl = position_state[3] + immediate_reward - position_state[1]*constant.COMMISSION - position_state[2]*constant.COMMISSION  # FYI either position_state[0 or 1] is zero or both

    # set flat
    position_state[0] = 1;
    position_state[1] = 0;
    position_state[2] = 0;
    position_state[3] = 0.0;
    
    return position_state, immediate_reward, full_pnl

# let's skip the flat position between states, take opposite position at opposite signal, simplified version 
# or actually different, should there be an additional action, exit => go Flat ???, yes , implemented with signal 0 (if active signal)
def getNextPositionState(action, position_state, prev_price, price, eod, prev_eod):  # or think it like current  price and  next_price  !!!!!!!!!!!!!!!!!!!!!!
# position state [Flat,Long,Short, Pnl]
    
    
    price_diff = price - prev_price
    immediate_reward = 0.0
    full_pnl = 0.0
    comission_count = 0
    
    if constant.IGNORE_EOD_ACTIVATION and prev_eod == 1:  # no new state (should be okay after last bars flat set, BUT set anyway here again
        position_state[0] = 1
        position_state[1] = 0
        position_state[2] = 0
        position_state[3] = 0.0
        return position_state, 0.0, 0.0
    
    if action == 0: # next one used anyway now # and constant.ACTIONZERO == 1:
        full_pnl = position_state[3] - position_state[1]*constant.COMMISSION - position_state[2]*constant.COMMISSION  # either one [1],[2] or both are zero 
        # immediate_reward = 0.0
        position_state[0] = 1
        position_state[1] = 0
        position_state[2] = 0
        position_state[3] = 0.0
        return  position_state, immediate_reward, full_pnl   # 
        
        
    # make some type cast not to compare floats
    F = int(position_state[0])  # Flat, either 0 or 1
    LC = int(position_state[1])  # Long, how many contracts, stock_count or ..
    SC = int(position_state[2])  # Short, how many ..
    
    
    #prev_state = position_state[1] 
    
    if action == 1:  # buy
        if SC > 0:
            full_pnl = position_state[3] - SC*constant.COMMISSION 
            position_state[3] = price_diff - constant.COMMISSION # one buy
        if LC < constant.MAXCONTRACTS:  
            immediate_reward = price_diff - constant.COMMISSION # one buy, more 
            position_state[1] += 1 
            if LC > 0: # SC can't be positive then, no need to worry next at that point ,, CHECK LC == 0 and 
                position_state[3] += (LC+1)*price_diff
                
        if LC == constant.MAXCONTRACTS:
            position_state[3] += LC*price_diff   # and no immediate reward any more 
        if F == 1: 
            # immediate_reward = price_diff  # already above at LC <
            position_state[1] == 1
            # position_state[2] == 0 
            position_state[3] = price_diff - constant.COMMISSION
            
        position_state[0] = 0
        position_state[2] = 0
        # position_state[3]  # should be calculated above to all possibilities
        
    if action == 2:  # sell
        if LC > 0:
            full_pnl = position_state[3] - LC*constant.COMMISSION 
            position_state[3] = -1.0*price_diff - constant.COMMISSION # one buy
        if SC < constant.MAXCONTRACTS:  
            immediate_reward = -1.0*price_diff - constant.COMMISSION # one buy, more 
            position_state[2] += 1 
            if SC > 0: # SC can't be positive then, no need to worry next at that point ,, CHECK LC == 0 and 
                position_state[3] += (LC+1)*-1*price_diff
        if SC == constant.MAXCONTRACTS:
            position_state[3] += -1.0*SC*price_diff   # and no immediate reward any more 
        if F == 1: 
            # immediate_reward = price_diff  # already above at LC <
            position_state[2] == 1
            # position_state[2] == 0 
            position_state[3] = -1.0*price_diff - constant.COMMISSION
            
        position_state[0] = 0
        position_state[1] = 0
        # position_state[3]  # should be calculated above to all possibilities
  
      
   
    
    if eod == 1:     # make flat after this BUT important, either action 1 or 2 can have affect (calculated above) , so last bar action has a very special handling
        full_pnl = full_pnl - position_state[1]*constant.COMMISSION - position_state[2]*constant.COMMISSION + immediate_reward # either one [1],[2] or both are zero 
        # full_pnl and immediate reward is calculated at action 1 and 2 above
        print("************************", full_pnl) # see, this is not zero all the time
        # immediate reward based to action above, if buy or sell 
        position_state[0] = 1
        position_state[1] = 0
        position_state[2] = 0
        position_state[3] = 0.0
        return  position_state, immediate_reward, full_pnl   # 
    
    return  position_state, immediate_reward, full_pnl  # realize that position_state is not a new allocation, it points to prev np array where values has been changed 




##############################
# own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################        
