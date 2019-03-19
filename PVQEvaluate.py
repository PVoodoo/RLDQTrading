# evaluate training results trained by PVQTrain.py
# example copypaste down, see the models directory for existence of model .h5 
# python PVQEvaluate.py RLDataForCL30 RLDataForCL30_500 
# Programming marko.rantala@pvoodoo.com
# v1.0.0.1 20190305 
# v1.0.0.2 20190307 eod
# v1.0.1.0 20190310 Start of 
##############################
# my own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################


import keras
import matplotlib.pyplot as plt
from keras.models import load_model

from agent.PVAgent import PVAgent
from functions import *
import sys
import constant

Debug=constant.Debug

if len(sys.argv) != 3:
	print("Usage: python PVQEvaluate.py [stock] [model] ")
	exit()

stock_name, model_name = sys.argv[1], sys.argv[2] 
model = load_model("models/" + model_name +".h5")  # sort of duplicate just to get feature count!!, bad programming... 
#feature_count = model.layers[0].input.shape.as_list()[1]
if Debug:
    print(model.layers[0].input.shape.as_list())
timesteps = model.layers[0].input.shape.as_list()[1]

dayTrading=False
if stock_name[-1:] == 'D':  # just to see if datafile is ending with D, writer program https://pvoodoo.blogspot.com/2019/03/writetraindata-tool-for-reinforcement.html is adding this automatically if DayTrading data generated 
    dayTrading = True

prices, data, eod = getStockDataVec(stock_name, timesteps, model_name=model_name, dayTrading=dayTrading)
agent = PVAgent(data.shape[1], data.shape[2], is_eval=True, model_name=model_name)

l = len(data) - 1
batch_size = 32

ignorePositionState = False   # just for debugging purposes
if Debug:
    print(prices.shape, data.shape, eod.shape)

    
total_profit = 0.0
agent.inventory = []

market_state = getState(data, 0)
#position_state = np.array([1,0,0,0.0]).reshape(1,4) # [Flat,Long,Short,PnL]  what the hell is turning this ... 
position_state = np.zeros(constant.PositionStateWidth).reshape(1,constant.PositionStateWidth) 
state = [market_state, position_state]


# all next ts_ only for illustration purposes, and some +1's to illustrate that easier
ts_buy = []
ts_sell = []
ts_flat = []
ts_eod = []
ts_PnL =  np.zeros(l)
ts_CumPnL = np.zeros(l)
ts_Action = np.zeros(l)  # let's show that too what is proposed by nn
numTrades = 0

for t in range(l):
    action = agent.act(state)
    ts_Action[t] = -1.0 if action >=2 else action  # just to show the action nicer , 2 = sell  
    if int(eod[t]) == 1: # t or t + 1   ????????????????????????, t is right here, maybe the whole data should be sifted 1 
        ts_eod.append(t)

    # if ignorePositionState:  
        # state[1][0][0] = 1  # ok, think as a flat always ...
        # state[1][0][1] = 0
        # state[1][0][2] = 0
        # state[1][0][3] = 0
        # if int(eod[t+1]) == 1:
            # print(state[0])
        
    next_market_state = getState(data, t + 1)
    
  
    # only change places drawn to the priceline
    if (action == 1) and (state[1][0][1] < constant.MAXCONTRACTS): #  and eod[t+1] < 1:# buy do not and if IGNORE_EOD_ACTIVATION, first action ignored , to be remembered  
        ts_buy.append(t)
        numTrades +=1
        if Debug:
            print("Buy before, state",state[1][0])
    elif action == 2 and state[1][0][2] < constant.MAXCONTRACTS: #  and eod[t+1] < 1: # sell
        ts_sell.append(t)
        numTrades +=1
        if Debug:
            print("Sell before, state",  state[1][0])
    elif action == 0 and state[1][0][0] < 1: # flat, either active or passive, see constant   
        ts_flat.append(t)
    
        
    next_position_state, immediate_reward, PnL = getNextPositionState(action, state[1][0], prices[t], prices[t+1], eod[t+1], eod[t])
    total_profit += PnL*constant.POINTVALUE
    #reward = immediate_reward + PnL
    
    if (Debug):
        if (action == 1) and (state[1][0][1] < constant.MAXCONTRACTS):# buy
            print("Buy after", next_position_state, PnL)
        elif action == 2 and state[1][0][2] < constant.MAXCONTRACTS: # sell 
            print("Sell after",  next_position_state, PnL)
        if int(eod[t+1]) == 1 :
            print(" ****************************************** EOD PNL:", PnL)
    
     

    if PnL != 0.0:
        ts_PnL[t] = PnL*constant.POINTVALUE
    ts_CumPnL[t] = total_profit 
    
    done = True if t == l - 1 else False
    next_state = [next_market_state, next_position_state.reshape(1,constant.PositionStateWidth)]
 
    #agent.memory.append((state, action, reward, next_state, done))
    state = next_state
    
    
    if done:
        print("--------------------------------")
        print(stock_name + " Total Profit : ", total_profit, " tradeCount: ", numTrades )
        print("--------------------------------")

        
        
# hi, matplot gurus, please help here :)        

data = np.array(prices)
ts = np.arange(l).astype(int)
ts_buy = np.array(ts_buy).astype(int)
ts_sell = np.array(ts_sell).astype(int)
ts_flat = np.array(ts_flat).astype(int)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax2.plot(ts, data[ts], zorder=3)
ax2.scatter(ts_buy, data[ts_buy], c="g", label="Buy")
ax2.scatter(ts_sell, data[ts_sell], c="r", label="Sell")
ax2.scatter(ts_eod, data[ts_eod], color="magenta", label="EoD")  # this could be written directly from eod
ax2.scatter(ts_flat, data[ts_flat], color="gray", label="Flat")
ax2.set_ylabel('Price')

ax1.set_ylabel('PnL')
ax1.bar(ts, ts_PnL, color="b", label="PnL", width=0.8, zorder=4)
ax1.plot(ts, ts_CumPnL, color="c", label="Cumulative PnL", zorder=1)
ax1.bar(ts, ts_Action, color="y", label="Action", width=1.0, zorder=2)
ax1.axhline(0, color='gray', lw=1)

plt.legend()
fig.tight_layout() 
plt.show()

##############################
# my own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################