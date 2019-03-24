# example of reinforcement learning train program for stock data, type of DQN or sort of DDQN with special two sided keras model
# start training example copypaste
# python PVQTrain.py RLDataForCL30 1000 4 
# 20190305 
# v1.0.0.1 initial revisio
# v1.0.0.2 Daytrading mode added, eod
# v1.0.0.3 plt
# v1.0.1.0 20190310 Start of ..
# v1.0.2.0 # state size 3
# v1.0.3.0 main callable for  ipynb usage
# Programming marko.rantala@pvoodoo.com 
# Input data format, see https://pvoodoo.blogspot.com/2019/03/writetraindata-tool-for-reinforcement.html
# The model is based very much to traditional cartpole example, added with some "smart"? ...

##############################
# own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################

# if input file (stockfile) ends with "D" before .csv, like RLDataForCL30D.csv , this expect it to have a day break info after price, 1 if last bar of session otherwise 0
# and this use daytrading rules, Flat at end of session for day
from agent.PVAgent import *
from functions import *
import constant

import matplotlib.pyplot as plt

def main(stock_name, episode_count=1000, timesteps=4):

    #from environment import SimpleTradeEnv, as this could be implemented as a gym type
   
    dayTrading=False
    if stock_name[-1:] == 'D':  # just to see if datafile is ending with D, writer program https://pvoodoo.blogspot.com/2019/03/writetraindata-tool-for-reinforcement.html is adding this automatically if DayTrading data generated 
        dayTrading = True

        
    batch_size = 32  # worth to check and check the memory length of agent replay memory
    #agent = Agent(timesteps)

    prices, market_data, eod = getStockDataVec(stock_name, timesteps, dayTrading=dayTrading)
    agent = PVAgent(timesteps, market_data.shape[2]) # feature size 
    l = len(prices) - 1

    timer = Timer(episode_count)
    if Debug:
        print("DataLength: ", l)
        print("Features: ", market_data.shape[2])
        print("Timesteps to use: ", market_data.shape[1])
        #print("Eod: ", eod[-5:])
        
        
    pnl_track = []
        
        
    #position_state = np.array([1,0,0, 0.0 ]).reshape(1,4)
    position_state = np.zeros(constant.PositionStateWidth).reshape(1,constant.PositionStateWidth) 
    best_profit = 0.0

    for e in range(episode_count + 1):
        if (Debug):
            print("Episode " + str(e) + "/" + str(episode_count))
        #state = getState(data, 0, timesteps + 1)
        market_state = getState(market_data, 0)
        #print(state)
        
        #position_state = np.array([1,0,0, 0.0 ]).reshape(1,4) # initialize flat [Flat,Long,Short,PnL]  what the hell is turning this ... anyway , ints to float so maybe some extra check for nextposition state calculation
        position_state = np.zeros(constant.PositionStateWidth).reshape(1,constant.PositionStateWidth) # flat removed
        total_profit = 0
        agent.inventory = []

        state = [market_state, position_state]
      
        for t in range(l):
            action = agent.act(state)  # lets add position state later to the state, needed!, calculate restrictions there or here?, that's reason not to combine those yet, maybe next version... combined!
              # State could be just a list of [market_state, position_state, ....] and so on... as it might be useful to have different type of nn to them 

            # start from flat 
         
            next_market_state = getState(market_data, t + 1)  # 
            reward = 0
            
            next_position_state, immediate_reward, PnL = getNextPositionState(action, state[1][0], prices[t], prices[t+1], eod[t+1], eod[t] )  # FYI, state[1] = position_state, lets think this eod t or t + 1 again!!!!!! t + 1 is correct, eof bar at next state!
            #print("after", next_position_state)
      
            
            # let's make a very special reward, immediate (=augmented) + long Term PNL at next flat  , the REAL reward is the PnL 
            # the immediate reward is the next step reward, which can be counted immediately, this might  speed up calculations, but total profit is based to real PnL
            reward = immediate_reward + PnL
            #reward = PnL 
            
            #reward =  max(reward, 0)  #is this needed ???????, help please and the reason???
            total_profit += PnL
          
            #if Debug:
            #    if t % 200 == 0:
            #        print(PnL, next_position_state, immediate_reward)

            done = True if t == l - 1 else False
            next_state = [next_market_state, next_position_state.reshape(1,constant.PositionStateWidth)]
            #print(state, next_state)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            #position_state = next_position_state
            #state = [next_market_state, next_position_state]

            if done:
                #print("--------------------------------")
                print("Total Profit (Points) : {0:.4f}".format(total_profit))
                print ("Training left: ", timer.remains(e+1))
                print("--------------------------------")
                pnl_track.append(total_profit)  

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 100 == 0 or total_profit > best_profit:
            agent.model.save("models/" + stock_name + "_" + str(e) + ".h5")
            best_profit = total_profit
     
    # save the final mode   
    agent.model.save("models/" + stock_name + "_" + str(e) + ".h5") 

    plt.plot(pnl_track)
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.show()


####   here are the initialization, if called from command line
if __name__ == '__main__':
    import sys
    from agent.PVAgent import *
    from functions import *
    import constant
    import matplotlib.pyplot as plt
    
    if len(sys.argv) <= 2:
        print("Usage: python PVQTrain.py [stockfile] [episodes] [timesteps] ")
        exit(1)

    stock_name = sys.argv[1]
        
    episode_count = 1000
    timesteps = 4

    if len(sys.argv) >= 3:
        episode_count = int(sys.argv[2])
        
    if len(sys.argv) >= 4:
        timesteps = max(int(sys.argv[3]), 1)
    
    main(stock_name, episode_count, timesteps)

##############################
# own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################
