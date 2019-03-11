# RLDQTrading 

Check RLDataExample now, just format and features of the written datafiles with MLWriteRLTrainData
https://pvoodoo.blogspot.com/2019/03/writetraindata-tool-for-reinforcement.html

If you wonder the special Bar type, MBT, it is actually a special adaptive bar type, which can switch to next bar at next minute line if range or volume has been high.. (blog has some info of the adaptive bar)

Supervised models available too, check the blog.


## Files:
Additional files coming step by step:

PVQTrain.py = main train program, DayTrading model inside, 
Usage: python PVQTrain.py [stockfile] [episodes] [timesteps]
Example: python PVQTrain.py RLDataForCL30 1000 4

DayTrading:
Example: python PVQTrain.py RLDataForCL30D 1000 4 

PVQEvaluate.py to show and predict trades, out of data too
Usage: python PVQEvaluate.py [stockfile] [model]

Example: python PVQEvaluate.py RLDataForCL30 RLDataForCL30_1000 

Example: python PVQEvaluate.py RLDataOutOfDataForCL30D RLDataForCL30D_1000 

PVAgent.py keras model and reinforcement learning setup 

functions.py  Actually have an important function, getNextPositionState as it defines how the predicted actions are handled, now step by step, like no immediate reverse position from other position (has to go via Flat state)

constant.py Defines some important values like Comissions, need to be changed based to instrument 

[Further info of files & directories at blog](https://pvoodoo.blogspot.com/2019/03/example-of-reinforcement-learning.html?view=flipcard)

## Other:

Probably some notebook formats added to use the whole system in colab.research.google.com (keras, GPU)
FYI, No reason to run this with GPU as epoch = 1, faster with CPU model (data transfer do not delay)

## Resources:

[Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1807.02787)

