# RLDQTrading 

Check RLDataExample now, just format and features of the written datafiles with MLWriteRLTrainData
https://pvoodoo.blogspot.com/2019/03/writetraindata-tool-for-reinforcement.html

If you wonder the special Bar type, MBT, it is actually a special adaptive bar type, which can switch to next bar at next minute line if range or volume has been high.. (blog has some info of the adaptive bar)

Supervised models available too, check the blog.


##Files:
Additional files coming step by step:

PVQTrain.py = main train program, let see if Day Trade progam is inside or separate

PVQEvaluate.py to show and predict trades, out of data too

PVAgent.py keras model and reinforcement learning setup 

functions.py  Actually have an important function, getNextPositionState as it defines how the predicted actions are handled, now step by step, like no immediate reverse position from other position (has to go via Flat)

constant.py Defines some important values like COmissions, need to be changed based to instrument 

##Other:

Probably some notebook formats added to use the whole system in colab.research.google.com (keras, GPU)

##Resources:

[Financial Trading as a Game: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/1807.02787)

