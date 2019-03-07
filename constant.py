# use this file to set some common "variables", could be instrument wise too
# 20190305 v1.0.0.1
# Programming marko.rantala@pvoodoo.com 
# 
 
# as many contract allowed together, FYI, current model trade those one by one , even reverse
# here, although some different models can be thought where this is not fixed but variable based to account value and so on...  
MAXCONTRACTS = 1


#actully previous could be implemented as 
#MAXLONGCONTRACTS = 1
#and 
#MAXSHORTCONTRACTS = 1
# but PVQEvaluate need  additional changes then, not only the obvious, if either one is 0 , unbalanced easy
# functions.py simple change would be enough

# slippage is included to commission!, check the datafile as it has TickSize info, usually you should have a slippage at least 1 Tick as well for Commission
# so set COMMISSION like 2*Ticksize mentioned in datafile, example, CL = TickSize 0.01 set COMMISSION 0.02
# Pointvalue is as info in the datafile too
COMMISSION = 0.0

POINTVALUE = 1.0   # just to get right USD value for trades, example from CL, Training side  "PointValue" = 1 but to give final results right, so no effect to training

# some important setups could be given given here , need to be implemented at getNextPositionState (now in functions.py)
#STOPLOSS 
#TARGET 

Debug=True

##############################
# own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################
