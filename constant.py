# use this file to set some common "variables", could be instrument wise too
# 20190305 v1.0.0.1
# Programming marko.rantala@pvoodoo.com 
# 
##############################
# own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################
 
# as many contract allowed together, FYI, current model trade those one by one , even reverse

MAXCONTRACTS = 1
# slippage is included to commission!, check the datafile as it has TickSize info, usually you should have a slippage at least 1 Tick as well for Commission
# so set COMMISSION like 2*Ticksize mentioned in datafile, example, CL = TickSize 0.01 set COMMISSION 0.02
# Pointvalue is as info in the datafile too
COMMISSION = 0.0

POINTVALUE = 1000   # just to get right USD value for trades, example from CL, Training side  "PointValue" = 1 but to give final results right, so no effect to training

Debug=True


