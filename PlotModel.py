# 
# example copypaste down, see the models directory for existence of model .h5 
# python PlotModel.py  RLDataForCL60D_2000 
# Programming marko.rantala@pvoodoo.com
# v1.0.0.1 20190322 
# reason to make a separate PlotModel instead of adding couple lines to PVQEvaluate is that I think that graphviz can be sort of problematic!?

import sys
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model


if len(sys.argv) != 2:
	print("Usage: python PlotModel.py  [model] ")
	exit()

model_name = sys.argv[1]
model = load_model("models/" + model_name +".h5")  # 

plot_model(model, to_file='models/' + model_name + '.png', show_shapes=True)

print("See the file: models/"+model_name + '.png')

#feature_count = model.layers[0].input.shape.as_list()[1]
#if Debug:
#    print(model.layers[0].input.shape.as_list())


##############################
# Own ad: For NinjaTrader related stuff: check https://pvoodoo.com or blog: https://pvoodoo.blogspot.com/?view=flipcard
##############################