import numpy as np
import glob 
import os
import inspect

import main, fun 
reload(main)
reload(fun)
import main, fun

print main.paths.sample_dir
for i in inspect.getmembers(main.paths,fun.isString):
	print i
main.paths.sample_paths = glob.glob(os.path.join(main.paths.sample_dir,"*"))




#/home/morten/Dropbox/Advanced_Machine_Learning/Retina/data/sample
#/home/morten/Dropbox/Advanced_Machine_Learning/Retina/code/data/sample