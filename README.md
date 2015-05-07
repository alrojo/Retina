# Retina
DeepMagic

Starting point tomorrow.  Implement somthing like `convroll4` see end of *doc.pdf*


## Morten Development

### main.py
I'm thinking this should start the simulation.  So far it only contains a variable to control how much of the data is loaded in.  I thought this would be nice while debugging to control running time. 

#### TODO related to main.py
- [ ] Implement cli arguments and method like in theano logistic regression such that it can start "by itself"

### data.py
This module so far has all the paths, and creates a training, validation, and test set which (to preserve memory) are simple arrays of paths.

#### TODO related to data.py
- [ ] I don't think my class of paths is optimum.  I think this could possible be done better by creating a class in its own module and show all the variables shared. At the moment I don't know what happens when new instances are created (and whether they will be created).  I think it should be created as global/shared instance...instead of being a generator.  
- [ ] Make sure order and any sorting/lack thereof works. 

### file_creation.py
This module is implement to speed up loading of data


### fun.py
This module I'm thinking should contain any function.  At the moment it just creates any function I so far have stumbled into in the benanne_retina and plankton projects -- concerning these function this module is certainly not exhaustive.

So far it contains 
1. load  *Load all images into memory for faster processing*
2. save *Handles the saving of data. Saves a file given its name, value, and path to save.  


#### TODO related to fun.py
- [ ] *load* is importing images as grey scale, this is wrong.







