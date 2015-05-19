<<<<<<< HEAD
# Retina by *Deepmagic*
In case you guys didn't know.  This README.md is github flavored markdown (gfm), and can be used just like our google docs document, but allows us to keep the stuff in one place (and more importantly in our local git repo on our own computer)

[Markdown](https://guides.github.com/features/mastering-markdown/) is pretty awesome 

![retina_example](https://kaggle2.blob.core.windows.net/competitions/kaggle/4104/media/retina.jpg)

*Jonathan wrote on google docs:*

>###Fremgangsmåde
Vi skal have kortlagt en fremgangsmåde vi udvikler vores model på.

>Jeg foreslår følgende:
####Sæt en benchmark
1. Byg en simpel model, expand den til de overfitter, gå ét step tilbage og kald det fit, gå 2 steps tilbage og kald det underfit.
2. Dette skulle gerne resultere i 3 modeller, en der overfitter, en der fitter og en der underfitter.
3. Hver af benchmark modellerne skal have CV for 100(underfit), 200(fit), 500(overfit) epochs.
>####Ved nye tiltag(én af gange, eller i anden struktureret orden)
1. Test det nyt tiltag med de 3 modeller for de 3 forskellige epochs.
2. Dette sikre os “sanity check” - er tiltaget udviklet korrekt?
3. Samt ydeevne - giver det os noget?

>####Dokumentation
1. Ved hver test skal der opbygges en log for performance 


>###Model Arkitektur
*“going to 512x512 straight away is a pretty big jump.
Would maybe try 256x256 first, with a large stride at the bottom of the image, and maybe go all-convolutional with some kind of global max pooling at the top i.e. replace the fully connected layers with 1x1 convolutions the architecture.
Will need plenty of experimentation to get right, I reckon.”*

>###Loss function(I sværheds rækkefølge)
1. MSE
2. Clip MSE
- QW Kappa(ligner umildbart Clip MSE meget)
- Ordinal classification (when it’s MSE cost but a classification task) 
>http://www.kaggle.com/c/diabetic-retinopathy-detection/forums/t/13115/paper-on-using-ann-for-ordinal-problems
=======
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







>>>>>>> 15de43acb190602c80cd80f0a54d2c4bdc818bfa
