# Plankton 

##TOC
- [TODO](#TODO)
- [Links](#Links)
- [Possibly useful for retina challenge](#Retina tips)
- [I'm stupid (...but I'm improving ;-)...)](#I'm stupid)
- [Curiosities](#Curiosities)
- [Questions of Life](#Questions of Life)

##TODO <a id = "TODO"></a>
- [ ]  Read and implement this https://www.kaggle.com/c/datasciencebowl/details/tutorial

- [ ]  Understand code
    1. ** *Start* ** with **Data.py**
    2. **create_data_files.py**
    3. **Load.py**
    - **train_convnet.py <config_name>**
    - *Configuration files*  Understand the `resume_path` feature.  It's used by **train_convnet.py**  Should allow an interrupted training run to resume.  
    - Browse **nn_plankton.py**  Browse to see if anything is important.
    - **predict_convnet.py**
    - **extract:features.py** **compute_image_moment_stats.py**
    - **create_validation_split.py** and **create_baggin_validation_split.py**
    - 

    * [ ] **Data.py** module.  It handles laoding, pre-processing and augmentation of the images.  Try to convert as much as possible directly to Retina.  This module also handles scikit-image, which you should learn to play with
    * [ ] **buffering.py** I think this is optional.  This is basically so data augmentaion and model training can happen in parallel by manipulating threads.
    * [ ]  **dihedral.py** Optional.  Implements cyclic pooling and rolling. 
    * [ ]  **diheadral_fast.py** *dihedral_ops.py**  This is definitely optional.  Custom CUDA kernels to make cyclic rolling faster. 
    * [ ]  **tt.py** and **icdf.py**  Optional.  Deals with randoms and distributions
    * [ ]  **load.py** I think this one is relevant.  Basically describes classes for loading inputs.  These classes are instantiated in the configuration files.  Is used to create data generators. Used in train_convnet.py
    * [ ]  **nn_plankton.py**  Browse to see if anything is important.  Extensions they made to the Lasagne library.
    * [ ]  **tmp_dnn.py** cuDNN-based alyers for Lasagne.  Legacy code as it has been merged into main library(of lasagne I suppose).  Some of the plankton code still depends on them.  Could be interesting to have a look at them...for understanding cuDNN
    * [ ]  **utils.py** miscelaneous...browse for importance
    * [ ]  **configurations/*.py*  For sure.  The intelligence behind the machine
    * [ ]  **create_data_files.py** Script to creat teh numpy files tha contain the training and testing images and the training labels.  This enables faster loading of the data. 
    * [ ]  **train_convnet.py <config_name>** Trains the model   
    * [ ]  **predict_convnet.py** Generate prediction
    *  I can't be bothered to do this for all the files.  check **doc.pdf** if you want to know the rest.
- [ ] Seems rather complicated to start the training, running the models.  See doc.pdf for instructions.    
 


##Links <a id = "Links"></a>

- [Blog about winning solution](http://benanne.github.io/2015/03/17/plankton.html) and associated [github](https://github.com/benanne/kaggle-ndsb)
- [ghalton --quasi random number generator](https://github.com/fmder/ghalton)
- [OxfordNet, *Very Deep Convolutional Networks for Large-Scale Image Recognition*](http://arxiv.org/abs/1409.1556)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification])http://arxiv.org/abs/1502.01852)
-[ImageNet Classification with Deep Convolutional
Neural Networks] (http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
-[Adam: A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980)
-[Exact solutions to the nonlinear dynamics of learning in
deep linear neural networks](http://arxiv.org/abs/1312.6120)
-[Chapter: Stacked Convolutional Auto-Encoders for Hieracrchical Feature Extraction](http://link.springer.com/chapter/10.1007/978-3-642-21735-7_7)from the book *Artificial Neural Networks and Mchine Learning - ICANN 2011* (Masci, Meier, Ciresan, Schmidhuber) ( I think I might have it, or that digital library does)
-[Distilling the Knowledge in a Neural Network](http://arxiv.org/abs/1503.02531)
-[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)
[Self-informed neural network structure learning](http://arxiv.org/abs/1412.6563)




##Possibly useful for retina challenge <a id = "Curiousitie"></a>
- Concerning data augmentation.  Since we do not quite know what to look for(maybe it would make sense to do some medical research on this), possible rotations could be useful -- maybe it's veins or something that gave away signs of the disease and as such, and I assume this go in all directions. 
- Remember benanne won the galaxy zoo challenge too.  Maybe that code contains some nice stuff to be used.  Possibly Kaggle-eyes too.
- Cyclic pooling (and why only try four rotations, throw in all in there m8).  I guess also rotating the images inside the layer speeds up the computation time as this is possible done inside the gpu, discarding some gpu bandwidth latency.  They pooled the images, how about trying with just stacking and more weights like for the galaxy zoo?  If the augmentation happens inside the gpu and with decreases latency problem, then the extra weights should be no problem if you use a model like Alex Krizhevsky's giant 6 gpu running [cuda-convnet2](https://code.google.com/p/cuda-convnet2/)
- Rotating an image 90 and 180 degrees is basically free...it's just flipping and transposing
- For deep layer models (10+) they had great sucess with * **Leaky ReLUs** * `y = max(x, 0)` instead of `y = max(x, a*x)` making `a` a tunable parameter ...parameterazied ReLUs.  They ended up settling on  `a=1/3.0`
- Some times a tweak does not affect the predictive performance exactly, but allows the model to be extended by mitigating overfitting.
- To allow for larger images considere making the pooling filters larger and using bigger strides. This done over several layers can significantly allow for greater images at same computation cost. 
- Combine networks focusing on different features.  Like a separate network worrying more about the size. 
Late fusing (search term in [benanne plankton blog](http://benanne.github.io/2015/03/17/plankton.html)) Something about using the extracted features in post small neaural networks...but I don't quite get how this differs from fully connected layers.
- Solve added variance from randomly initiated weights by training the same network several times and uniformly blending the learned weights. 

##I'm stupid (...but I'm improving ;-)...) <a id = "I'm stupid"></a>
- Why would Benanne use the ghalton random number generator?  What's so bad about the one in numpy?
- They use cyclic pooling and rolling feature maps.  I get both, but I don't understand why they say the rolling feature maps are cheap since the feature maps are already being computed--wouldn't this cost something in having to compute that many more feature maps( or is the benefit from the weight sharing?) -- FIND OUT!  Maybe it relates to the fact that since the images are already inserted in the four orientations and thus their computations can be shared to another orientations roated filter feature map?
- I don't understand the difference between late fusing (search for term in this page) and fully connected end layers. 

##Curiosities <a id = "Curiousities"></a>
- Optimising the loss function is correlated with improved classification accuracy but not exactly the same.  *"Although the two are obviously correlated, we paid special attention to this because it was often the case that significant improvements to the log loss would barely affect the classification accuracy of the models"**
- During data augmentation.  The parameters for the augmentation can be sampled from different distributions. Uniform, gaussian etc. 
- Keep track of the progress of your implementations.  Have a spreadsheet to note your performance over time.
- I like their term for zero mean and unit variance (ZMUV)
- Seems knowing how to combine PyCUDA with Theano is essential to maintaining speedy code 
- Weight decay is usefull for not just regularization for predictive performance but also for stabilizing large models.  Large models can tend to diverge unless the learning rate is deceased considerably(which unfortunately sometimes is not feasible since it would slow down things too much)
- Unsupervised pre-training sounds awesome.  Like unconvolve something and then train to convolve it.  Kinda creating fake labels. See the link for *Stacked Convolutional Auto-Encoders for Hieracrchical Feature Extraction* above.
- In my quest for knowledge (articles and books I'll never read)...search for stuff cited/written by your machine learning heros. Hinton ftw!
- Specialist learnings ... not really relevant to Kaggle as only 4 labels exist, but maybe relevant to thesis. 


##Questions of Life <a id = "Questions of Life"></a>
- Study loss functions.  Is the cross entropy the same as the log function for a multiclass classfication problem? 
- I should learn PyCUDA and cuDNN...I don't want to be stuck in theano
- Learn about [image moments](http://en.wikipedia.org/wiki/Image_segmentation). This can potentially be used for not just preprocessing but also to add features to train on.
- What is elastic distortions.  Hint: data augmentation.
- I'm tired of lacking more abstract/theoretical math...I want to know about group theory like [cyclic groups](http://en.wikipedia.org/wiki/Cyclic_group)

