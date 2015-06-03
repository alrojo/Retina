# Deep Magic and the Leaky ReLUs
## Week 8 - Group formed
- Read for next week:
    + Chapt. 1-3 in Alex Graves: http://www.cs.toronto.edu/~graves/preprint.pdf
    + Chapt. 5 in Bishop.
- Sample lectures from Coursera neural networks course: https://www.coursera.org/course/neuralnets
- Install Theano http://deeplearning.net/software/theano/
- Attend atari deepmind thesis presentation next monday 12:30

## Week 9
- We need to find a specific topic to focus on.  Evaluate kaggle competitions for relevance for RNN, and investigate whether deepmind's atari project is too comprehensive.
- Get amazon instance up and running, and runt through Daniel Nouri's blog on theano/lasagne on amazon http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
- Attend the kaggle workshop the 12th of March.

## Week 10
- Finish setting up an amazon AMI
- Run through theano tutorial for multilayer perceptron 
- http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
- Attend kaggle workshop next week

## Week 11
- Attended workshop
- Project focus settled on implementing a convolutional neural network to compete in the kaggle diabetic retinopathy challenge.
- For next week: play with Dieleman's code.  Also view his galaxy zoo code
  https://github.com/benanne/kaggle-galaxies

## Week 12
- Read all of http://cs231n.stanford.edu/ and run exercises if possible
- Read http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf

## Week 13
- Bennane plankton solution just published, read: http://benanne.github.io/2015/03/17/plankton.html
- Nouri's nolearn is not an option, go benanne!
- 96x96 images are too small, we have to preprocess our own.  
- Sign up for cuda developers to get access to cudnn

## Week 14
- Vacation and catch up

## Week 15
- Fix Amazon AMI, cudnn not working.
- Preprocessing is still not up.

## Week 16
- Preprocessing works for sample images.
- Amazon still has problems.  GTK dependent (matplotlib, skimage) features don't work without an X-server

## Week 17
- Amazon problems fixed.  
- Makefile created to authenticate and download data directly to EBS volume.
- Get baseline working!

## Week 18
- Get baseline working!
- Read
    + http://arxiv.org/abs/1409.1556
    + http://arxiv.org/abs/1311.2901
    + http://arxiv.org/abs/1312.6120
    + http://web.stanford.edu/class/polisci203/ordered.pdf
    + This book seems good https://www.iro.umontreal.ca/~bengioy/DLbook/

## Week 19
- Images processed
- Image runs, but kappa function needs to be fixed.  Currently the model doesn't learn
- Investigate possibility for computing on multiple GPUs.  Amazon just launched their g2.x8large

## Week20
- Training on multiple GPUs is out of scope for now, but gridsearching over several of the g2.x8large seems easy
- Incorporate eye correlation
- Get submission generation working
- Poster

## Week 21
- Presentation

## Week 22 -> 
- TSNE would be nice
- Finish eye correlation
- Report

