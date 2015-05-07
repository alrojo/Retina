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
