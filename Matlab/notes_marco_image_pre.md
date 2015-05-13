
### Marco matlab files
1. tests roation by seeing whether height is bigger than width.  If so rotates the images.  I should run through the images to test for this
2. Removes noise from the BLACK background...I guess this makes sense as we shouldn't be concerned with the background
    1. get mean of third dimension (the colors).  
    2. Divide all means (shape of image, dimension 1 and 2) by the max mean
    3. defines v as the mean_normalized values in the 50% center (end/5:end*3/4) in both the first and second dimension
    4. takes median of v
    5. divides mean_normalized(call it ma from now on) by median.
    6. finish off by doing this `  ma = uint8(ma > 30/250);` This converts it to uint8 but what's up with the boolean argument?  Check it out. 
3.  Remove small connected components...whoa, how bout no! 
4. Detects edges in the image...I don't know these functions I should definitely check this out. 
5.  finds outer edges in the eye and fits a circle to these, I should also check this out
    1. compute distance of circle and refine the fit ... why remove points inside.  is the refining of the fit to fit the circle again?
    2. I'm lost from now on, I need to play with it in matlab

