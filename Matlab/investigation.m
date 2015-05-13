clear
close all
clc
ii = 1
% Set resolution
res=512;  
development = true
doplot = false;
doplot = true;

% Path of the images
% img_folder='C:\train';
img_folder='/home/morten/Git_and_dropbox_not_friends/Retina/sample';

% mkdir(fullfile(img_folder,['outimages' sprintf('%i',res)]))

%% Get images in the folder
fl = dir(fullfile(img_folder,'*.jpeg')); 
% load('dirlist')

% fl(cc).name = '29984_left.jpeg'; cc=cc+1;
% fl(cc).name = '35058_left.jpeg'; cc=cc+1;
if development 
    fl = fl(1);
end


   
%% Get the image
% fprintf('%s. Processing %i of %i\n',fl(ii).name, ii,length(fl));
A = imread(fullfile(img_folder,fl.name));

if size(A,1)>size(A,2) % 15840_left.jpeg in the test set is wrongly rotated
    A=rot90(A);
end
    
%%

ma = mean(A,3); ma = ma / max(ma(:));
v = ma(round(end/4:end*3/4),round(end/4:end*3/4));
median(v(:));
ma = ma / median(v(:));

subplot(2,1,1);     
B = mean(A,3) > 30; 
imshow(B)

subplot(2,1,2);    
B = ma > 30/250; 
imshow(B)

ma = uint8(ma > 30/250);

%% Remove small connected components
cc = bwconncomp(ma); 
for j=1:length(cc.PixelIdxList)
    lp = cc.PixelIdxList{j};
    if length(lp) < 100, 
        ma(lp) = 0;
    end
end

%% Detect edges in the image
B = edge(mean(bsxfun(@times, A, ma)  ,3),'canny');
if doplot,
   imshow(A)
end
[h,w] = size(B);