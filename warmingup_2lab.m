%Warming Up Questions for Lab #2: Almog Stern and Joanna Molad
clear all; close all;

%1.1.2 - Added Brightness
I = imread('tire.tif');
subplot(2,3,1);imshow(I); title('Original Image');
subplot(2,3,4);imhist(I); title('Original Image Hist'); 

I_addB= I+50; 
subplot(2,3,2);imshow(I_addB); title('Added Brightness Image');
subplot(2,3,5);imhist(I_addB); title('Added Brightness Hist Image');

I_subB = I_addB-50;
subplot(2,3,3);imshow(I_subB); title('Removed Brightness Image');
subplot(2,3,6);imhist(I_subB); title('Removed Brightness Hist Image');

%%
%1.1.3 - Added Contrast
close all; clear all;

I = imread('tire.tif');
subplot(2,3,1);imshow(I); title('Original Image');
subplot(2,3,4);imhist(I); title('Original Image Hist'); 

I_addC= 0.4*I; 
subplot(2,3,2);imshow(I_addC); title('Added Contrast Image');
subplot(2,3,5);imhist(I_addC); title('Added Contrast Hist Image');

I_subC = I_addC/0.4;
subplot(2,3,3);imshow(I_subC); title('Removed Contrast Image');
subplot(2,3,6);imhist(I_subC); title('Removed Contrast Hist Image');


%%
%1.2 - Random Signals PDF and CDF
close all; clear all;

L = 1000;
X = unidrnd(256,1,L);
Xhist = imhist(uint8(X));
sx = numel(X);
Xpdf = Xhist/sx;
Xcdf = cumsum(Xpdf);

figure(1);
subplot(2,2,1); plot(X); title('Random Discrete Uniform Signal');
subplot(2,2,2); plot(Xhist); title('Histogram of Discrete Signal'); xlim([0 400]); %ylim([0 50]);
subplot(2,2,3); plot(Xpdf); title('PDF of Random Uniform Signal'); xlim([0 250]); ylim([0 0.015]);
subplot(2,2,4); plot(Xcdf); title('CDF of Random Uniform Signal'); xlim([0 250]); ylim([0 1]);

mu = 200; sigma = 30;
norm = normrnd(mu , sigma, [1 L]);
Ghist = imhist(uint8(norm));
Gpdf = Ghist/(numel(norm));
Ncdf = cumsum(Gpdf);

figure(2);
subplot(2,2,1); plot(norm); title('Normal Uniform Signal');
subplot(2,2,2); plot(Ghist); title('Histogram of Gaussian Signal'); xlim([0 400]); %ylim([0 50]);
subplot(2,2,3); plot(Gpdf); title('PDF'); xlim([0 300]); ylim([0 0.015]);
subplot(2,2,4); plot(Ncdf); title('CDF'); xlim([0 300]); ylim([0 1]);

%%
%1.3 - Transformation
close all; clear all;

L = 1000;
x = linspace(0, 255, L);
T(1:L) = 2*x;
fx = T;






