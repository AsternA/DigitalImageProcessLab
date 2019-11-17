I = imread('pout.tif');

Inoise = imnoise(I, 'gaussian', 0, 0.01);
Isub = I - Inoise;

figure();
subplot(2,3,1); imshow(I); title('original image');
subplot(2,3,2); imshow(Inoise); title('image with noise');
subplot(2,3,3); imshow(Isub); title('substraction');