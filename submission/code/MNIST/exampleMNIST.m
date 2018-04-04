%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

Load MNIST and show some examples
	
	MNIST source: 
	http://yann.lecun.com/exdb/mnist/

%}

clear;
close all;

addpath('./utility');
addpath('MNIST', './MNIST/data', './MNIST/loadMNIST');

[train, test] = loadMNIST();

% print some images
sz = 28;

images = 10;

I = zeros(images*28);

for i = 0:images
	row = i*sz + 1;
	for j = 0:images
		col = j*sz + 1;
		I(row:row+sz-1, col:col+sz-1) = reshape(train{1}(:,i*images+j+1),[sz, sz]);
	end
end

f = instantiateFig(1);
imshow(I);
prettyPictureFig(f);
