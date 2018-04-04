%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

returns:
	train
		1x2 cell
		60e3 images - 28x28 pixels in row-major order in columns
		60e3 labels - integers in  the set [0,9]
	test
		1x2 cell
		10e3 images - 28x28 pixels in row-major order
		10e3 labels - integers in  the set [0,9]

%}

function [train, test] = loadMNIST(N_train)

	if nargin == 0
		N_train = 60e3;
	end

	imagesTrain = loadMNISTImages('train-images-idx3-ubyte');
	labelsTrain = loadMNISTLabels('train-labels-idx1-ubyte');

	imagesTest = loadMNISTImages('t10k-images-idx3-ubyte');
	labelsTest = loadMNISTLabels('t10k-labels-idx1-ubyte');

	all{1} = [imagesTrain imagesTest];
	all{2} = [labelsTrain; labelsTest];
	
	% random indices
	inds = randperm(length(all{2}));
	all{1} = all{1}(:, inds);
	all{2} = all{2}(inds);
	
	train = {all{1}(:, 1:N_train) all{2}(1:N_train)};
	test = {all{1}(:, N_train+1:end) all{2}(N_train+1:end)};

end