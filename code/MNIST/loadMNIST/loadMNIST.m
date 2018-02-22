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

function [train, test] = loadMNIST()

	imagesTrain = loadMNISTImages('train-images-idx3-ubyte');
	labelsTrain = loadMNISTLabels('train-labels-idx1-ubyte');

	train = {imagesTrain, labelsTrain};

	imagesTest = loadMNISTImages('t10k-images-idx3-ubyte');
	labelsTest = loadMNISTLabels('t10k-labels-idx1-ubyte');

	test = {imagesTest, labelsTest};

end