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


function [train, test] = loadYaleB(N_train)

	if nargin == 0
		N_train = 1207; % half the data
	end

	load('YaleB.mat')
	
	% concatenate all images
	labels = [];
	images = [];
	for i = 1:size(YaleB,1)
		
		theseImages = YaleB{i,2};
		thisLabel = i*ones(size(YaleB{i,2},2),1);
		
		images = [images, theseImages];
		labels = [labels; thisLabel];
		
	end
	
	% random indices
	inds = randperm(length(labels));
	images = images(:, inds);
	labels = labels(inds);
	
	% construct structs
	train = {images(:, 1:N_train) labels(1:N_train)};
	test = {images(:, N_train+1:end) labels(N_train+1:end)};
	
end