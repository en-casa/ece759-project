%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

%}

clear; close all;
addpath('utility', 'MNIST', 'MNIST/data', 'MNIST/loadMNIST', ...
	'decision tree');

seed = 152039828;
rng(seed); % for reproducibility

% define parameters

% partition data

%{
	1/2 of the dataset should be for training
	the other for testing
	
	MNIST contains 70k examples
%}

[train, test] = loadMNIST(35e3);

%% dimensionality reduction / feature generation
% via prinicpal component analysis (pca) (svd)

sz = 28;

% takes a few secs
for i = 1:length(train{2})
	%[train{3}{1,i}, train{3}{2,i}, train{3}{3,i}] = svd(reshape(train{1}(:,i),[sz, sz]));
	%train{3}{2,i} = diag(train{3}{2,i});
	[~, S, ~] = svd(reshape(train{1}(:,i),[sz, sz]));
	train{3}(:,i) = diag(S);
end

% average singular value for each digit
for i = 1:9
	inds = find(train{2} == i);
	digitAvg(:,i) = mean(train{3}(:,inds),2);
end

clear i;

%% train

tree = trainDecisionTree(train);

%% test
