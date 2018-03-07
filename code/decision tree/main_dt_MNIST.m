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

fprintf('begin MNIST decision tree script\n');

seed = 152039828;
rng(seed); % for reproducibility

% partition data

%{
	1/2 of the dataset should be for training
	the other for testing
	
	MNIST contains 70k examples
%}
N_tr = 35e3; % training samples
N_te = 35e3; % test samples

[train, test] = loadMNIST(N_tr);

%% dimensionality reduction / feature generation
% via prinicpal component analysis (pca) (svd)
st = cputime;

numFeatures = 20;
train = pca_(train, numFeatures);

fprintf('Features Generated in %4.2f minutes\n', (cputime - st)/60);

%% train ()
st = cputime;

% tree is about 1MB
% TODO: add arg for tree depth (avoid overfitting)
tree = trainDecisionTree({train{2:3}});

fprintf('Trained in %4.2f minutes\n',(cputime - st)/60);

%% test (~18 secs)
st = cputime;

numFeatures = 20;
test = pca_(test, numFeatures);

test = testDecisionTree(test, tree);

fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

% Classification Error
errors = nnz(test{2}(:,1) ~= test{2}(:,2));
errorRate = (errors/N_te)*100

filename = sprintf('tree%2.0f.mat', errorRate);
save(filename);
