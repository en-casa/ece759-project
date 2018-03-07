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

%% dimensionality reduction / feature generation (~3 minutes)
% via prinicpal component analysis (pca) (svd)
st = cputime;

numFeatures = 200;
[train, U, V] = pca_(train, numFeatures);

fprintf('Features Generated in %4.2f minutes\n', (cputime - st)/60);

%% train (~50 minutes)
st = cputime;

minLeaf = 4; % to prevent overfitting
tree = trainDecisionTree({train{2:3}}, minLeaf);

fprintf('Trained in %4.2f minutes\n',(cputime - st)/60);

%% test (~2 minutes)
st = cputime;

%test = pca_(test, numFeatures);

test{3} = (U'*test{1}'*V)';

test = testDecisionTree(test, tree);

fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

% Classification Error
errors = nnz(test{2}(:,1) ~= test{2}(:,2));
errorRate = (errors/N_te)*100

filename = sprintf('tree%2.0f.mat', errorRate);
save(filename, 'tree');
%filename = sprintf('UV%2.0f.mat', errorRate);
%save(filename, 'U', 'V', '-v7.3');

fprintf('numFeatures: %d, minLeaf: %d, error rate: %2.2f\n', ...
	numFeatures, minLeaf, errorRate);

%% Test Results
%{
					  ,  *  *   *   &   &   ,
numFeatures   : 30 30 60 200 200 200 200
minLeaf       :  1  1  2   2   3   4   4
Error Rate    : 22 26 26  26  23  24  
Mins To Train : 48  2  2   3  12  12  

, - all attributes were considered for the best information gain for a
particular set.

* - for these experiments, the decision tree was grown by considering
only the first attribute in the set. all things held equal, this was simply
an attempt to reduce the computational complexity.

& - if the number of attributes was > 5, consider only the first 5, else,
consider all attributes (1-5)

%}