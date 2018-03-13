 %{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

features are generated using principal compenent analysis

%}

clear; close all;
addpath('utility', 'YaleB', 'YaleB/data');

fprintf('begin MNIST decision tree script\n');

seed = 152039828;
rng(seed); % for reproducibility

%{
	partition data

	1/2 of the dataset should be for training
	the other for testing
	
	YaleB contains 2414 examples
%}
N_tr = 2414/2; % training samples
N_te = N_tr; % test samples

[train, test] = loadYaleB(N_tr);

%% dimensionality reduction / feature generation
% via prinicpal component analysis (pca) (svd)
st = cputime;

numFeatures = 60;
[train, U, V] = pca_(train, numFeatures);

fprintf('Features Generated in %4.2f minutes\n', (cputime - st)/60);

%% train
st = cputime;

minLeaf = 1; % to prevent overfitting
tree = trainDecisionTree({train{2:3}}, minLeaf);

fprintf('Trained in %4.2f minutes\n',(cputime - st)/60);

%% test
st = cputime;

test{3} = (U'*test{1}'*V)';
test{3} = test{3}(1:numFeatures,:);

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
 					 
numFeatures   : 
minLeaf       : 
Error Rate    : 
Mins To Train : 

, - all attributes were considered for the best information gain for a
particular set.

%}