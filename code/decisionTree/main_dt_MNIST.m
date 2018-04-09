%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

%}

clear; close all;
addpath('utility', 'MNIST', 'MNIST/data', 'MNIST/loadMNIST', 'lda');

fprintf('begin MNIST decision tree script\n');

% hyper-parameters
N_tr = 35e3; % training samples
N_te = 35e3; % test samples

% for feature selection
usePCA = false;
if usePCA
	numFeatures = 30;
else % use LDA
	numFeatures = 10;
end

% for decision tree
minLeaf = 1; % to prevent overfitting

%seed = 152039828;
%rng(seed); % for reproducibility

% partition data
% MNIST contains 70k examples
[train, test] = loadMNIST(N_tr);

%% dimensionality reduction / feature generation
% via prinicpal component analysis (pca) (svd)
st = cputime;

if usePCA
	[train, U, V] = pca_(train, numFeatures);
else
	[train, test] = lda_features(train, test, 0:numFeatures-1);
end

fprintf('Features Generated in %4.2f minutes\n', (cputime - st)/60);

%% train
st = cputime;

tree = trainDecisionTree({train{2:3}}, minLeaf);

fprintf('Trained in %4.2f minutes\n',(cputime - st)/60);

%% test
st = cputime;

if usePCA
	test{3} = (U'*test{1}'*V)';
	test{3} = test{3}(1:numFeatures,:);
end
	
test = testDecisionTree(test, tree);

fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

% Classification Error
errors = nnz(test{2}(:,1) ~= test{2}(:,2));
errorRate = (errors/N_te)*100;

filename = sprintf('mnist_tree%2.0f_%d.mat', errorRate, minLeaf);
%save(filename, 'tree');
%filename = sprintf('UV%2.0f.mat', errorRate);
%save(filename, 'U', 'V', '-v7.3');

fprintf('numFeatures: %d, minLeaf: %d, error rate: %2.2f\n', ...
	numFeatures, minLeaf, errorRate);