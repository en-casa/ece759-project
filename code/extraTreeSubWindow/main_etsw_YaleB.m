%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the extra-tree classifier

there are no features used besides the raw pixels

we augment the set by taking subsets of the original images

%}

clear; close all;
addpath('utility', 'YaleB', 'YaleB/data');

fprintf('begin Yale B extra tree with sub-window script\n');

% hyper-parameters
N_tr = 2000; % training samples
N_te = 2414 - N_tr; % test samples

% for extra tree
minLeaf = 1; % to prevent overfitting
numTrees = 100; % ensemble for majority voting

%seed = 152039828;
%rng(seed); % for reproducibility

% partition data
% MNIST contains 70k examples
[train, test] = loadYaleB(N_tr);

% features are the raw pixels, so we reorder the cell
train = {train{2}, train{1}};
test = {test{2}, test{1}};

%% extract subwindows to augment the set
numWindows = 4;
% this will introduce 4x more samples with 504 pixels.
train = extractSubwindows(train, numWindows);
test = extractSubwindows(test, numWindows);

%% train
st = cputime;

% create an ensemble of random trees
trees = cell(numTrees, 1);
for tree = 1:numTrees
	
	fprintf('tree: %d\n', tree);
	trees{tree} = trainExtraTree(train, minLeaf);
	
end

fprintf('Trained %d extra-trees in %4.2f minutes\n', numTrees, (cputime - st)/60);

%% test
st = cputime;
	
test = testExtraTrees(test, trees);

fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

% Classification Error
errors = nnz(test{1}(:,1) ~= test{1}(:,2));
errorRate = (errors/N_te)*100;

filename = sprintf('mnist_extratree%2.0f_%d.mat', errorRate, minLeaf);
%save(filename, 'trees');

fprintf('minLeaf: %d, error rate: %2.2f\n', ...
	minLeaf, errorRate);

%% Test Results
%{
                 ,  *  *   *   &   &    ,    ,    ,    ,        ,
numFeatures   : 30 30 60 200 200 200   30  100  100   20 10 (lda)
minLeaf       :  1  1  2   2   3   4    4    4    1    1        1
Error Rate    : 22 26 26  26  23  24 22.6 23.6 23.5 22.9    18.42
Mins To Train : 48  2  2   3  12  12   47   60  102   11        3

, - all attributes were considered for the best information gain for a
particular set.

* - for these experiments, the decision tree was grown by considering
only the first attribute in the set. all things held equal, this was simply
an attempt to reduce the computational complexity.

& - if the number of attributes was > 5, consider only the first 5, else,
consider all attributes (1-5)

%}