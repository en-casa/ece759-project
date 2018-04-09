%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the cross validation of
the decision tree classifier 
across all hyperparameters, 
taking performance and time results along the way

features are generated using lda, as they yield the best performance

%}

clear; close all;
addpath('utility', 'MNIST', 'MNIST/data', 'MNIST/loadMNIST', 'lda');

fprintf('begin Cross Validation on MNIST decision trees\n');

% hyper-parameters
k = 5; % k-fold cross validation
% use k to partition data
N_te = 70e3/k;
N_tr = 70e3 - N_te;

% for lda
numFeatures = 10;

% for decision tree
minLeaves = 1:1:10;

% partition data
% MNIST contains 70k examples
[train, test] = loadMNIST(N_tr);

% todo: merge sets, generate indices for each fold


%% k-fold cross validation across minLeaf

trainTimes = zeros(minLeaves(end), k);
errorRates = zeros(minLeaves(end), k);

for minLeaf = minLeaves
	
	for fold = 1:k
		
		% todo: choose fold
		
		% dimensionality reduction / feature generation
		[train, test] = lda_features(train, test, 0:numFeatures-1);
		
		% train
		st = cputime;
		tree = trainDecisionTree({train{2:3}}, minLeaf);
		thisTrainTime = (cputime - st)/60;
		fprintf('Trained in %4.2f minutes\n', thisTrainTime);
		trainTimes(minLeaf, fold) = thisTrainTime;
		
		% test
		test = testDecisionTree(test, tree);

		% Classification Error
		thisError = nnz(test{2}(:,1) ~= test{2}(:,2));
		thisErrorRate = (thisError/N_te)*100;
		errorRates(minLeaf, fold) = thisErrorRate;

		fprintf('numFeatures: %d, minLeaf: %d, error rate: %2.2f\n', ...
			numFeatures, minLeaf, thisErrorRate);
		
	end
	
end