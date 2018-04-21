%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

%}

function [errorRate] = demonstration_dt_MNIST()

	addpath('utility', 'MNIST', 'MNIST/data', 'MNIST/loadMNIST', 'lda', ...
		'decisionTree');

	fprintf('begin MNIST decision tree demonstration\n');

	% hyper-parameters
	N_tr = 35e3; % training samples
	N_te = 35e3; % test samples

	% for feature selection
	numFeatures = 10;

	% for decision tree
	minLeaf = 1; % to prevent overfitting

	% partition data
	% MNIST contains 70k examples
	[train, test] = loadMNIST(N_tr);

	%% dimensionality reduction / feature generation
	% via linear discriminant analysis (lda)
	st = cputime;
	
	[train, test] = lda_features(train, test, 0:numFeatures-1);

	fprintf('Features Generated in %4.2f minutes\n', (cputime - st)/60);

	%% train
	st = cputime;

	tree = trainDecisionTree({train{2:3}}, minLeaf);

	fprintf('Trained in %4.2f minutes\n',(cputime - st)/60);

	%% test
	st = cputime;

	test = testDecisionTree(test, tree);

	fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

	% Classification Error
	errors = nnz(test{2}(:,1) ~= test{2}(:,2));
	errorRate = (errors/N_te)*100;

	fprintf('\nnumFeatures: %d, minLeaf: %d, error rate: %2.2f\n', ...
		numFeatures, minLeaf, errorRate);
	
end