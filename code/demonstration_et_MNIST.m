%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

%}

function [errorRate] = demonstration_et_MNIST()

	addpath('utility', 'MNIST', 'MNIST/data', 'MNIST/loadMNIST', ...
		'extraTree');

	fprintf('begin MNIST extra tree demonstration\n');

	% hyper-parameters
	N = 70e3;
	N_tr = 35e3; % training samples
	N_te = 35e3; % test samples

	% for decision tree
	minLeaf = 1; % to prevent overfitting
	numTrees = 100; % ensemble for majority voting
	
	% partition data
	[train, test] = loadMNIST(N_tr);

	% features are the raw pixels, so we reorder the cell
	train = {train{2}, train{1}};
	test = {test{2}, test{1}};
	
	%% train
	st = cputime;

	% create an ensemble of random trees
	trees = cell(numTrees, 1);
	for tree = 1:numTrees

		fprintf('tree: %d\n', tree);
		trees{tree} = trainExtraTree(train, minLeaf);

	end

	fprintf('Trained in %4.2f minutes\n',(cputime - st)/60);

	%% test
	st = cputime;

	test = testExtraTree(test, trees);

	fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

	% Classification Error
	errors = nnz(test{1}(:,1) ~= test{1}(:,2));
	errorRate = (errors/N_te)*100;

	fprintf('\nnumTrees: %d, minLeaf: %d, error rate: %2.2f\n', ...
		numTrees, minLeaf, errorRate);
	
end