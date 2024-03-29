%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

%}

function [errorRate] = demonstration_et_YaleB()

	addpath('utility', 'YaleB', 'YaleB/data', ...
		'extraTree');

	fprintf('begin Yale B extra tree demonstration\n');

	% hyper-parameters
	N = 2414;
	N_tr = 2000; % training samples
	N_te = N - N_tr; % test samples

	% for decision tree
	minLeaf = 1; % to prevent overfitting
	numTrees = 50; % ensemble for majority voting
	
	% partition data
	[train, test] = loadYaleB(N_tr);

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

	test = testExtraTrees(test, trees);

	fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

	% Classification Error
	errors = nnz(test{1}(:,1) ~= test{1}(:,2));
	errorRate = (errors/N_te)*100;

	fprintf('\nnumTrees: %d, minLeaf: %d, error rate: %2.2f\n', ...
		numTrees, minLeaf, errorRate);
	
end