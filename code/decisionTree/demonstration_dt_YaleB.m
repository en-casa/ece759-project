 %{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

features are generated using principal compenent analysis

%}

function [errorRate, minLeaf] = demonstration_dt_YaleB()

	addpath('utility', 'YaleB', 'YaleB/data');

	fprintf('begin Yale B decision tree demonstration\n');

	% hyper-parameters
	% for feature selection
	N_tr = 2300; % training samples
	N_te = 2414 - N_tr; % test samples
	numFeatures = 38;

	minLeaf = 1; % to prevent overfitting

	% partition data
	% YaleB contains 2414 examples
	[train, test] = loadYaleB(N_tr);

	%% dimensionality reduction / feature generation
	% via prinicpal component analysis (pca) (svd)
	st = cputime;
	
	[train, test] = lda_features(train, test, 1:numFeatures);

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