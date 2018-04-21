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
addpath('utility', 'YaleB', 'YaleB/data', 'decisionTree', 'lda');

fprintf('begin Cross Validation on Yale B decision trees\n');

% hyper-parameters
k = 5; % k-fold cross validation
% use k to partition data
N = 2414;
N_te = floor(N/k);
N_tr = N - N_te;

% for lda
numFeatures = 38;

% for decision tree
minLeaves = 0:30:300;
minLeaves(1) = 1;

% partition data
[train, test] = loadYaleB(N_tr);

% merge sets, they're already randomly shuffled
all = {[train{1}, test{1}],[train{2}; test{2}]};

%% k-fold cross validation across minLeaf

trainTimes = zeros(length(minLeaves), k);
errorRates = zeros(length(minLeaves), k);

for minLeaf = minLeaves
	
	% random indices
	%inds = randperm(length(all{2}));
	%all{1} = all{1}(:, inds);
	%all{2} = all{2}(inds);
	
	for fold = 1:k
		
		% choose fold
		inds_bool = false(N,1);
		ind_start = (fold-1)*N_te + 1;
		inds_bool(ind_start:ind_start + N_te - 1) = 1;
		
		test = {all{1}(:,inds_bool), all{2}(inds_bool)};
		train = {all{1}(:,~inds_bool), all{2}(~inds_bool)};
		
		% dimensionality reduction / feature generation
		[train, test] = lda_features(train, test, 1:numFeatures);
		
		% train
		st = cputime;
		tree = trainDecisionTree({train{2:3}}, minLeaf);
		thisTrainTime = (cputime - st)/60;
		fprintf('Trained in %4.2f minutes\n', thisTrainTime);
		trainTimes(minLeaves == minLeaf, fold) = thisTrainTime;
		
		% test
		test = testDecisionTree(test, tree);

		% Classification Error
		thisError = nnz(test{2}(:,1) ~= test{2}(:,2));
		thisErrorRate = (thisError/N_te)*100;
		errorRates(minLeaves == minLeaf, fold) = thisErrorRate;

		fprintf('minLeaf: %d, fold: %d, error rate: %2.2f\n', ...
			minLeaf, fold, thisErrorRate);
		
	end
	
end

%% save, print results
filename = 'decisionTree/crossValidation/cv_yaleb_dt.mat';
save(filename, 'errorRates', 'trainTimes');

fprintf('end Cross Validation on Yale B decision trees\n');