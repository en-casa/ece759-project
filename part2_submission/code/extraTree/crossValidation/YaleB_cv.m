%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the cross validation of
the extra tree classifier 
across all hyperparameters, 
taking performance and time results along the way

features are generated using lda, as they yield the best performance

%}

clear; close all;
addpath('utility', 'YaleB', 'YaleB/data', 'extraTree');

fprintf('begin Cross Validation on Yale B extra-trees\n');

% hyper-parameters
k = 5; % k-fold cross validation
% use k to partition data
N = 2414;
N_te = floor(N/k);
N_tr = N - N_te;

% for decision tree
minLeaf = 5;
numTreess = [10 50 100]; % ensemble for majority voting

% partition data
[train, test] = loadYaleB(N_tr);

% features are the raw pixels, so we reorder the cell
train = {train{2}, train{1}};
test = {test{2}, test{1}};

% merge sets, they're already randomly shuffled
all = {[train{1}; test{1}],[train{2}, test{2}]};

%% k-fold cross validation across minLeaf

trainTimes = zeros(length(numTreess), k);
errorRates = zeros(length(numTreess), k);

for numTrees = numTreess
	
	% random indices
	%inds = randperm(length(all{2}));
	%all{1} = all{1}(:, inds);
	%all{2} = all{2}(inds);
	
	for fold = 1:k
		
		% choose fold
		inds_bool = false(N,1);
		ind_start = (fold-1)*N_te + 1;
		inds_bool(ind_start:ind_start + N_te - 1) = 1;
		
		test = {all{1}(inds_bool), all{2}(:,inds_bool)};
		train = {all{1}(~inds_bool), all{2}(:,~inds_bool)};
		
		% train
		st = cputime;
		% create an ensemble of random trees
		trees = cell(numTrees, 1);
		for tree = 1:numTrees

			%fprintf('tree: %d\n', tree);
			trees{tree} = trainExtraTree(train, minLeaf);

		end
		thisTrainTime = (cputime - st)/60;
		fprintf('Trained in %4.2f minutes\n', thisTrainTime);
		trainTimes(numTreess == numTrees, fold) = thisTrainTime;
		
		% test
		test = testExtraTrees(test, trees);

		% Classification Error
		thisError = nnz(test{1}(:,1) ~= test{1}(:,2));
		thisErrorRate = (thisError/N_te)*100;
		errorRates(numTreess == numTrees, fold) = thisErrorRate;

		fprintf('numTrees: %d, fold: %d, error rate: %2.2f\n', ...
			numTrees, fold, thisErrorRate);
		
	end
	
end

%% save, print results
filename = 'extraTree/crossValidation/cv_yaleb_et.mat';
save(filename, 'errorRates', 'trainTimes');

fprintf('end Cross Validation on Yale B extra-trees\n');