%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

%}

clear; close all;
addpath('utility', 'MNIST', 'MNIST/data', 'MNIST/loadMNIST', ...
	'decision tree');

seed = 152039828;
rng(seed); % for reproducibility

% define parameters
N_tr = 35e3; % training samples
N_te = 35e3; % test samples

% partition data

%{
	1/2 of the dataset should be for training
	the other for testing
	
	MNIST contains 70k examples
%}

[train, test] = loadMNIST(N_tr);

%% dimensionality reduction / feature generation
% via prinicpal component analysis (pca) (svd)
st = cputime;

sz = 28;

% takes a few secs
for i = 1:length(train{2})
	%[train{3}{1,i}, train{3}{2,i}, train{3}{3,i}] = svd(reshape(train{1}(:,i),[sz, sz]));
	%train{3}{2,i} = diag(train{3}{2,i});
	[~, S, ~] = svd(reshape(train{1}(:,i),[sz, sz]));
	train{3}(:,i) = diag(S);
end

% average singular value for each digit
for i = 1:9
	inds = find(train{2} == i);
	digitAvg(:,i) = mean(train{3}(:,inds),2);
end

clear i S;
fprintf('\nFeatures Generated in %4.2f seconds\n\n',cputime - st);

%% train
st = cputime;

% tree is about 1MB
tree = trainDecisionTree(train);

fprintf('\nTrained in %4.2f seconds\n\n',cputime - st);
%% test
st = cputime;
% for each test sample
for i = 1:N_te
	
	% generate features via svd
	[~, S, ~] = svd(reshape(test{1}(:,i),[sz, sz]));
	test{3}(:,i) = diag(S);
	
	% pass through the tree
	tree_walked = tree;
	classified = false;
	while (~classified)
		
		% if we're on a node
		if strcmp('node', string(tree_walked(1)))
			% compare attribute value to threshold
			attribute = cell2mat(tree_walked(2));
			threshold = cell2mat(tree_walked(3));
			if (test{3}(attribute) < threshold)
				tree_walked = tree_walked{4};
			else
				tree_walked = tree_walked{5};
			end
			
		% if we're on a leaf
		else
			test{2}(i,2) = tree_walked{2};
			classified = true;
			
		end		
		
	end
	
end

fprintf('\nTested in %4.2f seconds\n\n',cputime - st);

%% Classification Error

errors = nnz(test{2}(:,1) ~= test{2}(:,2));

