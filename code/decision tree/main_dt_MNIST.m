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

% partition data

%{
	1/2 of the dataset should be for training
	the other for testing
	
	MNIST contains 70k examples
%}
N_tr = 35e3; % training samples
N_te = 35e3; % test samples

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
	%train{3}(:,i) = diag(S);
	% just take the first 20 singular values
	train{3}(:,i) = diag(S(1:20,1:20));
end

% average singular value for each digit
for i = 1:9
	inds = find(train{2} == i);
	digitAvg(:,i) = mean(train{3}(:,inds),2);
end

clear i S;
fprintf('Features Generated in %4.2f seconds\n',cputime - st);

%% train ()
st = cputime;

% tree is about 1MB
tree = trainDecisionTree(train);

fprintf('Trained in %4.2f seconds\n',cputime - st);

%% test (~18 secs)
st = cputime;

test = testDecisionTree(test, sz, N_te, tree);

fprintf('Tested in %4.2f seconds\n',cputime - st);

% Classification Error
errors = nnz(test{2}(:,1) ~= test{2}(:,2));
errorRate = (errors/N_te)*100

filename = sprintf('tree%2.0f.mat', errorRate);
save(filename);
