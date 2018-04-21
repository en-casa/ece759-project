 %{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the decision tree classifier

features are generated using principal compenent analysis

%}

clear; close all;
addpath('utility', 'YaleB', 'YaleB/data');

fprintf('begin Yale B decision tree script\n');

%seed = 152039828;
%rng(seed); % for reproducibility

% hyper-parameters
% for feature selection
usePCA = true;
if usePCA
	N_tr = 2414/2; % training samples
	N_te = N_tr; % test samples
	numFeatures = 30;
else
	N_tr = 2300; % training samples
	N_te = 2414 - N_tr; % test samples
	numFeatures = 38;
end

minLeaf = 1; % to prevent overfitting

% partition data
% YaleB contains 2414 examples
[train, test] = loadYaleB(N_tr);

%% dimensionality reduction / feature generation
% via prinicpal component analysis (pca) (svd)
st = cputime;

if usePCA
	[train, U, V] = pca_(train, numFeatures);
else
	[train, test] = lda_features(train, test, 1:numFeatures);
end

fprintf('Features Generated in %4.2f minutes\n', (cputime - st)/60);

%% train
st = cputime;

tree = trainDecisionTree({train{2:3}}, minLeaf);

fprintf('Trained in %4.2f minutes\n',(cputime - st)/60);

%% test
st = cputime;

if usePCA
	test{3} = (U'*test{1}'*V)';
	test{3} = test{3}(1:numFeatures,:);
end

test = testDecisionTree(test, tree);

fprintf('Tested in %4.2f minutes\n', (cputime - st)/60);

% Classification Error
errors = nnz(test{2}(:,1) ~= test{2}(:,2));
errorRate = (errors/N_te)*100;

filename = sprintf('yaleB_tree%2.0f_%d.mat', errorRate, minLeaf);
%save(filename, 'tree');
%filename = sprintf('UV%2.0f.mat', errorRate);
%save(filename, 'U', 'V', '-v7.3');

fprintf('\nnumFeatures: %d, minLeaf: %d, error rate: %2.2f\n', ...
	numFeatures, minLeaf, errorRate);