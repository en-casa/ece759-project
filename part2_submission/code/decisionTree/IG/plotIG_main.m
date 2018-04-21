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
fprintf('Features Generated in %4.2f seconds\n',cputime - st);

%% train ()
st = cputime;

% tree is about 1MB
[info_gain, ind_best] = plotIG(train);

filename = 'IG.mat';
save(filename);

fprintf('Trained in %4.2f seconds\n',cputime - st);

f = instantiateFig(1);
plot(info_gain);
xlabel('index');
ylabel('information gain');
title(sprintf('Information Gain\nBest Index: %d', ind_best));
hold on;
plot(ind_best, info_gain_best, 'ro');
hold off;
prettyPictureFig(f);
saveImage('IG');
