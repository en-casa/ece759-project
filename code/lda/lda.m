%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the linear discriminant analysis classifier

%}

clear; close all;
addpath('utility', 'MNIST', 'MNIST/data', 'MNIST/loadMNIST', ...
	'lda');

seed = 152039828;
rng(seed); % for reproducibility

% define parameters
N_tr = 35e3; % training samples
N_te = 35e3; % test samples
k = 10; % number of classes
% partition data

%{
	1/2 of the dataset should be for training
	the other for testing
	
	MNIST contains 70k examples
%}

[train, test] = loadMNIST(N_tr);

% Construct scatter matrices and calculate within-class and between class
% covariance
mu = mean(train{1,1}, 2);
num_variables = size(train{1,1},1);
Si = zeros(num_variables); Sb = zeros(num_variables);
for i = 0:k-1
    ind = (train{1,2} == i);
    N_i = sum(ind);
    x = train{1,1}(:, ind);
    mu_i = mean(x, 2);
    Si = Si + (1/N_tr)*(x - (repmat(mu_i,1, N_i)))*(x - (repmat(mu_i,1, N_i)))';
    Sb = Sb + (N_i/N_tr)*(mu_i - mu)*(mu_i - mu)'; % (1/k)
end

% We apply singular value decomposition in order to find eigenvalues and
% eigenvectors
[U D V] = svd(pinv(Si)*Sb);
a = [];
for i = 1:(k)
    a = [a D(i,i)]
end
% from here we can see that we only have 9 highest values as we expected

% We transform the training and testing data to a subspace
transf_train = train{1,1}'* U(:,1:(k-1));
transf_test = test{1,1}'*U(:, 1:(k-1));

parfor n = 1:13
% we apply Nearest neigbors in order to find which class it belongs
    accuracy(n) = classifyNN(n,transf_test', transf_train', test{1,2}, train{1,2});
end
% we plot the results to see the best number of nearest neighbors
plot([1:13],accuracy*100, 'r.')

% instead we can use Euclidean distance metric to evaluate the classes







