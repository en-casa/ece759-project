clear; close all;
addpath('./utility', './MNIST', './MNIST/data', './MNIST/loadMNIST', ...
	'./lda');

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
%% use PCA to see the results
% numFeatures = 20;
% [train, U, V] = pca_(train, numFeatures);

%% Construct scatter matrices and calculate within-class and between class
% covariance
mu = mean(train{1,1}, 2);
num_variables = size(train{1,1},1);
% Let's standardize the data;
%variance = var(train{1,1}, 0,2);
%train{1,1} = (train{1,1}-repmat(mu,1, 60000))./variance;

%
Si = zeros(num_variables); Sb = zeros(num_variables);
S_cov = zeros(num_variables);
for i = 0:k-1
    ind = (train{1,2} == i);
    N_i = sum(ind);
    x = train{1,1}(:, ind);
    mu_i = mean(x, 2);
    S_cov = S_cov + cov(x');
    Si = Si + (1/N_tr)*(x - (repmat(mu_i,1, N_i)))*(x - (repmat(mu_i,1, N_i)))';
    Sb = Sb + (N_i/N_tr)*(mu_i - mu)*(mu_i - mu)'; % (1/k)
end

% We apply singular value decomposition in order to find eigenvalues and
% eigenvectors
[U D V] = svd(pinv(Si)*Sb); % lets try S_cov/k instead of Si; but it is the same result
a = [];
for i = 1:(k)
    a = [a D(i,i)];
end

% from here we can see that we only have 9 highest values as we expected

% We transform the training and testing data to a subspace
transf_matrix = U(:,1:(k-1));
transf_train = train{1,1}'* transf_matrix;
transf_test = test{1,1}'*transf_matrix;
% We find mean vector and covariance matrix for each class
mu_each_class = zeros(k-1, k);
cov_each_class = {};
sum_cov = zeros(k-1,k-1);
for i =0:k-1
    ind = find(train{1,2} == i);
    X = transf_train(ind, :);
    mu_each_class(:, i+1) = mean(X, 1)';
    cov_each_class{1, i+1} = cov(X);
    sum_cov = sum_cov + cov(X);
end
% since we assume equal covariance in all classes, we take the average of
% covariance matrices
average_cov = sum_cov/k;
cov_equal_each_class = {average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov};
% this part is just a test on how nearest neigbors work
% parfor n = 1:13
% % we apply Nearest neigbors in order to find which class it belongs
%     accuracy(n) = classifyNN(n,transf_test', transf_train', test{1,2}, train{1,2});
% end
[acc_test acc_train] = classify_comparison_same_cov(k,5,mu_each_class, cov_each_class, average_cov, ...
transf_test', test{1,2}, transf_train', train{1,2}); % 0.88 and 0.89 resp using just knn
%[acc_test_comp acc_train_comp] = classify_comparison(k,5,mu_each_class, cov_equal_each_class, transf_test', test{1,2}, transf_train', train{1,2}); % 0.88 and 0.89 resp using just knn
