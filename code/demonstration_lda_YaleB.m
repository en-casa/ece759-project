clear; close all;
addpath('./utility');
addpath('./YaleB', './YaleB/data', './YaleB/50Train', './decisionTree');
%%
[train, test] = loadYaleB();
train{1,2} = train{1,2} - 1;
test{1,2} = test{1,2} - 1;
k = 38;
N_tr = size(test{1,2},1);
%% Construct scatter matrices and calculate within-class and between class
% covariance
mu = mean(train{1,1}, 2);
num_variables = size(train{1,1},1);
% Let's standardize the data;
%variance = var(train{1,1}, 0,2);
%train{1,1} = (train{1,1}-repmat(mu,1, 1900))./variance;
% 
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
%% We transform the training and testing data to a subspace
transf_train = train{1,1}'* U(:,1:(k-1));
transf_test = test{1,1}'*U(:, 1:(k-1));
% calculting the multivariate parameters
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
average_cov = sum_cov/k;
cov_equal_each_class = {average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov};
% to see how our model works
[acc_test_comp acc_train_comp] = classify_comparison_same_cov(k,5,mu_each_class, cov_each_class,average_cov, transf_test', test{1,2}, transf_train', train{1,2}); % 0.88 and 0.89 resp using just knn
