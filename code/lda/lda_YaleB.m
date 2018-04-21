%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the linear discriminant analysis classifier

%}

clear; close all;
addpath('../utility');
addpath('../YaleB', '../YaleB/data', '../YaleB/50Train', '../decision tree');
%% before 
% [faces, labels] = loadYaleB();
% labels = labels - ones(size(labels,1));
% seed = 152039828;
% rng(seed); % for reproducibility
% load('2.mat'); % we load the indices to train and test sets
% train = {faces(trainIdx, :)', labels(trainIdx)};
% test = {faces(testIdx, :)', labels(testIdx)};
% k = 38;
% N_tr = size(trainIdx,1);
%%
[train, test] = loadYaleB();
train{1,2} = train{1,2} - 1;
test{1,2} = test{1,2} - 1;
k = 38;
N_tr = size(test{1,2},1);
%% use PCA to see the results
% numFeatures = 50;
% [train, U, V] = pca_(train, numFeatures);
% train{1,1} = train{1,3};
% [test, U, V] = pca_(test, numFeatures);
% test{1,1} = test{1,3};
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
a = [];
% for i = 1:(k)
%     a = [a D(i,i)];
% end
% from here we can see that we only have 9 highest values as we expected
%% this way is done using pca
% We transform the training and testing data to a subspace
% transf_train = train{1,1}'* U(:,1:numFeatures);
% transf_test = test{1,1}'*U(:, 1:numFeatures);
% % calculting the multivariate parameters
% mu_each_class = zeros(numFeatures, k);
% cov_each_class = {};
% sum_cov = zeros(numFeatures,numFeatures);
% for i =0:k-1
%     ind = find(train{1,2} == i);
%     X = train{1,1}';
%     X = X(ind, :);
%     mu_each_class(:, i+1) = mean(X, 1)';
%     cov_each_class{1, i+1} = cov(X);
%     sum_cov = sum_cov + cov(X);
% end
% average_cov = sum_cov/k;
% 
% cov_equal_each_class = {average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov};
% % to see how our model works
% [acc_test_comp acc_train_comp] = classify_comparison(k,5,mu_each_class, cov_equal_each_class, test{1,1}, test{1,2}, train{1,1}, train{1,2}); % 0.88 and 0.89 resp using just knn

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
%[acc_test_comp acc_train_comp] = classify_comparison(k,5,mu_each_class, cov_equal_each_class, transf_test', test{1,2}, transf_train', train{1,2}); % 0.88 and 0.89 resp using just knn
%%
t0 = cputime;
N_tr = 2414; % training samples
N_te = 0; % test samples
transf_matrix = U(:,1:(k-1));
[train, test] = loadYaleB(N_tr);
train{1,2} = train{1,2} - 1;
test{1,2} = test{1,2} - 1;
% transform the data
transf_train = train{1,1}' * transf_matrix;
N_cross_val = 5;
CVO = cvpartition(train{1,2},'k',N_cross_val);
err = zeros(CVO.NumTestSets,1);
for j = 1:N_cross_val
    ind = CVO.training(j);
    train_cv = transf_train(ind,:); train_cv_label = train{1,2}(ind,:);
    test_cv = transf_train(~ind,:); test_cv_label = train{1,2}(~ind);
    mu_each_class = zeros(k-1, k);
    sum_cov = zeros(k-1,k-1);
    for i = 0:(k-1)
        ind1 = find(train_cv_label == i);
        X = train_cv(ind1, :);
        mu_each_class(:, i+1) = mean(X, 1)';
        sum_cov = sum_cov + cov(X);
    end
    E_cov = sum_cov/k;
    [acc_cross_valid_test(j) acc_cross_valid_train(j)] = classify_comparison_same_cov(k,5,mu_each_class, ...
        cov_each_class, E_cov, test_cv', test_cv_label, train_cv', train_cv_label);
    err(j) = 1 - acc_cross_valid_train(j);
end
mean_acc_test = mean(acc_cross_valid_test);
sd_test = sqrt(var(acc_cross_valid_test));
mean_acc_train = mean(acc_cross_valid_train);
sd_train = sqrt(var(acc_cross_valid_train));
fprintf('Tested in %4.2f minutes\n', (cputime - t0)/60);
f = instantiateFig(1);
plot([1:5],err*100, 'r')
prettyPictureFig(f);
xlabel('5-fold cross validation');
ylabel('Error rate');
title('Error rate for Yale B');
print('../../images/cr-err-yale', '-dpng');

%% this is different cross-validation technique
t0 = cputime;
N_cross_val = 5;
CVO = cvpartition(train{1,2},'k',N_cross_val);
err = zeros(CVO.NumTestSets,1);
for j = 1:N_cross_val
    ind = CVO.training(j);
    train_cv = transf_train(ind,:); train_cv_label = train{1,2}(ind,:);
    test_cv = transf_train(~ind,:); test_cv_label = train{1,2}(~ind);
    mu_each_class = zeros(k-1, k);
    sum_cov = zeros(k-1,k-1);
    for i = 0:(k-1)
        ind1 = find(train_cv_label == i);
        X = train_cv(ind1, :);
        mu_each_class(:, i+1) = mean(X, 1)';
        sum_cov = sum_cov + cov(X);
    end
    E_cov = sum_cov/k;
    [acc_cross_valid_test(j) acc_cross_valid_train(j)] = classify_comparison_same_cov(k,5,mu_each_class, E_cov, test_cv', test_cv_label, train_cv', train_cv_label);
end
mean_acc_test = mean(acc_cross_valid_test);
sd_test = sqrt(var(acc_cross_valid_test));
mean_acc_train = mean(acc_cross_valid_train);
sd_train = sqrt(var(acc_cross_valid_train));
fprintf('Tested in %4.2f minutes\n', (cputime - t0)/60);
%% this part is for testing with kNN
[acc_test_5 acc_train_5] = classifyNN(k,5,mu_each_class, cov_equal_each_class, transf_test', test{1,2}, transf_train', train{1,2}); % 0.88 and 0.89 resp using just knn
[acc_test_5_p acc_train_5_p] = classifyNN_pure(5,transf_test', transf_train', test{1,2}, train{1,2});

% let's now change the n's to see how it works
parfor n = 1:13
% we apply Nearest neigbors in order to find which class it belongs
    accuracy(n) = classifyNN(n,transf_test', transf_train', test{1,2}, train{1,2});
end
% we plot the results to see the best number of nearest neighbors

f = instantiateFig(1);
plot([1:13],accuracy*100, 'r.')
prettyPictureFig(f);
xlabel('Nearest neighbor number');
ylabel('Accuracy of test model');

print('../../images/YaleBNNafterLDA', '-dpng');
% instead we can use Euclidean distance metr
ic to evaluate the classes by
%% calculating the distances from each class centroid
% for our surprise, it performs really well
centroid = zeros(k, k-1);
for i = 0:k-1
    ind = (train{1,2} == i);
    N_i = sum(ind);
    centroid(i+1, :) = mean(transf_train(ind,:), 1);    
end

accuracy1 = classify_from_centroid(transf_test', test{1,2},centroid);
accuracy1_train = classify_from_centroid(transf_train', train{1,2},centroid);