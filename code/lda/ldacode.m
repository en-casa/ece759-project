%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this script orchestrates the training and testing of
the linear discriminant analysis classifier

%}

clear; close all;
addpath('../utility', '../MNIST', '../MNIST/data', '../MNIST/loadMNIST', ...
	'../lda');

seed = 152039828;
rng(seed); % for reproducibility

% define parameters
N_tr = 60e3; % training samples
N_te = 10e3; % test samples
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
% for i = 0:k-1
%     ind = (train{1,2} == i);
%     N_i = sum(ind);
%     x = train{1,1}(:, ind);
%     mu_i = mean(x, 2);
%     Si = Si + (1/N_tr)*(x - (repmat(mu_i,1, N_i)))*(x - (repmat(mu_i,1, N_i)))';
%     Sb = Sb + (N_i/N_tr)*(mu_i - mu)*(mu_i - mu)'; % (1/k)
% end
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
cov_equal_each_class = {average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov average_cov};
% parfor n = 1:13
% % we apply Nearest neigbors in order to find which class it belongs
%     accuracy(n) = classifyNN(n,transf_test', transf_train', test{1,2}, train{1,2});
% end
[acc_test_comp acc_train_comp] = classify_comparison(k,5,mu_each_class, cov_equal_each_class, transf_test', test{1,2}, transf_train', train{1,2}); % 0.88 and 0.89 resp using just knn
[acc_test_5 acc_train_5] = classifyNN(k,5,mu_each_class, cov_each_class, transf_test', test{1,2}, transf_train', train{1,2}); % 0.88 and 0.89 resp using just knn
[acc_test_5 acc_train_5] = classifyNN(k,5,mu_each_class, cov_equal_each_class, transf_test', test{1,2}, transf_train', train{1,2}); % 0.87 and 0.88 resp using just knn
[acc_test_5_p acc_train_5_p] = classifyNN_pure(5,transf_test', transf_train', test{1,2}, train{1,2});

% cov each class separately works better in each case
% we plot the results to see the best number of nearest neighbors
% [acc_test_5 acc_train_5] = classifyNN(5,transf_test', transf_train', test{1,2}, train{1,2}); % 0.86 and 0.85 resp using just knn
% this is my first attempt

f = instantiateFig(1);
plot([1:13],accuracy*100, 'r.')
prettyPictureFig(f);
xlabel('Nearest neighbor number');
ylabel('Accuracy of test model');

print('../../images/NN after LDA', '-dpng');
% instead we can use Euclidean distance metric to evaluate the classes by
% calculating the distances from each class centroid
centroid = zeros(k, k-1);
for i = 0:k-1
    ind = (train{1,2} == i);
    N_i = sum(ind);
    centroid(i+1, :) = mean(transf_train(ind,:), 1);    
end

accuracy1_train = classify_from_centroid(transf_train', train{1,2}, centroid);
accuracy1 = classify_from_centroid(transf_test', test{1,2},centroid);

%%
mdl = fitcdiscr(transf_train, train{1,2},'DiscrimType','linear'); % this one works since n>m in tansf_train
mdl = fitcdiscr(train{1,1}', train{1,2},'DiscrimType','linear'); % error saying Predictor x1 has zero within-class variance.
pred = predict(mdl, transf_test);
count = 0;
for i = 1:size(transf_test,1)
   if pred(i) ==  test{1,2}(i);
       count = count + 1;
   end
end
acc = count/size(transf_test,1) % 0.8639
% we get a similar result as our method






