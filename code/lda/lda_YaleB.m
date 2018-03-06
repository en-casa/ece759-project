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
addpath('../YaleB', '../YaleB/data', '../YaleB/50Train');

[faces, labels] = loadYaleB();
labels = labels - ones(size(labels,1));
seed = 152039828;
rng(seed); % for reproducibility
load('1.mat'); % we load the indices to train and test sets
train = {faces(trainIdx, :)', labels(trainIdx)};
test = {faces(testIdx, :)', labels(testIdx)};
k = 38;
N_tr = size(trainIdx,1);
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
[U D V] = svd(inv(Si)*Sb);
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

f = instantiateFig(1);
plot([1:13],accuracy*100, 'r.')
prettyPictureFig(f);
xlabel('Nearest neighbor number');
ylabel('Accuracy of test model');

print('../images/YaleB_NN_after_LDA', '-dpng');
% instead we can use Euclidean distance metric to evaluate the classes by
% calculating the distances from each class centroid
centroid = zeros(k, k-1);
for i = 0:k-1
    ind = (train{1,2} == i);
    N_i = sum(ind);
    centroid(i+1, :) = mean(transf_train(ind,:), 1);    
end

accuracy1 = classify_from_centroid(transf_test', test{1,2},centroid);