function [acc_test acc_train] = classify_comparison_same_cov(total_classes,num_neighbors, mean_vector, covariance, test_data, test_label, train_data,  train_label)
%
% Description:  
% Classify test data using Nearest Neighbor method withEuclidean distance
% criteria. 
% 
% Usage:
% [accuracy] = classifyNN(test_data, train_data, test_label, train_label)
%
% Parameters:
% test_data = test images projected in reduced dimension  dxtn
% train_data = train images projected in reduced dimension dxN
% test_label = test labels for each data tn x 1
% train_label = train labels for each train data Nx1
%
% Returns:
% accuracy: a scalar number of the classification accuracy

train_size = size(train_data, 2);
train_N = size(train_data,2);
test_N = size(test_data, 2);
counter = zeros(test_N, 1);
C_inv = inv(covariance);
% performance on test data
parfor test_n = 1:test_N
    test_vector = test_data(:, test_n);
    f = zeros(total_classes,1);
    for i = 1:total_classes
        f(i,1) = mean_vector(:,i)'*C_inv*test_vector - (1/2)*mean_vector(:,i)'*C_inv*mean_vector(:,i) + log(1/total_classes);
    end
    a = find(f == max(f)) - 1;
    if a == test_label(test_n)
        counter(test_n) = counter(test_n) + 1;
    end
end

acc_test = double(sum(counter)) / test_N;
% performance on training data
counter = zeros(train_N, 1);
parfor test_n = 1:train_N
    test_vector = train_data(:, test_n);
    f = zeros(total_classes,1);
    for i = 1:total_classes
        f(i,1) = mean_vector(:,i)'*C_inv*test_vector - (1/2)*mean_vector(:,i)'*C_inv*mean_vector(:,i)+ log(1/total_classes);
    end
    a = find(f == max(f)) - 1;
    if a == train_label(test_n)
        counter(test_n) = counter(test_n) + 1;
    end
end

acc_train = double(sum(counter)) / train_N;