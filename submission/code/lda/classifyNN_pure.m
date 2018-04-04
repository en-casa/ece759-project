function [acc_test acc_train] = classifyNN_pure(num_neighbors,test_data, train_data, test_label, train_label)
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
% performance on test data
parfor test_n = 1:test_N

    test_mat = repmat(test_data(:, test_n), [1,train_size]);
    distance = sum(abs(test_mat - train_data).^2);
    [~,distances_index] = sort(distance);
    neighbors=distances_index(1:num_neighbors);
    a = mode(train_label(neighbors));
    if a == test_label(test_n)
        counter(test_n) = counter(test_n) + 1;
    end
end

acc_test = double(sum(counter)) / test_N;
% performance on training data
counter = zeros(train_N, 1);
parfor test_n = 1:train_N

    test_mat = repmat(train_data(:, test_n), [1,train_size]);
    distance = sum(abs(test_mat - train_data).^2);
    [~,distances_index] = sort(distance);
    neighbors=distances_index(2:num_neighbors+1); % excluding itself
    a = mode(train_label(neighbors));
    if a == train_label(test_n)
        counter(test_n) = counter(test_n) + 1;
    end
end

acc_train = double(sum(counter)) / train_N;