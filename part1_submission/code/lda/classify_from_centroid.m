function [accuracy1] = classify_from_centroid(test_data, test_label, centroids)
% This function will take inputs of centroid coordinates and measure the
% euclidean distances of test data from different class centroids and
% classify to the minimum distance one
test_N = size(test_data, 2);
counter = zeros(test_N, 1);
centroid_size = size(centroids, 1);
centroids_T = centroids';
for test_n = 1:test_N
    test_mat = repmat(test_data(:, test_n), [1,centroid_size]);
    distance = sum(abs(test_mat - centroids_T).^2);
    [M,I] = min(distance);
    if I-1 == test_label(test_n)
        counter(test_n) = counter(test_n) + 1;
    end
    
end

accuracy1 = double(sum(counter)) / test_N;








