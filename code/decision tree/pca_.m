%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get subsets of a set

%}

function set = pca_(set, numFeatures)

	X = set{1}';

	% subtract column-wise empirical mean from each column 
	% to render each column zero mean
	colAvg = zeros(size(X,2),1);

	for i = 1:size(X,2)
		colAvg(i) = mean(X(:,i));
		X(:,i) = X(:,i) - colAvg(i);
		colAvg(i) = mean(X(:,i));
	end

	% compute SVD of entire matrix
	[U, S, ~] = svd(X);

	T = U*S; % pca score matrix
	set{3} = T(:,1:numFeatures)';

end