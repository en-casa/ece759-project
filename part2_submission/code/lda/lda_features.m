%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this funciton generates features using lda

%}

function [train, test] = lda_features(train, test, classes)

	N_tr = length(train{2});
	
	% construct scatter matrices and calculate within-class and between class
	% covariance
	mu = mean(train{1}, 2);
	num_variables = size(train{1},1);

	Si = zeros(num_variables); Sb = zeros(num_variables);
	S_cov = zeros(num_variables);
	for i = classes
		 ind = (train{2} == i);
		 N_i = sum(ind);
		 x = train{1}(:, ind);
		 mu_i = mean(x, 2);
		 S_cov = S_cov + cov(x');
		 Si = Si + (1/N_tr)*(x - (repmat(mu_i,1, N_i)))*(x - (repmat(mu_i,1, N_i)))';
		 Sb = Sb + (N_i/N_tr)*(mu_i - mu)*(mu_i - mu)'; % (1/k)
	end

	% apply singular value decomposition in order to find eigenvalues and
	% eigenvectors
	[U, ~, ~] = svd(pinv(Si)*Sb); % lets try S_cov/k instead of Si; but it is the same result

	% We transform the training and testing data to a subspace
	transf_matrix = U(:,1:numel(classes));
	train{3} = (train{1}'*transf_matrix)';
	test{3} = (test{1}'*transf_matrix)';

end