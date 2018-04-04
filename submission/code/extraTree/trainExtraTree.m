%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to train an extra tree

parameters:
	- set
		struct with class labels {1} and data vectors {2}
	- minLeaf 
		parameter to introduce a stopping condition. it reduces the
		depth of the tree and is meant to prevent overfitting and reduce
		computational overhead

outputs:
	- tree
		struct of structs that represents a decision tree.
		each element takes one of two forms:
			- node
				{'node', attribute, thresholdBest, subtree1, subtree2};
				where attribute is the feature that the node splits on,
				threshold is the continuous value at which the decision is
				made,	subtree1 is the branch that is taken if the feature
				is LESS than threshold and subtree2 is the branch that is taken
				if the feature is GREATER than threshold.
			- leaf
				{'leaf', class};
				where class is the chosen class for this leaf. if a data vector
				finds itself at this leaf, it is assigned to class in the
				decision tree.				

%}

function tree = trainExtraTree(set, minLeaf)

	% Check for base cases
	% 1. no more features to split on
	% this condition is rare
	if (isempty(set{2}))
		% return the single node tree root with label
		% = mode label of set
		%fprintf('    no more features to split on\n');
		tree = {'leaf', mode(set{1})};
		return
	end
	
	% 2. set is smaller than minLeaf
	% this should reduce overfitting
	if length(set{1}) < minLeaf
		%fprintf('    set is smaller than minLeaf\n');
		tree = {'leaf', mode(set{1})};
		return
	end
	
	% 3. all samples belong to the same class
	class = set{1}(1);
	diffClasses = false;
	% this loop breaks as soon as the first class divergence is detected
	for i = 2:length(set{1})
		if set{1}(i) ~= class
			diffClasses = true;
			break;
		end
	end
	
	if ~diffClasses
		% create leaf node for decision tree to choose that class
 		%fprintf('    all samples in same class\n');
 		tree = {'leaf', set{1}(1)};
 		return
	end
	
	% choose a random splitting threshold and index
	% select a pixel location at random
	ind = randi(size(set{2},1),1);
	% select a threshold at random according to N(mu, sigma)
	% where mu and sigma are the mean and variance of ind
	% across all samples in the set
	mu = mean(set{2}(ind,:));
	sigma = var(set{2}(ind,:));
	threshold = normrnd(mu, sigma);
	if threshold < 0
		threshold = 0;
	elseif threshold > 1
		threshold = 1;
	end
	
	% recur on the sublists obtained by splitting on ind, threshold
	% and add those nodes as children of this node
	subsets = getSubsets_et(set, ind, threshold);
	subtree1 = trainExtraTree(subsets{1}, minLeaf);
	subtree2 = trainExtraTree(subsets{2}, minLeaf);
	
	% create a decision node that splits on attributeBest
	tree = {'node', ind, threshold, subtree1, subtree2};
	
	return
	
end