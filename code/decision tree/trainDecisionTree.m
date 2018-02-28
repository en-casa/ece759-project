%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to train a decision tree

%}

function tree = trainDecisionTree(set)

	% Check for base cases
	% 1. all samples belong to the same class
	thisSetEntropy = getEntropy(set);
	if (~thisSetEntropy)
		% create leaf node for the decision tree saying to choose that class
		
	end
	
	% 2. no more features to split on
	if (isempty(set{3}))
		% return the single node tree root with label
		% = mode label of set
		
	end
	
	% let attribute_best be the attribute with the highest normalized information
	% gain (after splitting)
	attribute_best = 0;
	threshold_best = inf;
	info_gain_best = 0;
	
	% for each attribute i
	for i = 1:size(set{3},1)
		
		% choose a splitting threshold
		% naïve: split on halfway between min and max
		threshold = (max(set{3}(i,:)) - min(set{3}(i,:))) / 2;
		
		% find the normalized information gain ratio from splitting on i.
		info_gain = thisSetEntropy - getSplitEntropy(set, i, threshold);
		
		if (info_gain > info_gain_best)
			attribute_best = i;
			threshold_best = threshold;
			info_gain_best = info_gain;
		end
		
	end
	
	% create a decision node that splits on attribute_best.
	
	
	% recur on the sublists obtained by splitting on attribute_best, 
	% and add those nodes as children of node.
	subsets = getSubsets(set, attribute_best, threshold_best);
	subtree1 = trainDecisionTree(subsets{1});
	subtree2 = trainDecisionTree(subsets{2});
	
end
