%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to train a decision tree

%}

function tree = naivetrainDecisionTree(set)

	% Check for base cases
	% 1. all samples belong to the same class
	thisSetEntropy = getEntropy(set);
	if (~thisSetEntropy && ~isempty(set{2}))
		% create leaf node for decision tree to choose that class
		fprintf('    all samples in same class\n');
		tree = {'leaf', set{2}(1)};
		return
	end
	
	% 2. no more features to split on
	% this condition seems rare.
	if (isempty(set{3}))
		% return the single node tree root with label
		% = mode label of set
		fprintf('    no more features to split on\n');
		tree = {'leaf', mode(set{2})};
		return
	end
	
	% let attribute_best be the attribute with the highest 
	% normalized information gain (after splitting)
	attribute_best = 0;
	threshold_best = 0;
	info_gain_best = 0;
	
	% for each attribute
	for att = 1:size(set{3},1)
		
		% sort by attribute value
		[~,I] = sort(set{3}(att,:));
		set{3} = set{3}(:,I);
		set{2} = set{2}(I);
		set{1} = set{1}(:,I);
		
		% naïve: split on halfway between min and max 
		% (~70% error rate with MNIST)
		middleInd = round(size(set{3},2)/2);
		threshold = (set{3}(att,middleInd) + set{3}(att,middleInd + 1)) / 2;
		
		% find the normalized information gain ratio from splitting on i.
		info_gain = thisSetEntropy - getSplitEntropy(set, att, middleInd);

		if (info_gain > info_gain_best)
			attribute_best = att;
			threshold_best = threshold;
			info_gain_best = info_gain;
		end
		
	end
	
	if (~info_gain_best || ~attribute_best)
		fprintf('    couldn''t find attribute to split on\n');
		% return the single node tree root with label
		% = mode label of set
		tree = {'leaf', mode(set{2})};
		return
	end
	
	% recur on the sublists obtained by splitting on attribute_best, 
	% and add those nodes as children of node.
	% need to re-sort according to attribute_best
	[~,I] = sort(set{3}(attribute_best,:));
	set{3} = set{3}(:,I);
	set{2} = set{2}(I);
	set{1} = set{1}(:,I);
	subsets = getSubsets(set, attribute_best, middleInd);
	subtree1 = naivetrainDecisionTree(subsets{1});
	subtree2 = naivetrainDecisionTree(subsets{2});
	
	% create a decision node that splits on attribute_best.
	tree = {'node', attribute_best, threshold_best, subtree1, subtree2};
	
	return
	
end
