%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to train a decision tree

%}

function tree = trainDecisionTree(set, minLeaf)

	% Check for base cases
	% 1. no more features to split on
	% this condition seems rare
	if (isempty(set{2}))
		% return the single node tree root with label
		% = mode label of set
		fprintf('    no more features to split on\n');
		tree = {'leaf', mode(set{1})};
		return
	end
	
	% 2. set is smaller than minLeaf
	% this should reduce overfitting
	if length(set{1}) < minLeaf
		fprintf('    set is smaller than minLeaf\n');
		tree = {'leaf', mode(set{1})};
		return
	end
	
	% 3. all samples belong to the same class
	thisSetEntropy = getEntropy(set);
	if (~thisSetEntropy)
		% create leaf node for decision tree to choose that class
		fprintf('    all samples in same class\n');
		tree = {'leaf', set{1}(1)};
		return
	end
	
	% let attributeBest be the attribute with the highest 
	% normalized information gain (after splitting)
	attributeBest = 0;
	thresholdBest = 0;
	infoGainBest = 0;
	indBest = 0;
	
	% for each attribute
	%if size(set{2},1) > 5
	%	atts = 5;
	%else
	%	atts = size(set{2},1);
	%end
	for att = 1:size(set{2},1)
		
		% sort by attribute value
		[~,I] = sort(set{2}(att,:));
		set{2} = set{2}(:,I);
		set{1} = set{1}(I);
		%set{1} = set{1}(:,I);
		
		% optimize IG over thresholds via line search
		% see page 2: https://www.jair.org/media/279/live-279-1538-jair.pdf
		
		% note: set size is always at least 2 - see 2nd if clause
		midInd = round(size(set{2},2)/2);
		% choose indices halfway between midpoint and endpoints
		leftInd = ceil(midInd/2);
		rightInd = floor((length(set{1})-midInd)/2) + midInd;

		midInfoGain = 0;
		closeToMax = false;
		foundMax = false;
		span = 10; % for fine steps
		iter = 0; % for iteration limit
		while (~foundMax)
			
			% don't exceed array dimensions
			if rightInd > length(set{1})
				rightInd = length(set{1});
			end
			if leftInd < 1
				leftInd = 1;
			end
			
			% find the normalized information gain ratio from splitting on att
			midInfoGain = thisSetEntropy - getSplitEntropy(set, att, midInd);
			leftInfoGain = thisSetEntropy - getSplitEntropy(set, att, leftInd);
			rightInfoGain = thisSetEntropy - getSplitEntropy(set, att, rightInd);

			if length(set{1}) > 5000
				fprintf('a: %d, l: %d, %1.5f, m: %d, %1.5f, r: %d, %1.5f\n', ...
					att, leftInd, leftInfoGain, midInd, midInfoGain, rightInd, rightInfoGain);
			end
			
			if (leftInfoGain > midInfoGain)
				midInfoGain = leftInfoGain;
				oldMidInd = midInd;
				midInd = leftInd;
				rightInd = floor((oldMidInd - leftInd)/2) + leftInd;
				leftInd = leftInd - ceil((oldMidInd - leftInd)/2);
				
				% to diversify search if we were close
				if closeToMax
					rightInd = rightInd + 20;
					leftInd = leftInd - 20;
				end
				closeToMax = false;
			elseif (rightInfoGain > midInfoGain)
				midInfoGain = rightInfoGain;
				oldMidInd = midInd;
				midInd = rightInd;
				leftInd = ceil((rightInd - oldMidInd)/2) + oldMidInd;
				rightInd = floor((rightInd - oldMidInd)/2) + rightInd;
				
				if closeToMax
					rightInd = rightInd + span*2;
					leftInd = leftInd - span*2;
				end
				closeToMax = false;
			else % close to max
				if ~closeToMax
					closeIter = iter;
				end
				closeToMax = true;
				
				% fine steps
				if iter - closeIter > span
					foundMax = true;
				end
				
				leftInd = midInd - (iter - closeIter) - 1;
				rightInd = midInd + (iter - closeIter) + 1;
			end
			
			% prevent infinite loop
			iter = iter + 1;
			if (iter > length(set{1}))
				break;
			end
	
		end

		if (midInfoGain > infoGainBest)
			attributeBest = att;
			infoGainBest = midInfoGain;
			indBest = midInd;
			thresholdBest = (set{2}(att,indBest) + set{2}(att,indBest + 1)) / 2;
		end
		
	end
	
	if (~infoGainBest || ~attributeBest)
		fprintf('    couldn''t find attribute to split on\n');
		% return the single node tree root with label
		% = mode label of set
		tree = {'leaf', mode(set{1})};
		return
	end
	
	% recur on the sublists obtained by splitting on attribute_best, 
	% and add those nodes as children of node
	% need to re-sort according to attribute_best
	[~,I] = sort(set{2}(attributeBest,:));
	set{2} = set{2}(:,I);
	set{1} = set{1}(I);
	subsets = getSubsets(set, attributeBest, indBest);
	subtree1 = trainDecisionTree(subsets{1}, minLeaf);
	subtree2 = trainDecisionTree(subsets{2}, minLeaf);
	
	% create a decision node that splits on attribute_best
	tree = {'node', attributeBest, thresholdBest, subtree1, subtree2};
	
	return
	
end
