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
	ind_best = 0;
	
	% for each attribute
	for att = 1:size(set{3},1)
		
		% sort by attribute value
		[~,I] = sort(set{3}(att,:));
		set{3} = set{3}(:,I);
		set{2} = set{2}(I);
		set{1} = set{1}(:,I);
		
		% optimize IG over thresholds
		% see page 2: https://www.jair.org/media/279/live-279-1538-jair.pdf
		% for each member of the set
		
		% two loops to improve computational complexity
		if size(set{3},2) > 10000
			skipMed = false;
			J = round(linspace(1, size(set{3},2)-1, 100));
		elseif size(set{3},2) > 100
			skipMed = false;
			J = round(linspace(1, size(set{3},2)-1, 50));
		else
			skipMed = true;
			J = 1:size(set{3},2);
		end
		
		% coarse approximation of best IG
		for j = J
			
			if j > size(set{3},2)-1
					break;
			end
			
			fprintf('-- a: %d, j: %d\n', att, j);
			
			% naïve: split on halfway between min and max 
			% (~70% error rate with MNIST)
			%threshold = (max(set{3}(att,:)) - min(set{3}(att,:))) / 2;
			
			% split halfway between adjacent values
			threshold = (set{3}(att,j) + set{3}(att,j+1)) / 2;

			% find the normalized information gain ratio from splitting on i.
			info_gain = thisSetEntropy - getSplitEntropy(set, att, j);

			if (info_gain > info_gain_best)
				attribute_best = att;
				threshold_best = threshold;
				info_gain_best = info_gain;
				ind_best = j;
			end
			
		end
		
		if ~skipMed
		
			% medium approximation of best IG
			stride_J = J(2)-J(1)+1;
			if size(set{3},2) > 10000
				K = ind_best-stride_J:10:ind_best+stride_J;
			else
				K = ind_best-stride_J:10:ind_best+stride_J;
			end
			
			for k = K
				
				if k < 1 
					continue;
				elseif k > size(set{3},2)-1
					break;
				end
			
				fprintf('-- a: %d, j: %d, k: %d\n', att, j, k);

				% split halfway between adjacent values
				threshold = (set{3}(att,k) + set{3}(att,k+1)) / 2;

				% find the normalized information gain ratio from splitting on i.
				info_gain = thisSetEntropy - getSplitEntropy(set, att, k);

				if (info_gain > info_gain_best)
					attribute_best = att;
					threshold_best = threshold;
					info_gain_best = info_gain;
					ind_best = k;
				end
			
			end
			
			% fine approximation of best IG
			L = ind_best-10:ind_best+10;
			
			for l = L
				
				if l < 1 
					continue;
				elseif l > size(set{3},2)-1
					break;
				end
			
				fprintf('-- a: %d, j: %d, k: %d, l: %d\n', att, j, k, l);

				% check if adjacent values are in the same class
				% splitting on that value wouldn't improve the IG
				% so we skip it
				if (set{2}(l) == set{2}(l+1))
					fprintf('same class\n');
					continue;
				end

				% split halfway between adjacent values
				threshold = (set{3}(att,l) + set{3}(att,l+1)) / 2;

				% find the normalized information gain ratio from splitting on i.
				info_gain = thisSetEntropy - getSplitEntropy(set, att, l);

				if (info_gain > info_gain_best)
					attribute_best = att;
					threshold_best = threshold;
					info_gain_best = info_gain;
					ind_best = l;
				end
			
			end
			
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
	subsets = getSubsets(set, attribute_best, ind_best);
	subtree1 = trainDecisionTree(subsets{1});
	subtree2 = trainDecisionTree(subsets{2});
	
	% create a decision node that splits on attribute_best.
	tree = {'node', attribute_best, threshold_best, subtree1, subtree2};
	
	return
	
end
