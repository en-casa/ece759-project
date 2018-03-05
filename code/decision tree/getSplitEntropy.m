%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get entropy of a split set

%}

function splitEnt = getSplitEntropy(set, attribute, index)

	splitEnt = 0;

	subsets = getSubsets(set, attribute, index);
	
	for t = 1:length(subsets)
		
		% proportion of # elements in subsets to the # of elements in set
		probSub = numel(subsets{t}{2})/numel(set{2});
		
		splitEnt = splitEnt + probSub*getEntropy(subsets{t});
		
	end

end