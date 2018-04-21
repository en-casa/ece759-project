%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get subsets of an unsorted

%}

function subsets = getSubsets_et(set, ind, threshold)
	
	% exclude the attribute row 
	indsExcludeAttr = 1:size(set{2},1) ~= ind;
	
	indsLessThanThres = set{2}(ind,:) <= threshold;
	
	% these form the left branch
	subsets{1} = {set{1}(indsLessThanThres), ...
		set{2}(indsExcludeAttr, indsLessThanThres)};
	
	% right branch forms a set partition
	subsets{2} = {set{1}(~indsLessThanThres), ...
		set{2}(indsExcludeAttr, ~indsLessThanThres)};

end