%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get subsets of a set

%}

function subsets = getSubsets(set, attribute, index)
	
	% exclude the attribute row 
	indsExcludeAttr = 1:size(set{2},1) ~= attribute;
	
	% these form the left branch
	subsets{1} = {set{1}(1:index), ...
		set{2}(indsExcludeAttr, 1:index)};
	
	% right branch forms a set partition
	subsets{2} = {set{1}(index+1:end), ...
		set{2}(indsExcludeAttr, index+1:end)};

end