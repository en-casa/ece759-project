%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get subsets of a set

%}

function subsets = getSubsets(set, attribute, threshold)
	
	% these form the left branch
	indsLessThan = set{3}(attribute,:) < threshold;
	
	subsets{1} = {set{1}(:, indsLessThan), ...
		set{2}(indsLessThan), ...
		set{3}(:, indsLessThan)};
	
	% right branch forms a set partition
	subsets{2} = {set{1}(:, ~indsLessThan), ...
		set{2}(~indsLessThan), ...
		set{3}(:, ~indsLessThan)};

end