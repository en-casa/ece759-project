%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get subsets of a set

%}

function subsets = getSubsets(set, attribute, index)
	
	% exclude the attribute row 
	indsExcludeAttr = 1:size(set{3},1) ~= attribute;

	% these form the left branch (expensive)
	%indsLessThan = set{3}(attribute,:) < threshold;
	
	%subsets{1} = {set{1}(:, indsLessThan), ...
	%	set{2}(indsLessThan), ...
	%	set{3}(indsExcludeAttr, indsLessThan)};
	
	% right branch forms a set partition
	%subsets{2} = {set{1}(:, ~indsLessThan), ...
	%	set{2}(~indsLessThan), ...
	%	set{3}(indsExcludeAttr, ~indsLessThan)};
	
	% these form the left branch
	subsets{1} = {set{1}(:, 1:index), ...
		set{2}(1:index), ...
		set{3}(indsExcludeAttr, 1:index)};
	
	% right branch forms a set partition
	if index+1 > numel(set{2})
		subsets{2} = {[],[],[]};
	else
		subsets{2} = {set{1}(:, index+1:end), ...
			set{2}(index+1:end), ...
			set{3}(indsExcludeAttr, index+1:end)};
	end

end