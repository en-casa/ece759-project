%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get entropy of a set

%}

function ent = getEntropy(set)

	ent = 0;

	classes = unique(set{2});
	
	for i = 1:length(classes)
		
		class = classes(i);
		% proportion of the # of elements in class x 
		% to the number of elements in set
		inds = find(set{2} == class);
		probX = numel(inds)/numel(set{2});
		
		% entropy
		ent = ent - probX*log2(probX);
		
	end

end