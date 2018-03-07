%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to get entropy of a set

%}

function ent = getEntropy(set)
	
	% initialize entropy
	ent = 0;

	if isempty(set{1})
		return
	end
	
	% find unique ~and~ count instances in O(n) (saves n computations)
	classes = [set{1}(1) 1];
	for i = 2:length(set{1})
		
		ind = find(classes(:,1) == set{1}(i), 1);
		if isempty(ind)
			% new class encountered
			classes = [classes; set{1}(i) 1];
		else
			% increment counter
			classes(ind, 2) = classes(ind, 2) + 1;
		end
		
	end
	
	for i = 1:size(classes, 1)
		
		% proportion of the # of elements in class x 
		% to the number of elements in set
		probX = classes(i,2)/length(set{1});
		
		% entropy
		ent = ent - probX*log2(probX);
		
	end

end