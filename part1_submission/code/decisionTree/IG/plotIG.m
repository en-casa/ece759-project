%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

test to illustrate the information gain over thresholds

%}

function [info_gain, ind_best] = plotIG(set)

	thisSetEntropy = getEntropy(set);
	
	% let attribute_best be the attribute with the highest 
	% normalized information gain (after splitting)
	attribute_best = 0;
	threshold_best = 0;
	info_gain_best = 0;
	ind_best = 0;
	
	% for first attribute
	att = 1;
		
	% sort by attribute value
	[~,I] = sort(set{3}(att,:));
	set{3} = set{3}(:,I);
	set{2} = set{2}(I);
	set{1} = set{1}(:,I);

	% optimize IG over thresholds
	% see page 2: https://www.jair.org/media/279/live-279-1538-jair.pdf
	% for each member of the set

	J = 1:size(set{3},2);
	info_gain = zeros(size(set{3},2),1);

	% coarse approximation of best IG
	for j = J

		if j > size(set{3},2)-1
				break;
		end

		fprintf('-- a: %d, j: %d\n', att, j);

		% find the normalized information gain ratio from splitting on i.
		info_gain(j) = thisSetEntropy - getSplitEntropy(set, att, j);

		if (info_gain(j) > info_gain_best)
			info_gain_best = info_gain(j);
			ind_best = j;
		end

	end
	
	return
	
end
