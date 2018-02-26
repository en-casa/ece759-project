%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to train a decision tree

%}

function [tree] = trainDecisionTree(train)

	% Check for base cases
	% 1. all samples belong to the same class
	if (min(train{2}) == max(train{2}))
		% create leaf node for the decision tree saying to choose that class
		
	end
	
	% 2. no feature 
	
	% For each attribute a, find the normalized information gain ratio from splitting on a.
	% Let a_best be the attribute with the highest normalized information gain.
	% Create a decision node that splits on a_best.
	% Recur on the sublists obtained by splitting on a_best, and add those nodes as children of node.

end
