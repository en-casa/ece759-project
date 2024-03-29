%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to test a decision tree

parameters:
	- test
		struct with class labels {2} and data vectors {3}
	- tree
		struct of structs that represent the decision tree

%}

function test = testDecisionTree(test, tree)

	% for each test sample
	for i = 1:size(test{2},1)

		% pass through the tree
		tree_walked = tree;
		classified = false;
		S = test{3}(:,i);
		
		while (~classified)

			% if we're on a node
			if strcmp('node', char(tree_walked(1)))

				attribute_tree = tree_walked{2};
				threshold = tree_walked{3};
				attribute_test = S(attribute_tree);

				% remove attribute
				inds = 1:length(S) ~= attribute_tree;
				S = S(inds);

				% compare attribute value to threshold
				if (attribute_test < threshold)
					% choose left branch
					tree_walked = tree_walked{4};
				else
					tree_walked = tree_walked{5};
				end

			% if we're on a leaf
			else
				test{2}(i,2) = tree_walked{2};
				classified = true;

			end		

		end

	end

end
