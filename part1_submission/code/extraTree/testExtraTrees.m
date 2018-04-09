%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to test the ensemble of extra-trees

parameters:
	- test
		struct with class labels {2} and data vectors {3}
	- trees
		struct of structs of structs that represent the ensemble of
		extra-trees

%}

function test = testExtraTrees(test, trees)

	% for each test sample
	for i = 1:size(test{1},1)

		votes = zeros(size(trees, 1), 1);
		
		% for each extra-tree
		for tree = 1:size(trees)
			
			% pass through the tree
			tree_walked = trees{tree};
			classified = false;
			% this sample's features
			S = test{2}(:,i);

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
					if (attribute_test <= threshold)
						% choose left branch
						tree_walked = tree_walked{4};
					else
						tree_walked = tree_walked{5};
					end

				% if we're on a leaf
				else
					votes(tree) = tree_walked{2};
					%fprintf('tree %d votes class %d\n', tree, votes(tree));
					classified = true;

				end		

			end
		
		end
		
		% take majority vote of all trees
		test{1}(i,2) = mode(votes);

	end

end
