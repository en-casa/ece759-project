%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

function to test a decision tree

%}

function test = testDecisionTree(test, sz, N_te, tree)

	% for each test sample
	for i = 1:N_te

		% generate features via svd
		[~, S, ~] = svd(reshape(test{1}(:,i),[sz, sz]));
		test{3}(:,i) = diag(S);
		S = diag(S);

		% pass through the tree
		tree_walked = tree;
		classified = false;
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
