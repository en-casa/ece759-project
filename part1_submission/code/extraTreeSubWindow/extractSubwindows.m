%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

this function extracts subwindows from raw pixel data

%}

function [setSW] = extractSubwindows(set, numWindows)

	% instantiate sets
	setSWLabels = zeros(length(set{1})*numWindows,1);
	setSWData = zeros(size(set{2},1)/numWindows, size(set{2},2)*numWindows);
	
	% duplicate classes and split up images
	for i = 0:length(set{1})-1
		
		startInd = 1 + numWindows*i;
		setSWLabels(startInd:(startInd + numWindows - 1)) = set{1}(i+1);
		
		setSWData(:,startInd:(startInd + numWindows - 1)) = ...
			reshape(set{2}(:,i+1), [size(setSWData,1),numWindows]);
		
	end
	
	setSW = {setSWLabels, setSWData};

end
