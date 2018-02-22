%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

returns:
	
%}

function [faces, labels] = loadYaleB()

	load('YaleB_32x32.mat')
	
	faces = fea;
	labels = gnd;
	
	clear fea gnd

end