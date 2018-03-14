%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

returns:
	faces - 2414 faces in rows, 1024 columns represent the pixels in
		row-major order
	
	labels - 2414 labels for each face. i.e. 'who' each face belongs to
		int in [1,38]
%}

function [faces, labels] = loadYaleB_old()

	load('YaleB_32x32.mat')
	
	faces = fea;
	labels = gnd;
	
	clear fea gnd

end