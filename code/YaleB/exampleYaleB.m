%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/03/16

Load Extended Yale B and show some examples
	
	source: 
	http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html
	
	info:
	https://computervisiononline.com/dataset/1105138686

%}

clear;
close all;

addpath('./utility');
addpath('YaleB', './YaleB/data');

[faces, labels] = loadYaleB();

sz_col = 42; 
sz_row = 48;
stride = 12; 
rows = 8; 

Y = zeros(sz_row*rows, sz_col*stride); 
for i=0:rows-1 
  	for j=0:stride-1 
    	Y(i*sz_row+1:(i+1)*sz_row,j*sz_col+1:(j+1)*sz_col) ...
			= reshape(faces(i*stride+j+1,:), [sz_row,sz_col]); 
    end
end

f = instantiateFig(2);
imagesc(Y);
colormap(gray);
prettyPictureFig(f);