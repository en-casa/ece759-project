%{

n casale
ece 759

18/02/19

generate a colormap

%}

clear;

% red-blue colormap

% number of interpolated points between [min, max]
N = 256;
min = 0;
max = 1;

% red
r = ones(N, 1);
g = linspace(min, max, N)';
b = g;

red = [r g b];

% blue
r = linspace(min, max, N)';
g = r;
b = ones(N, 1);

blue = [r g b];

red_blue = [red; flipud(blue)];

save red_blue.mat red_blue

I = zeros(N*2, N*2, 3);

for i = 1:N*2
    for j = 1:N*2
        I(i, j, 1:3) = red_blue(i, :);
    end
end

image(I);