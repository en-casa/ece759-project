%{

set up the figure

%}

function f = instantiateFig(num, color)
    
    if nargin == 0
       num = 1; 
       color = 'w';
    elseif nargin == 1
       color = 'w';
    end

    f = figure(num); 
    clf;
    f.InvertHardcopy = 'off';
    
    set(f, 'Color', color, ...
        'MenuBar', 'none', ...
        'units', 'normalized', ...
        'visible', 'on');
    %'Position', [0.43 0.054 0.56 0.91], ...
    
    % suppress image size warning - doesn't like how big it is
    warning('off','images:initSize:adjustingMag'); 
    
end
