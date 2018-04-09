function prettyPictureFig(f, color)

    if nargin == 1
        color = [1 1 1];
    end

    set(f, 'Color', color, ...
        'MenuBar', 'none', ...
        'Position', [0 0 0.6 0.8], ...
        'units', 'normalized', ...
        'visible', 'on');
    
    set(findall(gca,'type','text'),'FontSize',20,'fontWeight','bold')
    set(findall(f, '-property', 'FontSize'), 'FontSize', 13, 'fontWeight', 'bold')

end