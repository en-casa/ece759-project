%{

17/8/20

%}

function k = saveImages(root, k)
    k = k+1;
    file = strcat(root, sprintf('%d',k));
    print(file, '-dpng');
end
