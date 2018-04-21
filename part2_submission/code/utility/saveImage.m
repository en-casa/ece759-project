%{

18/03/05

%}

function saveImage(filename)
    root = '../images/';
    file = strcat(root, filename);
    print(file, '-dpng');
end
