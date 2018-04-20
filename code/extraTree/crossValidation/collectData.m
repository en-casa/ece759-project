%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/04/20

collect cross validation data, display

%}


load('extraTree/crossValidation/cv_mnist_et.mat')

avgErrorRates = mean(errorRates(1:11,:),2);
avgTrainTimes = mean(trainTimes(1:11,:),2);

numTreess = [10 50 100];

f = instantiateFig(1);
yyaxis left;
plot(numTreess, avgErrorRates,'LineWidth', 2);
xlabel('minLeaf'); ylabel('Avg. 5-Fold Error (%)');
yyaxis right;
plot(numTreess, avgTrainTimes, 'LineWidth', 2);
ylabel('Training Time (minutes)');
title('Extra-Trees Average Error and Training Time Across numTrees for 5-Fold CV on MNIST');
prettyPictureFig(f);

%% yale b

load('extraTree/crossValidation/cv_yaleb_et.mat')

avgErrorRates = mean(errorRates,2);
avgTrainTimes = mean(trainTimes,2);

numTreess = [10 50 100];

f = instantiateFig(2);
yyaxis left;
plot(numTreess, avgErrorRates,'LineWidth', 2);
xlabel('numTrees'); ylabel('Avg. 5-Fold Error (%)');
yyaxis right;
plot(numTreess, avgTrainTimes, 'LineWidth', 2);
ylabel('Training Time (minutes)');
title('Extra-Trees Average Error and Training Time Across numTrees for 5-Fold CV on Yale B');
prettyPictureFig(f);