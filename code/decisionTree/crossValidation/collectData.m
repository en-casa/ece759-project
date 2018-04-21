%{

kudiyar orazymbetov
n casale

ECE 759 Project
18/04/20

collect cross validation data, display

%}


load('decisionTree/crossValidation/cv_mnist_dt.mat')

avgErrorRates = mean(errorRates(1:11,:),2);
avgTrainTimes = mean(trainTimes(1:11,:),2);

minLeaves = 0:50:500;
minLeaves(1) = 1;

f = instantiateFig(1);
yyaxis left;
plot(minLeaves, avgErrorRates,'LineWidth', 2);
xlabel('minLeaf'); ylabel('Avg. 5-Fold Error (%)');
yyaxis right;
plot(minLeaves, avgTrainTimes, 'LineWidth', 2);
ylabel('Training Time (minutes)');
title('Decision Tree Average Error and Training Time Across minLeaf for 5-Fold CV on MNIST');
prettyPictureFig(f);
set(findall(gca,'type','text'),'FontSize',17,'fontWeight','bold')

saveImage('cv_dt_mnist');

%% yale b

load('decisionTree/crossValidation/cv_yaleb_dt.mat')

avgErrorRates = mean(errorRates,2);
avgTrainTimes = mean(trainTimes,2);

minLeaves = 0:30:300;
minLeaves(1) = 1;

f = instantiateFig(2);
yyaxis left;
plot(minLeaves, avgErrorRates,'LineWidth', 2);
xlabel('minLeaf'); ylabel('Avg. 5-Fold Error (%)');
yyaxis right;
plot(minLeaves, avgTrainTimes, 'LineWidth', 2);
ylabel('Training Time (minutes)');
title('Decision Tree Average Error and Training Time Across minLeaf for 5-Fold CV on Yale B');
prettyPictureFig(f);
set(findall(gca,'type','text'),'FontSize',17,'fontWeight','bold')

saveImage('cv_dt_yaleb');