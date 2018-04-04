%Financial mathematics
%Black Scholes formula for dividend and non dividend palying stock;
%This is a copy right of Krishna Prasad, IIT Delhi,New Delhi
%For further information email to kprasad.iitd@gmail.com or +919891370585
%%Start of coding, Symbols have their usual meaning 
s0 =input('Enter the stock price at t = 0 : S0 =  ');
k = input('Enter the strike price for underlying asset ');
r = input('Enter the risk free rate of interest ');
T=  input('Enter the expiry time ');
sigma=input('Enter the volatility for underlying asset ');
CH = 1;
while CH == 1
    fprintf('1 : for Dividend \n 2 : non Dividend \n ');
    choice = input(' choice ');
    switch choice %for Dividend and non dividend 
        case 1
            disp('DATA For Dividend paying  Stocks ');
            dtime = input('Enter the time in  array  ');
            damount = input('Enter the dividend paying amount in array ');
            rd = -r*dtime;
            erd = exp(rd);
            ds = damount.*erd;
            sa = s0-sum(sum(ds));
            lsa = (log(sa/k)+(r+(sigma*sigma)/2)*T);
            d1 = lsa/(sigma*sqrt(T));
            d2 = d1 - sigma*sqrt(T);
            
            fprintf('1 : call option \n2 : put option \n ');
            ch = input(' choice for put or call '); 
            if ch == 1 %for call option
                c0 = sa*normcdf(d1)-k*exp(-r*T)*normcdf(d2);
                fprintf('The value of call option price is C0 = %4.4f\n',c0);
            end
            if ch == 2 % for put option
                p0 = k*exp(-r*T)*normcdf(-d2)-sa*normcdf(-d1);
                fprintf('The value of put option price is C0 = %4.4f\n',p0);
            end
            if ch ~=1 && ch~=2
                disp('In valid choice');
            end
        case 2 % for non dividend 
            lso = (log(s0/k)+(r+(sigma.*sigma)/2)*T);
            d1 = lso/(sigma*sqrt(T));
            d2 = d1 - sigma*sqrt(T);
            fprintf('1 : call option \n 2 : put option \n ');
            ch = input(' choice for put or call ');
            if ch == 1 % call option
                c0 = s0*normcdf(d1)-k*exp(-r*T)*normcdf(d2);
                fprintf('\nThe value of call option price is C0 = %f\n',c0);
            end
            if ch == 2 % put option
                c0 = s0*normcdf(d1)-k*exp(-r*T)*normcdf(d2);
                p0 = c0-s0+k*exp(-r*T);
                fprintf('\nThe value of put option price is C0 = %4.4f\n',p0);
            end
            if ch ~=1 && ch~=2
                disp('In valid choice');
            end
    end
    CH = input('1 : For more option pricing otherwise any no : ');
end


%%
function [acc_test acc_train] = classifyNN(num_neighbors,test_data, train_data, test_label, train_label)
%
% Description:  
% Classify test data using Nearest Neighbor method withEuclidean distance
% criteria. 
% 
% Usage:
% [accuracy] = classifyNN(test_data, train_data, test_label, train_label)
%
% Parameters:
% test_data = test images projected in reduced dimension  dxtn
% train_data = train images projected in reduced dimension dxN
% test_label = test labels for each data tn x 1
% train_label = train labels for each train data Nx1
%
% Returns:
% accuracy: a scalar number of the classification accuracy

train_size = size(train_data, 2);
train_N = size(train_data,2);
test_N = size(test_data, 2);
counter = zeros(test_N, 1);
% performance on test data
parfor test_n = 1:test_N

    test_mat = repmat(test_data(:, test_n), [1,train_size]);
    distance = sum(abs(test_mat - train_data).^2);
    [~,distances_index] = sort(distance);
    neighbors=distances_index(1:num_neighbors);
    a = mode(train_label(neighbors));
    if a == test_label(test_n)
        counter(test_n) = counter(test_n) + 1;
    end
end

acc_test = double(sum(counter)) / test_N;
% performance on training data
counter = zeros(train_N, 1);
parfor test_n = 1:train_N

    test_mat = repmat(train_data(:, test_n), [1,train_size]);
    distance = sum(abs(test_mat - train_data).^2);
    [~,distances_index] = sort(distance);
    neighbors=distances_index(2:num_neighbors+1); % excluding itself
    a = mode(train_label(neighbors));
    if a == train_label(test_n)
        counter(test_n) = counter(test_n) + 1;
    end
end

acc_train = double(sum(counter)) / train_N;