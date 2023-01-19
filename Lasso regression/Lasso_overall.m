 
data= YearPredictionMSD;
[n,p] = size(data);

randows = randperm(n);
T = data(randows(1: 1100),:);
n_train = 1000;
n_test = 100;


x=data(1:1000,2:end);
y=data(1:1000,1);
x_train = table2array(x)
y_train = table2array(y)

x_test = table2array(data(1001:1100,2:end));
y_test = table2array(data(1001:1100,1));
tic
 %rng default % For reproducibility
[B,FitInfo] = lasso(x_train,y_train,'CV',10);
lassoPlot(B,FitInfo,'PlotType','CV'); % Look at MSE vs Lambda
coeff = B(:,FitInfo.IndexMinMSE); % Regression coefficients for min MSE, use these coefficients.
Ypredict =  x_test*coeff + FitInfo.Intercept(FitInfo.IndexMinMSE)
toc

Ypredict = round(Ypredict)