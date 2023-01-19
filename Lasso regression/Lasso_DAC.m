data= YearPredictionMSD;
randows = randperm(1000);
m = 2;
x=data(1:1000,2:end);
y=data(1:1000,1);
X = table2array(x)
Y = table2array(y)
coeff_total = zeros(90,1);
Intercept = zeros(1, 100)
tic
for k = 1: m
    Tsub = data(randows(round(1000/m)*(k-1)+1:round(1000/m)*k),:);
    xsub = table2array(Tsub(:,2:end));
    ysub = table2array(Tsub(:,1));
    [B,FitInfo] = lasso(xsub,ysub,'CV',10);
    coeff = B(:,FitInfo.IndexMinMSE); % Regression coefficients for min MSE, use these coefficients.
    coeff_total = coeff_total + coeff
    Intercept_sub = FitInfo.Intercept(FitInfo.IndexMinMSE)
    Intercept = Intercept+ Intercept_sub
end
 %rng default % For reproducibility
%[B,FitInfo] = lasso(X,Y,'CV',10);
lassoPlot(B,FitInfo,'PlotType','CV'); % Look at MSE vs Lambda
coeff_mean = coeff_total/m
intercept_mean = Intercept/m
%coeff = B(:,FitInfo.IndexMinMSE); % Regression coefficients for min MSE, use these coefficients.
%Ypredict =  X*coeff + coeff = B(:,FitInfo.IndexMinMSE); % Regression coefficients for min MSE, use these coefficients.
Ypredict =  X*coeff_mean + intercept_mean
toc