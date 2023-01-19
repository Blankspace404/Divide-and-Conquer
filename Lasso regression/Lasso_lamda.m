data= YearPredictionMSD;

x=data(1:1000,2:end);
y=data(1:1000,1);
X = table2array(x)
Y = table2array(y)
[b,fitinfo] = lasso(X,Y,'CV',10);
lassoPlot(b,fitinfo,'PlotType','Lambda','XScale','log');
lam = fitinfo.Index1SE;
fitinfo.MSE(lam)
b(:,lam)
rhat = X\Y
res = X*rhat - Y;     % Calculate residuals
MSEmin = res'*res/1000