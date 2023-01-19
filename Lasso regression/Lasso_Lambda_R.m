data = RadiusQueries;
X = table2array(data(1:1000,1:2));
Y = table2array(data(1:1000,3));
[b,fitinfo] = lasso(X,Y,'CV',10);
lassoPlot(b,fitinfo,'PlotType','Lambda','XScale','log');
lam = fitinfo.Index1SE;
fitinfo.MSE(lam)

b(:,lam)
rhat = X\Y
res = X*rhat - Y;     % Calculate residuals
MSEmin = res'*res/1000