data = YearPredictionMSD;
data = data(1:10000,:)
[n,p] = size(data);
n_train = n*0.8
n_test = n*0.2
x = table2array(data(:,2:p));
y = table2array(data(:,1));
x_test = x(n_train:end,:);
y_test = y(n_train:end,:);
format long; 
k=10;
lambda_array = [0.0001,0.00025,0.0005,0.001,0.002,0.003,0.004]
MSE_T = []
for j = 1: length(lambda_array)
    lambda = lambda_array(j)
MSE = []
for i = 1:k
    T_train = table2array(data(1:n_train,:));    
    T_hold = T_train((n_train/k)*(i-1)+1:(n_train/k)*i,:);
    [ia, ib] = ismember(T_train, T_hold, 'rows')
    T_train(ia, :) = []
    x_train = T_train(:,2:p);
    y_train = T_train(:,1);
    x_hold = T_hold(:,2:p);
    y_hold = T_hold(:,1);
    X_train = [ones(length(x_train),1) x_train]
    X_hold = [ones(length(x_hold),1) x_hold]
    beta = inv(X_train'*X_train + lambda*eye(p))*X_train'*y_train
    pred = X_hold*beta
    MSE = [MSE mse(y_hold, pred)]        
end
MSE_mean = mean(MSE, 2)
MSE_T = [MSE_T MSE_mean]
end
plot(lambda_array, MSE_T)