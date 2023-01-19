data = YearPredictionMSD;
%data = data(1:10000,:)
[n,p] = size(data);
n_train = n*0.8
n_test = n*0.2
x = table2array(data(:,2:p));
y = table2array(data(:,1));
x_train = x(1:n_train,:);
y_train = y(1:n_train,:);
x_test = x(n_train:end,:);
y_test = y(n_train:end,:);
format long; 
lambda = 0.002;

m_array = [2, 5, 10,50,100,200]
b_t = []
t = []
MSE = []
for j = 1: length(m_array)
    b = []
    m = m_array(j)
    tic
  for k = 1:m      
      xsub = x_train((n_train/m)*(k-1)+1:(n_train/m)*k,:);
      ysub = y_train((n_train/m)*(k-1)+1:(n_train/m)*k,:);
      Xsub = [ones(length(xsub),1) xsub]
      b_sub = inv(Xsub'*Xsub + lambda*eye(p))*Xsub'*ysub
      b = [b,b_sub];
 
  end
  t = [t toc]
  b_mean = mean(b,2);
  b_t = [b_t, b_mean]
end
toc
X_test = [ones(length(x_test),1) x_test]
for i = 1: length(m_array)
    y_predict = X_test*b_t(:,i);
    MSE = [MSE, mse(y_test, y_predict)];
end
plot(m_array,MSE,'b')
plot(m_array, t,'r')