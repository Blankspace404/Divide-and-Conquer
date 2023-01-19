data= RadiusQueries;
x=data(1:2000,1:end-1);
y=data(1:2000,end);
x = table2array(x);
y = table2array(y);
%Use all samples or specify a number 
N = length(y);

%Random shuffle
shuffled_indexes = randperm(N);
x = x(shuffled_indexes,:);
y = y(shuffled_indexes,:);
t_split = 0.8;
N_train = N*t_split;
N_test = N-N_train;
MSE = [];

x_test = x(N_train+1:end,:);
x_train = x(1:N_train,:);

y_test = y(N_train+1:end,:);
y_train = y(1:N_train,:);
tic;
m_array = [1, 5,10,100,200,400,800];
for w = 1:length(m_array)
    m = m_array(w)
y_predicted_sample_total = [];
y_predicted_total = zeros(length(x_test),1);

for z = 1:m
    xsub = x_train(round(N_train/m)*(z-1)+1:round(N_train/m)*z,:);
    ysub = y_train(round(N_train/m)*(z-1)+1:round(N_train/m)*z,:);
    
    Ksub = zeros(round(N_train/m), round(N_train/m));
   for j=1:round(N_train/m)
       for i = 1:round(N_train/m)
        Ksub(i,j) = exp(-norm(xsub(j,:)-xsub(i,:)));
        
       end
   end
   %K = [K Ksub]
   y_predicted_sample = zeros(round(N_train/m),1);
   lambda = 1;
   for i = 1:round(N_train/m)
       if mod(i,10)==0
        fprintf('Training on Sample: %d of %d\n',i,round(N_train/m));
       end
    y_predicted_sample(i,1) = ysub'*((Ksub+lambda*eye(round(N_train/m)))\Ksub(i,:)');
    
   end
   y_predicted_sample_total = [y_predicted_sample_total; y_predicted_sample];
   
   ksub = zeros(N_test,round(N_train/m));
   for j = 1:round(N_train/m)
       for i = 1:N_test
        ksub(i,j) = exp(-norm(xsub(j,:)-x_test(i,:)));
       end
   end
   %k = [k ksub];  
   y_predicted = zeros(N_test,1);
   for i = 1:N_test
    y_predicted(i,1)=ysub'*((Ksub+lambda*eye(round(N_train/m)))\ksub(i,:)');
    
   end
   y_predicted_total = y_predicted_total + y_predicted;
end
toc;

y_predicted_mean = y_predicted_total/m;
in_sample_error = norm(y_predicted_sample_total-y_train)^2/N_train;

out_sample_error = norm(y_predicted_mean-y_test)^2/N_test;
MSE = [MSE, out_sample_error];
end
plot(m_array,MSE)