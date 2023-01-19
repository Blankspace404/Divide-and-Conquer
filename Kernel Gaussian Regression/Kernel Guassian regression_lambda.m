data= RadiusQueries;
x=data(1:2000,1:end-1);
y=data(1:2000,end);
x = table2array(x)
y = table2array(y)
%Use all samples or specify a number 
N = length(y);

%Random shuffle
shuffled_indexes = randperm(N);
x = x(shuffled_indexes,:);
y = y(shuffled_indexes,:);



%Train split amount
n_folds = 10;

%Validation splits split

N_train =int32(N*(n_folds-1)/n_folds);
N_test = N-N_train;

x_test = x(N_train+1:end,:);
x_train = x(1:N_train,:);

y_test = y(N_train+1:end,:);
y_train = y(1:N_train,:);

%Easy to use lambda hyperparameter optimization function

%Build gaussian kernel K
    K = KRR_Build_K(x_train);

    %Build gaussian kernel k
    k = KRR_Build_k(x_train,x_test);

    %Look for lambda with lowest k-fold cross-validation error
    interval = 1/n_folds;

    total_intervals = (1/interval);

    least_mean_squared_error = realmax;
    %best_lambda = interval;

    %for lambda = interval:interval:1
        %i = int32(lambda/interval);

        %fprintf('Testing Lambda Value #: %d of %d\n',i,total_intervals);

        %y_predicted = KRR_Predict(y_train,x_test,K,k,lambda);

        %error = Mean_Square_Error(y_test,y_predicted);

        %if error < least_mean_squared_error
            %least_mean_squared_error = error;
            %best_lambda = lambda;
        %end
    %end
best_prediction = [N_test,1];

error = zeros(1/interval,1);

for lambda = interval:interval:1
    i = int32(lambda/interval);
    
    fprintf('Testing Lambda Value #: %d of %d\n',i,total_intervals);
            
    y_predicted = KRR_Predict(y_train,x_test,K,k,lambda);
    
    error(i,1) = Mean_Square_Error(y_test,y_predicted);
    
    if error(i,1) < least_mean_squared_error
        least_mean_squared_error = error(i,1);
        best_prediction = y_predicted;
    end
end

    
    
figure;
hold on;

scatter3(x_test(:,1),x_test(:,2),y_test,'g');
scatter3(x_test(:,1),x_test(:,2),y_predicted,'r');

title('Actual vs Predicted');
xlabel({'x_1'});
ylabel({'x_2'});
zlabel('y');
view([-47.1 4.4]);
legend('y','predicted');

hold off;
figure;
hold on;

scatter(interval:interval:1,error,'b');

title('Lambda vs Error');
xlabel({'lambda'});
ylabel({'error'});

hold off;

figure;
hold on;


%%
function [K]=KRR_Build_K(x_train)
    N_train = length(x_train);

    K=zeros(N_train,N_train);
    for j=1:N_train
     for i=1:N_train
        K(i,j)=exp(-norm(x_train(j,:)-x_train(i,:)));
     end
    end
end


function [k]=KRR_Build_k(x_train,x_test)
    N_test = length(x_test);
    N_train = length(x_train);

    k=zeros(N_test,N_train);
    for j=1:N_train
     for i=1:N_test
        k(i,j)=exp(-norm(x_train(j,:)-x_test(i,:)));
     end
    end
end

function [y_predicted]=KRR_Predict(y_train,x_test,K,k,lambda)
    N_test = length(x_test);
    N_train = length(y_train);

    y_predicted=zeros(N_test,1);
    for i=1:N_test
        y_predicted(i,1)= y_train'*((K+ lambda*eye(N_train))\k(i,:)');
    end
end

function [error]=Mean_Square_Error(y,y_predicted)
    error = norm(y_predicted-y)^2/length(y);
end