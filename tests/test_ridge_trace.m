rng(123);
% Pick some coefficients check lambdas are 'stabilizing' the estimations
n=2500;
num_params=199;
low_x_bound=-50;
high_x_bound=50;  
X=zeros(n,num_params);

for i = 1:num_params
    range=(high_x_bound-low_x_bound)*randn(2,1)+low_x_bound;
    lo=min(range);
    hi=max(range);
        
    dist_type=randi([1,3]);
    if dist_type==1
        X(:,i)=lo+(hi-lo)*randn(n,1);
    elseif dist_type==2
        X(:,i)=lo+(hi-lo)*rand(n,1);
    else
        X(:,i)=lo+(hi-lo)*exprnd(n,1);
    end
end

% make 30% of the explanatory variables collinear
num_params=199;
cols=(1:num_params);
X_corr=X;
num_collns=floor(num_params*0.3);    
collns=randperm(num_params,num_collns);
others=setdiff(cols,collns);
    
for k = collns
    ind_cols_to_corr=randi(length(others));
    cols_to_corr=others(ind_cols_to_corr);
    
    X_corr(:,k)=(low_bound+(high_bound-low_bound)*randn(1,1))*X(:,cols_to_corr);
end

% generate coefficient values
low_bound=-20;
high_bound=20;
beta = low_bound + (high_bound-low_bound)*randn(num_params+1,1);
    
epsilons = 0.5 * randn(n,1);
    
y = [ones(n,1),X_corr]*beta+epsilons; 
lambdas=exp(-9:0.5:8);
bs=zeros(length(lambdas),num_params+1);
mses=zeros(length(lambdas),2);
for i = 1:length(lambdas)
    [b,~,mse,~] = ridge_regression(X,y);
    bs(i,:)=[transpose(b)];
    mses(i,:)=[lambdas(i),mse];

end

% b vs lambda value
%  select 8 bs that are within the same range (more visually pleasing)
lower=-15;
upper=20;
bs_select=bs(:,all(bs>=lower & bs<=upper,1));
total_cols=size(bs_select,2);
rand_inds=randperm(total_cols,8);
bs_select=bs_select(:,rand_inds);

lambdas_fine=(-9:0.01:8);
figure('Name','Ridge trace');
bs_smooth=zeros(length(lambdas_fine),8);
hold on
for i =1:8
    bs_smooth(:,i)=spline(log(lambdas),bs_select(:,i),lambdas_fine);
    plot(lambdas_fine,bs_smooth(:,i),'LineWidth',2)
end
hold off
xlim([0,5]);
saveas(gcf,'ridge_trace.fig');

% MSE vs lambda value
figure('Name','MSE vs Lambdas');
mses_smooth=spline(log(lambdas),mses(:,2),lambdas_fine);
plot(lambdas_fine,mses_smooth,'LineWidth',2);
xlim([0,5]);
saveas(gcf,'mse_vs_lambda.fig');
