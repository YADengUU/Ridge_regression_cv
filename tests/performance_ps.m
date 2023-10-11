
% time regression performance against p (number of parameters)

rng(123);

runtimes=zeros(19,2);

n = 5000;
low_x_bound=-50;
high_x_bound=50;
for counter = (1:19)

    num_params = 99+counter*100;
    cols=(1:num_params);
    low_bound = -20;
    high_bound = 20;
    beta = low_bound + (high_bound-low_bound)*randn(num_params+1,1);
    epsilons = 0.5 * randn(n,1);

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

    X_corr = X;
    num_collns = floor(num_params*0.3);
    collns = randperm(num_params,num_collns);
    others=setdiff(cols,collns);

    for k = collns
        ind_cols_to_corr=randi(length(others));
        cols_to_corr=others(ind_cols_to_corr);
                
        X_corr(:,k)=(low_bound+(high_bound-low_bound)*randn(1,1))*X(:,cols_to_corr);
    end

    % observations
    y = [ones(n,1),X_corr]*beta+epsilons;
    lambdas = exp(-9:5);

    % function handles for timing
    g = @()ridge_regression(X_corr,y,lambdas);
    tm = timeit(g);
    
    runtimes(counter,:)=[num_params,tm];

end

% plot runtime
ps_fine=(199:10:2000);
tms=spline(runtimes(:,1),runtimes(:,2),ps_fine);

figure('Name','Timing Performance for Ps');
hold on
plot(ps_fine,tms,'LineStyle','-','Color',"#D95319",'LineWidth',2);
hold off
saveas(gcf,'timing_performance_ps.fig');
