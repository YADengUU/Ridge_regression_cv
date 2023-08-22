% test model fitting

rng(123);

% performance of ridge vs OLS
%  compare by proportion of collinear coefficients and sample size

mse_propn=zeros(200,4); %[n,proportion,mse_ridge,mse_ols]
rsq_propn=zeros(200,4); %[n,proportion,r_sq_ridge,r_sq_ols]
lambdas_propn=zeros(200,3);%[n,proportion,best_lambda]

counter=0;
for ns = (500:500:5000)
    n=ns;
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
    
    % generate coefficient values
    low_bound=-20;
    high_bound=20;
    beta = low_bound + (high_bound-low_bound)*randn(num_params+1,1);
    
    % change X so that some columns become collinear with the others
    %  0% ~ 95% collinear, grid by 5%
    % generate random error terms
    epsilons = 0.5 * randn(n,1);
    
    % when 0% of the data are collinear
    y = [ones(n,1),X]*beta+epsilons; 
    lambdas = exp(-9:5);
    [b,b_ols,~,mse_ridge,mse_ols,r_sq,r_sq_ols,best_lambda] = ridge_regression(X,y,lambdas);
    mse_propn(1+counter*20,:)=[n,0,mse_ridge,mse_ols];
    rsq_propn(1+counter*20,:)=[n,0,r_sq_ridge,r_sq_ols];
    lambdas_propn(1,:)=[n,0,best_lambda];
    
    idx=2; % for counting the rows of mse_propn
    cols=(1:num_params);
    for i = (0.05:0.05:0.95)
        
        % for each proportion, do the test 10 times
        mses_ridge=zeros(10,1);
        mses_ols=zeros(10,1);
        r_sqs_ridge=zeros(10,1);
        r_sqs_ols=zeros(10,1);
        lambda_seq=zeros(10,1);
        for j = 1:10
            X_corr=X;
    
            num_collns=floor(num_params*i);    
            collns=randperm(num_params,num_collns);
            others=setdiff(cols,collns);
    
            for k = collns
                ind_cols_to_corr=randi(length(others));
                cols_to_corr=others(ind_cols_to_corr);
    
                X_corr(:,k)=(low_bound+(high_bound-low_bound)*randn(1,1))*X(:,cols_to_corr);
            end
    
            % observations
            y = [ones(n,1),X_corr]*beta+epsilons; 
    
    
            [b,b_ols,~,mse_ridge,mse_ols,r_sq,r_sq_ols,best_lambda] = ridge_regression(X_corr,y,lambdas);
            mses_ridge(j)=mse_ridge;
            mses_ols(j)=mse_ols;
            r_sqs_ridge(j)=r_sq_ridge;
            r_sqs_ols(j)=r_sq_ols;
            lambda_seq(j)=best_lambda;
    
        end
    
        mse_propn(idx+counter*20,:)=[n,i,mean(mses_ridge),mean(mses_ols)];
        rsq_propn(idx+counter*20,:)=[n,i,mean(r_sqs_ridge),mean(r_sqs_ols)];
        lambdas_propn(idx+counter*20,:)=[n,i,mean(lambda_seq)];
        idx=idx+1;
    
    end
    counter=counter+1;

end

% plot the heatmap 
hm_x=(500:500:5000); % 10
hm_y=(0:0.05:0.95); % 20
hm_z=zeros(length(hm_y),length(hm_x));
hm_z_ols=zeros(length(hm_y),length(hm_x));
for i = 1:length(hm_y)
    for j = 1:length(hm_x)
        hm_z(i,j)=mse_propn(20*(j-1)+i,3);
        hm_z_ols(i,j)=mse_propn(20*(j-1)+i,4);
    end
end

figure('Name','Compare Ridge and OLS');
min_val=min(min(hm_z(:)),min(hm_z_ols(:)));
max_val=max(max(hm_z(:)),max(hm_z_ols(:)));
subplot(1,2,1);
imagesc(hm_x,hm_y,(hm_z),[min_val,max_val]);
subplot(1,2,2);
imagesc(hm_x,hm_y,(hm_z_ols),[min_val,max_val]);
saveas(gcf,'Compare Ridge and OLS.tiff');

% compare by number of coefficients and sample size
mse_numcoef=zeros(190,4); %[n,numcoef,mse_ridge,mse_ols]

low_x_bound=-50;
high_x_bound=50;
counter=0;
for ns = (500:500:5000)
    n=ns;
    idx=1;
    for num_params = (199:100:2000)
        
        % generate coefficient values
        low_bound=-20;
        high_bound=20;
        beta = low_bound + (high_bound-low_bound)*randn(num_params+1,1);
        epsilons = 0.5 * randn(n,1);
        
        % for each num_params, do the test 10 times
        mses_ridge=zeros(10,1);
        mses_ols=zeros(10,1);

        for i = 1:10
            X=zeros(n,num_params);
            range=(high_x_bound-low_x_bound)*randn(2,1)+low_x_bound;
            lo=min(range);
            hi=max(range);
            dist_type=randi([1,3]);
            if dist_type==1
                X(:,num_params)=lo+(hi-lo)*randn(n,1);
            elseif dist_type==2
                X(:,num_params)=lo+(hi-lo)*rand(n,1);
            else
                X(:,num_params)=lo+(hi-lo)*exprnd(n,1);
            end

            X_corr=X; 
            num_collns=floor(num_params*0.3);    
            collns=randperm(num_params,num_collns);
            others=setdiff(cols,collns);

            for k = collns
                ind_cols_to_corr=randi(length(others));
                cols_to_corr=others(ind_cols_to_corr);
                
                X_corr(:,k)=(low_bound+(high_bound-low_bound)*randn(1,1))*X(:,cols_to_corr);
            end

            % observations
            y = [ones(n,1),X_corr]*beta+epsilons; 
    
    
            [b,~,~,mse_ridge,mse_ols,~,~,~] = ridge_regression(X_corr,y,lambdas);
            mses_ridge(i)=mse_ridge;
            mses_ols(i)=mse_ols;

        end
        mse_numcoef(idx+counter*19,:)=[n,num_params,mean(mses_ridge),mean(mses_ols)];
        idx=idx+1;       

    end
    counter=counter+1;

end


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
    [b,~,~,mse,mse_ols,~,~,~] = ridge_regression(X_corr,y,lambdas(i));
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

