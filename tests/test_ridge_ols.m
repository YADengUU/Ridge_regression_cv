
% test by comparing ridge vs OLS
%  1. based on proportion of collinear coefficients and sample size

mse_propn=zeros(200,4); %[n,proportion,mse_ridge,mse_ols]
rsq_propn=zeros(200,4); %[n,proportion,r_sq_ridge,r_sq_ols]

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
    [~,~,mse_ridge,r_sq_ridge] = ridge_regression(X,y);
    [~,mse_ols,r_sq_ols] = ols_estimator(X,y,0.8);
    mse_propn(1+counter*20,:)=[n,0,mse_ridge,mse_ols];
    rsq_propn(1+counter*20,:)=[n,0,r_sq_ridge,r_sq_ols];
    
    idx=2; % for counting the rows of mse_propn
    cols=(1:num_params);
    for i = (0.05:0.05:0.99)
        
        % for each proportion, do the test 10 times
        mses_ridge=zeros(10,1);
        mses_ols=zeros(10,1);
        r_sqs_ridge=zeros(10,1);
        r_sqs_ols=zeros(10,1);
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
    
            [~,~,mse_ridge,r_sq_ridge] = ridge_regression(X_corr,y);
            [~,mse_ols,r_sq_ols] = ols_estimator(X_corr,y,0.8);
            
            mses_ridge(j)=mse_ridge;
            mses_ols(j)=mse_ols;
            r_sqs_ridge(j)=r_sq_ridge;
            r_sqs_ols(j)=r_sq_ols;
    
        end
    
        mse_propn(idx+counter*20,:)=[n,i,mean(mses_ridge),mean(mses_ols)];
        rsq_propn(idx+counter*20,:)=[n,i,mean(r_sqs_ridge),mean(r_sqs_ols)];
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

% 2. by number of coefficients and sample size
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
    
    
            [~,~,mse_ridge,~] = ridge_regression(X_corr,y);
            [~,mse_ols,~] = ols_estimator(X_corr,y,0.8);
            mses_ridge(i)=mse_ridge;
            mses_ols(i)=mse_ols;

        end
        mse_numcoef(idx+counter*19,:)=[n,num_params,mean(mses_ridge),mean(mses_ols)];
        idx=idx+1;       

    end
    counter=counter+1;

end
