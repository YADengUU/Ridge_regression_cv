% time the regression performance

rng(123);

runtimes=zeros(30,4); %[sample size,num param,time_no_para,time_with_para]

low_x_bound=-50;
high_x_bound=50;

counter=0;

for ns = (500:500:5000)
    n=ns;
    idx=1;
    for num_params = [199,999,1999]
        cols=(1:num_params);
        % generate coefficient values
        low_bound=-20;
        high_bound=20;
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
    
        lambdas = exp(-9:5);

        % function handles for timing
        g = @()ridge_regression(X_corr,y,lambdas);
        h = @()ridge_parallel(X_corr,y,lambdas);

        tm=timeit(g);
        tm_par=timeit(h);
        
        runtimes(idx+counter*3,:)=[n,num_params,tm,tm_par];
        idx=idx+1;
    end
    counter=counter+1;
end

% plot runtime

ssz_fine=(500:10:5000);

tms199_par=spline((500:500:5000),runtimes((1:3:30),4),ssz_fine);
tms199_nopar=spline((500:500:5000),runtimes((1:3:30),3),ssz_fine);
tms999_par=spline((500:500:5000),runtimes((2:3:30),4),ssz_fine);
tms999_nopar=spline((500:500:5000),runtimes((2:3:30),3),ssz_fine);
tms1999_par=spline((500:500:5000),runtimes((3:3:30),4),ssz_fine);
tms1999_nopar=spline((500:500:5000),runtimes((3:3:30),3),ssz_fine);

figure('Name','Timing Performance');
hold on
plot(ssz_fine,tms199_nopar,'LineStyle','-','Color',"#0041C2",'LineWidth',2);
plot(ssz_fine,tms199_par,'LineStyle','--','Color',"#007ec4",'LineWidth',2);
plot(ssz_fine,tms999_nopar,'LineStyle','-','Color',"#ec4976",'LineWidth',2);
plot(ssz_fine,tms999_par,'LineStyle','--','Color',"#a1409b",'LineWidth',2);
plot(ssz_fine,tms1999_nopar,'LineStyle','-','Color',"#3EA99F",'LineWidth',2);
plot(ssz_fine,tms1999_par,'LineStyle','--','Color',"#587a71",'LineWidth',2);
hold off
saveas(gcf,'timing_performance.fig');