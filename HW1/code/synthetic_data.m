%% prepare synthetic data
N=200; d=75; k=10; sigma=1;
X=randn(d,N);
w_=[10*sign(randn(k,1));zeros(d-k,1)];
b_=0;
Y=X'*w_+b_+sigma*randn(N,1);

%% solve multiple lasso problems on a regularization path
[lambdas, ws, bs]=lasso_path(X,Y,1e-3,11);

%% visualize how w changes with respect to lambda
ww=cat(2,ws{:});
figure; 
plot(ww(1:end,:)','*-');
title('w vs. \lambda');
xlabel('\lambda');
ax = gca; 
ax.XTickLabel = cellfun(@(x)num2str(round(x)),lambdas,'UniformOutput',false);
ax.XTickLabelRotation = 45;

%% visualize how precision/recall changes with respect to lambda
precision=cellfun(@(x)sum(sign(x(x~=0))==sign(w_(x~=0)))/sum(x~=0), ws);
recall=cellfun(@(x)sum(sign(x(x~=0))==sign(w_(x~=0)))/k, ws);
figure;
plot([precision;recall]','*-');
legend('precision','recall');
title('precision/recall vs. \lambda');
xlabel('\lambda');
ax = gca; 
ax.XTickLabel = cellfun(@(x)num2str(round(x)),lambdas,'UniformOutput',false);
ax.XTickLabelRotation = 45;


%% solve multiple lasso problems on a regularization path
[lambdas, ws, bs]=lasso_path(X,Y,1e-3,11);

%% visualize how w changes with respect to lambda
ww=cat(2,ws{:});
figure; 
plot(ww(1:end,:)','*-');
title('w vs. \lambda');
xlabel('\lambda');
ax = gca; 
ax.XTickLabel = cellfun(@(x)num2str(round(x)),lambdas,'UniformOutput',false);
ax.XTickLabelRotation = 45;

%% visualize how precision/recall changes with respect to lambda
precision=cellfun(@(x)sum(sign(x(x~=0))==sign(w_(x~=0)))/sum(x~=0), ws);
recall=cellfun(@(x)sum(sign(x(x~=0))==sign(w_(x~=0)))/k, ws);
figure;
plot([precision;recall]','*-');
legend('precision','recall');
title('precision/recall vs. \lambda');
xlabel('\lambda');
ax = gca; 
ax.XTickLabel = cellfun(@(x)num2str(round(x)),lambdas,'UniformOutput',false);
ax.XTickLabelRotation = 45;

