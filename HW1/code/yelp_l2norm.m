%% load yelp training data and validation data
D=load('../hw1data/trainData.txt');
trX=sparse(D(:,2),D(:,1),D(:,3));
trLb=load('../hw1data/trainLabels.txt');
D=load('../hw1data/valData.txt');
valX=sparse(D(:,2),D(:,1),D(:,3));
valLb=load('../hw1data/valLabels.txt');

%% L2-normalize
trX=trX./repmat(sqrt(sum(trX.^2,1)),[2500,1]);
trX(isnan(trX))=0;
valX=valX./repmat(sqrt(sum(valX.^2,1)),[2500,1]);
valX(isnan(valX))=0;
%% solve multiple lasso problems on a regularization path
[las,ws,bs]=lasso_path2(trX,trLb,1e-3,2,10);

%% visualize
ww=cat(2,ws{:});
figure; 
plot(ww(1:end,:)','*-');
title('w vs. \lambda');
xlabel('\lambda');
ax = gca;
ax.XTickLabel = cellfun(@(x)num2str(x),las,'UniformOutput',false);
ax.XTickLabelRotation = 45;

%% record train/validation RMSE
nNonzero=cellfun(@(x)sum(x~=0),ws);
trRMSE=cellfun(@(x,y)sqrt(mean((trX'*x+y-trLb).^2)),ws,bs);
valRMSE=cellfun(@(x,y)sqrt(mean((valX'*x+y-valLb).^2)),ws,bs);
figure;
plot([trRMSE; valRMSE]');
title('train/validation err vs. \lambda');
xlabel('\lambda');
ax = gca;
ax.XTickLabel = cellfun(@(x)num2str(x),las,'UniformOutput',false);
ax.XTickLabelRotation = 45;
save('yelp_l2norm_result.mat','las','ws','bs','nNonzero','trRMSE','valRMSE');

%% train model using training and validation data
load('yelp_l2norm_result.mat');
[~,inx]=min(valRMSE)
opts.lambda=las{inx};
opts.tol=1e-3;
opts.w=ws{inx};
opts.b=bs{inx};
[w,b,obj]=lasso([trX,valX],[trLb;valLb],opts);
trvalLb=[trX,valX]'*w+b;
trvalLb(trvalLb<1)=1;
trvalLb(trvalLb>5)=5;
RMSE=sqrt(mean((trvalLb-[trLb;valLb]).^2));
trval.lambda=opts.lambda;
trval.w=w;
trval.b=b;
trval.obj=obj;
trval.RMSE=RMSE;
save('yelp_l2norm_result.mat','-append','trval');

%% list the top 10 positive/negative features
fh=fopen('../hw1data/featureTypes.txt');
fname=textscan(fh,'%s','delimiter', '\n');
fname=fname{1};
fclose(fh);
[~,finx]=sort(w,'descend');
fname(finx(1:10))
fname(finx(end-9:end))

%% predict on yelp test data
D=load('../hw1data/testData.txt');
tstX=sparse(D(:,2),D(:,1),D(:,3));
tstX=tstX./repmat(sqrt(sum(tstX.^2,1)),[2500,1]);
tstX(isnan(tstX))=0;
tstLb=tstX'*trval.w+trval.b;
tstLb(tstLb<1)=1;
tstLb(tstLb>5)=5;

%% write predicition into a csv file
csvwrite('predTestLabels.csv',[(1:25000)',tstLb]);