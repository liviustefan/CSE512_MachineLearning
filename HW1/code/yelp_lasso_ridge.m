load yelp_l2norm_result.mat
FIX=find(trval.w~=0);

%% load yelp training data and validation data
D=load('../hw1data/trainData.txt');
trX=sparse(D(:,2),D(:,1),D(:,3));
trLb=load('../hw1data/trainLabels.txt');
D=load('../hw1data/valData.txt');
valX=sparse(D(:,2),D(:,1),D(:,3));
valLb=load('../hw1data/valLabels.txt');

%% L2-normalize
trX=trX(FIX,:);
trX=trX./repmat(sqrt(sum(trX.^2,1)),[length(FIX),1]);
trX(isnan(trX))=0;
valX=valX(FIX,:);
valX=valX./repmat(sqrt(sum(valX.^2,1)),[length(FIX),1]);
valX(isnan(valX))=0;

%% solve multiple ridge problems on a regularization path
las=2:0.1:3.5;
cvRMSE=[];
ws={};bs={};
count=0;
for lambda=las
    count=count+1;
    opts=[];
    opts.lambda=lambda;
    [w,b,obj,cvErrs] = ridge([trX,valX],[trLb;valLb],opts);
    cvRMSE(count)=sqrt(mean(cvErrs.^2));
    ws{count}=w;
    bs{count}=b;
end
save('yelp_lass_ridge_result.mat','las','ws','bs','cvRMSE');
plot(las,cvRMSE,'*-');
[~,inx]=min(cvRMSE);
bestla=las(inx);
bestw=ws{inx};
bestb=bs{inx};

%% list the top 10 positive/negative features
fh=fopen('../hw1data/featureTypes.txt');
fname=textscan(fh,'%s','delimiter', '\n');
fname=fname{1}(FIX);
fclose(fh);
[~,inx]=sort(bestw,'descend');
fname(inx(1:10))
fname(inx(end-9:end))

%% predict on yelp test data
D=load('../hw1data/testData.txt');
tstX=sparse(D(:,2),D(:,1),D(:,3));
tstX=tstX(FIX,:);
tstX=tstX./repmat(sqrt(sum(tstX.^2,1)),[length(FIX),1]);
tstX(isnan(tstX))=0;
tstLb=tstX'*bestw+bestb;
tstLb(tstLb<1)=1;
tstLb(tstLb>5)=5;

%% write predicition into a csv file
csvwrite('predTestLabels.csv',[(1:25000)',tstLb]);