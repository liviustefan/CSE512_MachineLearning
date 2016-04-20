%% configure LibSVM
addpath('/Users/yangwang/Desktop/CSE512_Machine_Learning/libsvm/matlab/');

%% prepare train/test data
classes={'Living Room','Kitchen','Hallway', 'Penny''s Living Room',...
         'Cafeteria','Cheesecake Factory','Laundry Room','Comic BookStore'};
Tr=load('../bigbangtheory/train.mat');
Tst=load('../bigbangtheory/test.mat');
imPath=@(id)sprintf('../bigbangtheory/%06d.jpg',id);

%% compute bag-of-words
%HW3_BoW.main()
load('bows.mat','trD','tstD','trLbs');
nTrain=size(trD,2);
nTest=size(tstD,2);

%% 5 Fold split
nFold=5;
INXs=cell(nFold,1);
for i=1:8
    inx_i=find(trLbs==i);
    nSample=length(inx_i);
    order=1:nSample;%randperm(nSample);
    for j=1:nFold
       INXs{j}=[INXs{j};inx_i(order(((j-1)*ceil(nSample/nFold)+1):min(end,j*ceil(nSample/nFold))))];
    end
end

%% SVM with RBF kernel
cv_acc=[];
Cs=[1000];
Gs=[10];
for iC=1:length(Cs)
    for iG=1:length(Gs)
        pred={};acc={};dec={};
        for i=1:nFold
            trinx=unique(cat(1,INXs{setdiff(1:nFold,i)}));
            vlinx=unique(cat(1,INXs{i}));
            model = svmtrain(trLbs(trinx), trD(:,trinx)', sprintf('-s 0 -t 2 -g %f -c %f -q',Gs(iG),Cs(iG)));
            [pred{i}, acc{i}, dec{i}] = svmpredict(trLbs(vlinx), trD(:,vlinx)', model);
        end
        cv_acc(iC,iG)=sum(cellfun(@(x)x(1),acc).*cellfun(@(x)length(x),pred))/nTrain;
    end
end
cv_acc


%% SVM with exponential X2 kernel
cv_acc=[];
Cs=[1000];
Gs=[5];
for iC=1:length(Cs)
    for iG=1:length(Gs)
        % compute exponential K-square kernel
        [trK, tstK] = cmpExpX2Kernel(trD, tstD, Gs(iG));
        pred={};acc={};dec={};
        for i=1:nFold
            trinx=unique(cat(1,INXs{setdiff(1:nFold,i)}));
            vlinx=unique(cat(1,INXs{i}));
            model = svmtrain(trLbs(trinx), [(1:length(trinx))',trK(trinx,trinx)'], sprintf('-q -s 0 -t 4 -c %f',Cs(iC)));
            [pred{i}, acc{i}, dec{i}] = svmpredict(trLbs(vlinx), [(1:length(vlinx))',trK(trinx,vlinx)'], model);
        end
        cv_acc(iC,iG)=sum(cellfun(@(x)x(1),acc).*cellfun(@(x)length(x),pred))/nTrain;
    end        
end
cv_acc


%% test score
C=1000;
model = svmtrain(trLbs, [(1:nTrain)',trK'], sprintf('-q -s 0 -t 4 -c %f',C));
[pred, acc, dec] = svmpredict(ones(1,nTest)', [(1:nTest)',tstK'], model);
%% write predicition into a csv file
csvwrite('predTestLabels.csv',[Tst.imIds,pred]);