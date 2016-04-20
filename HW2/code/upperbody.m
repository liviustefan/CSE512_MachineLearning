run ../../vlfeat/toolbox/vl_setup.m
addpath(genpath('../hw2data/'));

%% Random Negative Data
%{
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
[w,b,alpha]=KernelSVM_wrap(trD,trLb,1);
HW2_Utils.genRsltFile(w, b, 'val', 'valRslt.mat');
ap=HW2_Utils.cmpAP('valRslt.mat','val');
% ap --> 68.08
%}

%% Hard Negative Mining
TrObj=[];
ValAP=[];
% initial training
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
PosD=[trD(:,trLb==1)];
NegD=[trD(:,trLb==-1)];
nPos=size(PosD,2);
nNeg=size(NegD,2);
X=[PosD,NegD];
Y=[ones(nPos,1);-ones(nNeg,1)];
C=1;
[w,b,alpha,obj]=KernelSVM_wrap(X,Y,C);
TrObj(end+1)=obj;
HW2_Utils.genRsltFile(w, b, 'val', 'valRslt.mat');
ap=HW2_Utils.cmpAP('valRslt.mat','val');
ValAP(end+1)=ap;
maxIter=10;
for iter=1:maxIter
    iter
    % detection on training images
    HW2_Utils.genRsltFile(w,b,'train','trainRslt.mat');
    ap = HW2_Utils.cmpAP(sprintf('trainRslt.mat'), 'train');
    % drop non support vectors if improved
    A=NegD(:,alpha(end-nNeg+1:end)>1e-5);
    load('trainRslt.mat','rects','feats');
    load('trainAnno.mat','ubAnno');
    nIm = length(ubAnno);
    [detScores, isTruePos] = deal(cell(1, nIm));
    for i=1:nIm
        rects_i = rects{i};
        detScores{i} = rects_i(5,:);
        ubs_i = ubAnno{i}; % annotated upper body
        isTruePos_i = -ones(1, size(rects_i, 2));
        for j=1:size(ubs_i,2)
            ub = ubs_i(:,j);
            overlap = HW2_Utils.rectOverlap(rects_i, ub);
            isTruePos_i(overlap >= 0.3) = 1;
        end;
        isTruePos{i} = isTruePos_i;
    end
    detScores = cat(2, detScores{:});
    isTruePos = cat(2, isTruePos{:});
    [~,inx]=sort(detScores,'descend');
    neginx=find(isTruePos(inx)==-1);
    hardNegInx=inx(neginx(1:min(end,1000))); %1000
    feats=cat(2,feats{:});
    B=HW2_Utils.l2Norm(feats(:,hardNegInx));
    % update the negative data
    NegD=[A,B];
    % retrain the svm
    nPos=size(PosD,2);
    nNeg=size(NegD,2);
    X=double([PosD,NegD]);
    Y=[ones(nPos,1);-ones(nNeg,1)];
    [w,b,alpha,obj]=KernelSVM_wrap(X,Y,C);
    TrObj(end+1)=obj
    HW2_Utils.genRsltFile(w, b, 'val', 'valRslt.mat');
    ap=HW2_Utils.cmpAP('valRslt.mat','val');
    ValAP(end+1)=ap
end
save('upperbody_train.mat','TrObj','ValAP','w','b');
%% 10 fold cross validation on training data
%{
nTrain=size(trD,2);
kfold=10;
inxss=ml_kFoldCV_Idxs(nTrain,kfold);
Cs=10.^((-3:9)/3);
mAs=[];
for C=Cs
    As=[];
    for i=1:kfold
        inx1=cat(2,inxss{setdiff(1:kfold,i)});
        inx2=inxss{i};
        [w,b]=KernelSVM_wrap(trD(:,inx1),trLb(inx1),C);
        score=trD(:,inx2)'*w+b;
        label=trLb(inx2);
        As(end+1,1)=mean((score>=0)==(label>0));
    end
    mAs(end+1,1)=mean(As);
end
%% detection
C=1;
[w,b]=KernelSVM_wrap(trD,trLb,C);
HW2_Utils.genRsltFile(w,b,'val',sprintf('valRslt_%.2f.mat',C));
ap = HW2_Utils.cmpAP(sprintf('valRslt_%.2f.mat',C), 'val');
%}