run ../../vlfeat/toolbox/vl_setup.m
addpath(genpath('../hw2data/'));
%% hard negative mining on trainval
% initial training
[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();
PosD=[trD(:,trLb==1),valD(:,valLb==1)];
NegD=[trD(:,trLb==-1),valD(:,valLb==-1)];
nPos=size(PosD,2);
nNeg=size(NegD,2);
X=[PosD,NegD];
Y=[ones(nPos,1);-ones(nNeg,1)];
C=1;
[w,b,alpha]=KernelSVM_wrap(X,Y,C);
maxIter=10;
for iter=1:maxIter
    iter
    % detection on training images
    HW2_Utils.genRsltFile(w,b,'trainval','trainvalRslt.mat');
    % drop non support vectors if improved
    ap = HW2_Utils.cmpAP(sprintf('trainvalRslt.mat'), 'trainval')
    A=NegD(:,alpha(end-nNeg+1:end)>1e-5);

    load('trainvalRslt.mat','rects','feats');
    load('trainvalAnno.mat','ubAnno');
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
    hardNegInx=inx(neginx(1:min(end,2000))); %2000
    feats=cat(2,feats{:});
    B=HW2_Utils.l2Norm(feats(:,hardNegInx));
    % update the negative data
    NegD=[A,B];
    % retrain the svm
    nPos=size(PosD,2);
    nNeg=size(NegD,2);
    X=double([PosD,NegD]);
    Y=[ones(nPos,1);-ones(nNeg,1)];
    [w,b,alpha]=KernelSVM_wrap(X,Y,C);
    save(sprintf('trainval_wb_%d.mat',iter),'w','b');
end
HW2_Utils.genRsltFile(w,b,'trainval','trainvalRslt.mat');
final_ap = HW2_Utils.cmpAP(sprintf('trainvalRslt.mat'), 'trainval')

%% test on test data
HW2_Utils.genRsltFile(w,b,'test','110014939.mat');
load('110014939.mat');
save('110014939.mat','rects');