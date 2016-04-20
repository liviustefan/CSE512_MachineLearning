function [trK, tstK] = cmpExpX2Kernel(trD, tstD, gamma)

nTrain=size(trD,2);
nTest=size(tstD,2);

trK=zeros(nTrain,nTrain);
for i=1:nTrain
    for j=(i+1):nTrain
        trK(i,j)=sum((trD(:,i)-trD(:,j)).^2./(trD(:,i)+trD(:,j)+eps));
    end
end
if ~exist('gamma','var') gamma=mean(trK(trK~=0)); end
trK=trK+trK';
trK=exp(-trK/gamma);

tstK=zeros(nTrain,nTest);
for i=1:nTrain
    for j=1:nTest
        tstK(i,j)=sum((trD(:,i)-tstD(:,j)).^2./(trD(:,i)+tstD(:,j)+eps));
    end
end
tstK=exp(-tstK/gamma);
end