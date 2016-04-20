load ../hw2data/q3_1_data.mat
%% training
C=100;
opts.lrParams=[1,100];
opts.maxEpoch=1000;
opts.plot=true;
[w,b,obj]=LinearSVM_SGD(trD,trLb,C,opts);

%% validation 
trScore=trD'*w+b;
trAcc=sum((trScore>=0)==(trLb>0))/length(trLb);
valScore=valD'*w+b;
thresh=0;%median(valScore);
tp=sum((valScore>=thresh)&(valLb>0));
tn=sum((valScore<thresh)&(valLb<0));
fp=sum((valScore>=thresh)&(valLb<0));
fn=sum((valScore<thresh)&(valLb>0));
accuracy=(tp+tn)/length(valLb);
fprintf('When C is %.3f:\n',C);
fprintf('The training error is %.2f%%.\n', (1-trAcc)*100);
fprintf('The test error is %.2f%%.\n', (1-accuracy)*100);
fprintf('The |w| is %.2f.\n', sqrt(w'*w));
fprintf('The objective value is %.2f.\n', obj(end));
%fprintf('The number of support vectors is %.2f.\n', sum(alpha>1e-4));
fprintf('The confusion matrix is:\n');
fprintf('|-------|\n|%3d|%3d|\n|%3d|%3d|\n|-------|\n',tp,fp,fn,tn);