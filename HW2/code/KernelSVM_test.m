load ../hw2data/q3_1_data.mat
%% training
C=100;
[w,b,alpha,obj]=KernelSVM_wrap(trD,trLb,C);
%% validation 
valScore=valD'*w+b;
thresh=0;
tp=sum((valScore>=thresh)&(valLb>0));
tn=sum((valScore<thresh)&(valLb<0));
fp=sum((valScore>=thresh)&(valLb<0));
fn=sum((valScore<thresh)&(valLb>0));
accuracy=(tp+tn)/length(valLb);
fprintf('When C is %.3f:\n',C);
fprintf('The classification accuracy is %.2f%%.\n', accuracy*100);
fprintf('The objective value is %.2f.\n', obj);
fprintf('The number of support vectors is %.2f.\n', sum(alpha>1e-4));
fprintf('The confusion matrix is:\n');
fprintf('|-------|\n|%3d|%3d|\n|%3d|%3d|\n|-------|\n',tp,fp,fn,tn);