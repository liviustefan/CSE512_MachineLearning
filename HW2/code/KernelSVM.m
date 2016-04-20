function [alpha,b,objective,exitflag]=KernelSVM(K,Y,C)
%% Quodratic Programming for Dual Kernel SVM
% K: nxn matrix; Kernel, linear or non-linear;
% Y: nx1 vector; Label, each entry is 1 or -1;
% C: scalar; larger C means more penalty on classification error;
%
% author:   Yang Wang (yangwangx@gmail.com)
% created:  Feb 25, 2016
% modified: Feb 29, 2016

n=size(Y,1);
H=diag(Y)*K*diag(Y);
f=-ones(n,1);
A=[];b=[];
Aeq=Y';beq=0;
LB=zeros(n,1);UB=C*ones(n,1);
opts.Display='off';
[alpha,objective,exitflag]=quadprog(H,f,A,b,Aeq,beq,LB,UB,[],opts);
objective=-objective; %original dual SVM problem is maximizaiton
[~,freesvs]=max(min(alpha,C-alpha));
b=Y(freesvs)-K(freesvs,:)*diag(Y)*alpha;
end
