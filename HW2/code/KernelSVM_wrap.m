function [w,b,alpha,obj]=KernelSVM_wrap(X,Y,C)
K=X'*X;
[alpha,b,obj]=KernelSVM(K,Y,C);
w=X*diag(Y)*alpha;
end