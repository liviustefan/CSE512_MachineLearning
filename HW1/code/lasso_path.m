function [lambdas, ws, bs]=lasso_path(trX,trLb,tol,nStep)
%% This function solves multiple lasso problems on a regularization path
opts.tol=tol;
opts.lambda=2*max(abs(trX*(trLb-mean(trLb))));
opts.w=zeros(size(trX,1),1);
opts.b=0;
lambdas={};ws={};bs={};
for step=1:nStep
    time=tic;
    [w,b]=lasso(trX,trLb,opts);
    lambdas{step}=opts.lambda; ws{step}=w; bs{step}=b;
    opts.lambda=opts.lambda/2;
    opts.w=w;
    opts.b=b;
    fprintf('[step %03d]: %7.2f seconds \n',step,toc(time));
end
end