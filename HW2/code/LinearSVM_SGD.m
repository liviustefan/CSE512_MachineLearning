function [w,b,obj]=LinearSVM_SGD(X,Y,C,opts)
d=size(X,1);
n=size(X,2);
if ~exist('opts','var') opts=[]; end
if ~isfield(opts,'lrParams') opts.lrParams=[1,100]; end
if ~isfield(opts,'maxEpoch') opts.maxEpoch=1000; end
if ~isfield(opts,'plot') opts.plot=false; end
if ~isfield(opts,'w') opts.w=zeros(d,1); end
if ~isfield(opts,'b') opts.b=0; end
w=opts.w;
b=opts.b;
obj=0.5*w'*w+C*sum(max(1-Y.*(X'*w+b),0));
for epoch=1:opts.maxEpoch
    lr=opts.lrParams(1)/(opts.lrParams(2)+epoch);
    shuffle=randperm(n);
    for j=shuffle
        if Y(j)*(w'*X(:,j)+b)>=1
            dLdw=(1/n)*w;
            dLdb=0;
        else
            dLdw=(1/n)*w-C*Y(j)*X(:,j);
            dLdb=-C*Y(j);
        end
        w=w-lr*dLdw;
        b=b-lr*dLdb;
    end
    obj=[obj,0.5*w'*w+C*sum(max(1-Y.*(X'*w+b),0))];
end
if opts.plot
    plot(0:1:opts.maxEpoch,obj,'.-');
    title('Objective Value vs. Epoch');
    xlabel('Epoch');ylabel('Obj');
end
end