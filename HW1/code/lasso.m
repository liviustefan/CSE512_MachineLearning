function [w,b,obj] = lasso(X,Y,opts)
%% This function is an efficient lasso regression solver
% * X is a dxn dense/sparse matrix; each column is a sample input;
% * Y is a nx1 vector; each entry is a sample output;
% * opts is the options for lasso solver
%    * opts.lambda controls the sparsity of w
%    * opts.tol sets the stopping criteria
%    * opts.w & opts.b initialize w & b
%
% Author: Yang Wang (yangwangx@gmail.com)
% Last Modified: Feb 14, 2016

%% parse data size and options
d=size(X,1);
n=size(X,2);
if ~isfield(opts,'lambda') opts.lambda=1; end
if ~isfield(opts,'tol') opts.tol=5e-3; end
if ~isfield(opts,'w') opts.w=zeros(d,1); end
if ~isfield(opts,'b') opts.b=0; end

%% initialize w, b, residule and objective
w=opts.w;
b=opts.b;
r=Y-(X'*w+b);
lambda=opts.lambda;
obj=r'*r+lambda*sum(abs(w));
% [I,J,S]=find(X);

%% efficient coordinate descent algorithm
a=2*sum(X.^2,2);
count=0; converge=false;
while ~converge
    count=count+1;
    r=Y-(X'*w+b); % recalc r each iteration to avoid numerical drift
    dr=mean(r);
    b=b+dr; % update b
    r=r-dr; % update r
    for k=1:d
        ck=w(k)*a(k)+2*X(k,:)*r; % calculate ck
        wk=w(k);                 % record old wk
        if ck<-lambda
            w(k)=(ck+lambda)/a(k);
        else
            if ck>lambda
                w(k)=(ck-lambda)/a(k);
            else
                w(k)=0;
            end
        end                     % update w(k)
        r=r-(w(k)-wk)*X(k,:)';  % update r
    end
    % stopping criteria
    obj(end+1)=r'*r+lambda*sum(abs(w));
    change=(obj(end-1)-obj(end))/obj(end-1);
    if change < opts.tol
        converge = true;
    end
end
end