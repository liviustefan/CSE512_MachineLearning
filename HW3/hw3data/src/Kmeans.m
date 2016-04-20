function [C, A, iter]=Kmeans(X, k, opts)
% []=Kmeans

% author: Yang Wang
% email:  yangwangx@gmail.com
% created: Mar 17, 2016
% modified: Mar 17, 2016

[d,n]=size(X);
if ~exist('opts','var') opts=[]; end
if ~isfield(opts,'maxIter') opts.maxIter=20; end
if ~isfield(opts,'verbose') opts.verbose=1; end

%% select k starting centers
if isfield(opts,'centerID')
    centerID=opts.centerID;
else
    centerID=randperm(n,k);
end
C=X(:,centerID);

%% initialize assignment
A=nan(n,1);

%% repeat until convergence or iteration limit
for iter=1:opts.maxIter
    % re-assigment
    D=pdist2(X',C','euclidean');
    [~,Anew]=min(D,[],2);
    change=sum(Anew~=A);
    A=Anew;
    if opts.verbose==1
        fprintf('Iter %03d: Assignment Change = %d\n',iter,change);
    end
    if change==0
        break;
    end
    % re-center
    for i=1:k
        inx=find(A==i);
        C(:,i)=mean(X(:,inx),2);        
    end
end
end