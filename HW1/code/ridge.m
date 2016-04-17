function [w,b,obj,cvErrs] = ridge(X,Y,opts)
%% This function is an efficient ridge regression solver
% * X is a dxn dense/sparse matrix; each column is a sample input;
% * Y is a nx1 vector; each entry is a sample output;
% * opts is the options for ridge solver
%    * opts.lambda controls the regularization
%
% Author: Yang Wang (yangwangx@gmail.com)
% Created: Feb 19, 2016
% Last Modified: Feb 19, 2016

%% parse data size and options
d=size(X,1);
n=size(X,2);
if ~isfield(opts,'lambda') opts.lambda=1; end

%% compute w, b, obj, cvErrs
X1=X; X1(d+1,:)=1;
I0 = eye(d+1); I0(d+1,d+1) = 0;
C = X1*X1' + opts.lambda*I0;
d=X1*Y;
wb = C\d;
w = wb(1:end-1);
b = wb(end);

pd=X1'*wb;
r = pd-Y;
obj = r'*r+opts.lambda*wb'*wb;
cvErrs = r./(1-sum(X1.*(C\X1),1)');
end





