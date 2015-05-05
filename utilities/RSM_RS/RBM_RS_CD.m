
% Written by David Carlson, (david.carlson@duke.edu), Duke ECE

function [v,h,ph1]=RBM_RS_CD(params,v,sweeps)
W=params.W;
b=params.b;
c=params.c;
[M,J]=size(W);
N=size(v,2);
if nargin<3
    sweeps=1;
end
D=full(sum(v));D=D(:)';
bD=bsxfun(@times,b,D);
ph1=sigmoid(W'*v+bD);
h=double(rand(J,N)<ph1);
for iter=1:sweeps
    pv=softmax(bsxfun(@plus,W*h,c));
    v=mnrnd(D',pv')';
    ph=sigmoid(W'*v+bD);
    h=double(rand(J,N)<ph);
end
