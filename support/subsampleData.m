function [vSS,ndx]=subsampleData(v,Nss)
[M,N]=size(v);
if nargin<2;Nss=min(1e2,N);end
ndx=datasample(1:N,Nss,'replace',false);
vSS=v(:,ndx);