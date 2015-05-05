
% Written by David Carlson, (david.carlson@duke.edu), Duke ECE

function [mllkho,mllkin]=RBM_RS_Test(params,v,vho,opts)
collect=opts.test.collect;
[~,~,ph]=RBM_RS_CD(params,v,0);
%% collection
pvh=zeros(size(v));
for i=1:collect
    h=double(rand(size(ph))<ph);
    pvhs=softmax(bsxfun(@plus,params.W*h,params.c));
    pvh=pvh+pvhs;
end
pvh=pvh./collect;
%% calculation of llk
Din=sum(v(:));
lpvh=log(pvh);
mllkin=sum(dot(v,lpvh))./Din;
Dho=sum(vho(:));
mllkho=sum(dot(vho,lpvh))./Dho;