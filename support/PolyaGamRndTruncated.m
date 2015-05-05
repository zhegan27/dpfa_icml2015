function x = PolyaGamRndTruncated(a,c,KK,IsBiased)
% Generating Polya-Gamma random varaibles using approximation method

if nargin<4
    IsBiased = false;
end;
x = 1/2/pi^2*sum(gamrnd(a*ones(1,KK),1)./bsxfun(@plus,((1:KK)-0.5).^2,c.^2/4/pi^2),2);
if ~IsBiased
    temp = max(abs(c/2),realmin);
    xmeanfull = (tanh(temp)./(temp)/4);    
    xmeantruncate = 1/2/pi^2*sum(1./bsxfun(@plus,((1:KK)-0.5).^2,c.^2/4/pi^2),2);
    x = x.*xmeanfull./(xmeantruncate);
end