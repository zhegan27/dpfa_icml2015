function s=logsumexp(x)

s=bsxfun(@plus,max(x),log(sum(exp(bsxfun(@minus,x,max(x))))));