function s=softmax(x)

s=bsxfun(@minus,x,max(x));
s=exp(s);
s=bsxfun(@rdivide,s,sum(s));