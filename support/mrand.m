
function x = mrand(n,cp)

% cp = cumsum( p );
x = sum(bsxfun(@gt,rand(n,1)*cp(end),cp),2)+1;
x = sparse(x,1,1,numel(cp),1);

end