function [Lsum,L] = CRT(x,r)

[xx,~,jj] = unique(x);
L = zeros(size(x));
Lsum = 0;
if ~isempty(x)
	for i=1:numel(xx)
		y = xx(i);
		if y > 0
			L(jj==i) = sum(bsxfun(@le,rand(nnz(jj==i),y),r./(r+(1:y)-1)),2);
		end
	end
	Lsum = sum(L);
end

end