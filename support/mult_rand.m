function [x_pk,x_kn] = mult_rand(X,Phi,Theta)

P = size(X,1); [K,N] = size(Theta);
x_pk = zeros(P,K); x_kn = zeros(K,N);

for n=1:N
	inz = find(X(:,n))';
	map = bsxfun(@times,Phi(inz,:),Theta(:,n)'); % P x K
	map = cumsum(map,2);
	x_kp = zeros(K,numel(inz));
	for m=1:numel(inz)
		x_kp(:,m) = x_kp(:,m) + mrand(X(inz(m),n),map(m,:));
	end
	
	x_kn(:,n) = sum(x_kp,2);
	x_pk(inz,:) = x_pk(inz,:)+x_kp';
end

end