% Deep Poisson Factor Analysis for Topic Modeling
% using Deep Boltzmann Machine
% for 20NEWS dataset
% 
% Written by Changyou Chen (cchangyou@gmail.com)

clear all; clc; close all;
seed = 0; rand('state',seed); randn('state',seed);

load '20news_data';

% Setup
[P,N] = size(wordsTrain); [~,Ntest] = size(wordsTest); wordsTest0 = wordsTest;
Ns = 50;
K = [128, 64, 32];
nlayer = length(K);
Phi = rand(P,K(1)); Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
Phi_ = log(Phi);
Theta = 1/K(1)*ones(K(1), Ns);
ThetaTest = 1/K(1)*ones(K(1), Ntest);

H = cell(nlayer, 1);
for k = 1:nlayer
    H{k} = ones(K(k), Ns);
end

Htest = cell(nlayer, 1);
for k = 1:nlayer
    Htest{k} = ones(K(k), Ntest);
end

b = 0*randn(K(nlayer), 1);
W = cell(1, nlayer-1);
c = cell(1, nlayer-1);
Winc = cell(1, nlayer-1);
Wavg = cell(1, nlayer-1);
cinc = cell(1, nlayer-1);
cavg = cell(1, nlayer-1);
for i = 1:nlayer-1
    W{i} = 0.1 * randn(K(i), K(i+1));
    c{i} = 0 * randn(K(i), 1);
    Winc{i}  = zeros(K(i), K(i+1));
	cinc{i} = zeros(K(i), 1);
    Wavg{i}  = W{i};
    cavg{i} = c{i};
end

binc = zeros(K(nlayer), 1);
bavg = b;

burnin_s = 20;
lag = 2;
ns = 3;
maxiter = burnin_s + (ns - 1) * lag + 1;
Theta_s = cell(1, ns);

H_s = cell(nlayer, 1);
for k = 1:nlayer
    H_s{k} = cell(1, ns);
end
pg_s = cell(1, ns);
xpk_s = cell(ns, 1);
xkn_s = cell(1, ns);

YflagTrain = wordsTrain>0;
YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

% options
burnin = 2000; collection = 1500; step = 1;

% initialization
c0 = 1; 
gamma0 = 1; gamma0_ = log(gamma0);
eta = 0.05;
r_k = 3*ones(K(1),1);
r_k_ = log(r_k);
p_ii = 0.5*ones(1, N); %%% p_i/(1-p_i) = 1 --> p_i=0.5;
p_i_test = 0.5*ones(1, Ntest);
e0 = 1e-2; f0 = 1e-2;

C = 0.005;
eta_s = 4e-07;
eta_w = 0.1;
sigma = 1;
h = sqrt(eta_s);
h_w = sqrt(eta_w);
eta_r = eta_s / 6;
h_r = sqrt(eta_r);
Leapfrog = 50;

momentum = 0.5;

u_phi = h * randn(size(Phi));
u_r = h_r * randn(size(r_k));
u_r0 = h_r * randn;

ALPHA_phi = C * ones(size(u_phi));
ALPHA_r = C * ones(size(u_r));
ALPHA_r0 = C;

gammaTrain = cell(nlayer-1, 1);
gammaTest = cell(nlayer-1, 1);

gradw = cell(nlayer-1, 1);

% collect results
result.loglike=[]; result.loglikeTrain=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTrain = 0; result.PhiThetaTest = 0; result.Count = 0; 
mid.loglike=[]; mid.loglikeTrain=[]; mid.loglikeHeldout=[];
mid.K=[]; mid.PhiThetaTrain = 0; mid.PhiThetaTest = 0; mid.Count = 0;

update_r = burnin + collection;
rand_phi = burnin-100;

dotest = 1;
do_fig = 0;

p = randperm(N);
select_doc = 1;
nsam = 1;
penalty = 0.005;

ph2 = cell(1, nlayer);
negdata = cell(1, nlayer);
negdatastates = cell(1, nlayer);
dW = cell(1, nlayer-1);
dc = cell(1, nlayer-1);

for iter = 1:burnin + collection
    
    if iter < burnin
        dotest = 0;
    else
        dotest = 1;
    end

	if select_doc+Ns-1 > N
		p = randperm(N);
		select_doc = 1;
	end
    idx = p(select_doc:select_doc+Ns-1);
	select_doc = select_doc+Ns;
    Xtrain_s = wordsTrain(:, idx);
    Yflagtrain_s = Xtrain_s > 0;
    p_i = p_ii(idx);

    Theta = 1/K(1)*ones(K(1), Ns);
    
    H{1} = ones(K(1), Ns);
	for l = 2:nlayer
		H{l} = zeros(K(l), Ns);
	end
        
    for ss = 1:maxiter

        % 1. Sample x_pnk
    	[x_pk,x_kn] = mult_rand(Xtrain_s,Phi,Theta);
        
        % go up
        for l = 2:nlayer-1
            ph2{l} = logistic(W{l-1}' * H{l-1} + W{l} * H{l+1} + repmat(c{l}, 1, Ns));
            H{l} = +(ph2{l} > rand(K(l), Ns));
        end
        ph2{nlayer} = logistic(W{nlayer-1}' * H{nlayer-1} + repmat(b, 1, Ns));
        H{nlayer} = +(ph2{nlayer} > rand(K(nlayer), Ns));
        
        % go down
        for l = nlayer-1:-1:2
            ph2{l} = logistic(W{l-1}' * H{l-1} + W{l} * H{l+1} + repmat(c{l}, 1, Ns));
            H{l} = +(ph2{l} > rand(K(l), Ns));
        end
        ph2{1} = logistic(W{1} * ph2{2} + repmat(c{1}, 1, Ns));
        
        % 2. Sample H1
        lix = (x_kn==0);
        [rix,cix] = find(x_kn==0);
        p1 = ph2{1}(lix).*((1-p_i(cix)').^r_k(rix));
        p0 = 1-ph2{1}(lix);
        H{1} = ones(K(1),Ns);
        H{1}(lix) = (p1./(p1+p0))>rand(size(rix));        
        
        Theta = gamrnd(x_kn+(r_k*ones(1,Ns)).*H{1},ones(K(1),1)*p_i); %assert(min(Theta(:)) > 0);
        
        %%% collect samples
        if ss > burnin_s && (lag == 1 || mod(ss-burnin_s, lag)==1)
            ii = floor((ss - burnin_s) / lag);
            if lag > 1
                ii = ii + 1;
            end
            Theta_s{ii} = Theta;
            for nl = 1:nlayer
                H_s{nl}{ii} = H{nl};
            end
            xpk_s{ii} = x_pk;
            xkn_s{ii} = x_kn;
        end
    end
    
	if nlayer < 3
		nthrough = 1;
	else
		nthrough = 1;
	end
    for ss = 1:nthrough
        % go down
        for l = nlayer-1:-1:2
            if ss == 1
                negdata{l} = ph2{l};
            else
                negdata{l} = logistic(W{l-1}' * negdata{l-1} + W{l} * negdata{l+1} + repmat(c{l}, 1, Ns));
            end
            negdatastates{l} = +(negdata{l} > rand(K(l), Ns));
        end
        if ss == 1
            negdata{1} = ph2{1};
            negdata{nlayer} = logistic(W{nlayer-1}' * ph2{nlayer-1} + repmat(b, 1, Ns));
        else
            negdata{1} = logistic(W{1} * negdata{2} + repmat(c{1}, 1, Ns));
            negdata{nlayer} = logistic(W{nlayer-1}' * negdata{nlayer-1} + repmat(b, 1, Ns));
        end
        negdatastates{1} = negdata{1} > rand(K(1), Ns);
    
        % go up
        for l = 2:nlayer-1
            negdata{l} = logistic(W{l-1}' * negdata{l-1} + W{l} * negdata{l+1} + repmat(c{l}, 1, Ns));
            negdatastates{l} = +(negdata{l} > rand(K(l), Ns));
        end
        negdata{nlayer} = logistic(W{nlayer-1}' * negdata{nlayer-1} + repmat(b, 1, Ns));
        negdatastates{nlayer} = +(negdata{nlayer} > rand(K(nlayer), Ns));

    end

    xpk_m = xpk_s{1};
    for k = 2:ns
        xpk_m = xpk_m + xpk_s{k};
    end
    gradPhi = xpk_m - repmat(sum(xpk_m, 1), P, 1) .* Phi;
    gradPhi = N * gradPhi / Ns / ns + (eta - 1 - exp(Phi_));%(eta-1)*(1 - P*Phi);
    
    mat_h1 = cell2mat(H_s{1});
    if iter < update_r
        x_k_s = cell2mat(xkn_s);
        [kk,~,counts] = find(x_k_s);
        ll = zeros(size(counts));
        L_k = zeros(K(1),1);
        for k=1:K(1)
            [L_k(k),ll(kk==k)] = CRT(counts(kk==k),r_k(k));
            L_k(k) = L_k(k) * N / Ns / ns;
        end
        sumbpi = sum(bsxfun(@times,mat_h1,log(max(1-repmat(p_i, 1, ns),realmin))),2) * N / Ns / ns;
        gradr = (gamma0 + L_k) ./ r_k + (sumbpi - c0);

        gradr = gradr .* r_k;

        p_prime_k = -sumbpi./(c0-sumbpi);
        gradr0 = (e0 + CRT(ceil(L_k), gamma0)) / gamma0 + sum(log(max(1-p_prime_k,realmin))) - f0;
        gradr0 = gradr0 * gamma0;
    end

	db = sum(ph2{nlayer}, 2) - sum(negdata{nlayer}, 2);
    if (iter > burnin-100)
        %apply averaging
        bavg = bavg - (1/nsam)*(bavg - b);
    else
        bavg = b;
    end
	binc = momentum*binc + eta_w*(db/Ns);
	for l = 1:nlayer-1
        dW{l} = (H{l}*ph2{l+1}' - negdatastates{l}*negdata{l+1}');
        dc{l} = sum(H{l}, 2) - sum(negdatastates{l}, 2);
        Winc{l} = momentum*Winc{l} + eta_w*(dW{l}/Ns - penalty*W{l});
        cinc{l} = momentum*cinc{l} + eta_w*(dc{l}/Ns);
		W{l} = W{l} + Winc{l};
		c{l} = c{l} + cinc{l};
        if (iter > burnin-100)
            %apply averaging
            Wavg{l} = Wavg{l} - (1/nsam)*(Wavg{l} - W{l});
            cavg{l} = cavg{l} - (1/nsam)*(cavg{l} - c{l});
            nsam = nsam+1;
        else
            Wavg{l} = W{l};
            cavg{l} = c{l};
        end
	end
    
    if iter > rand_phi
        u_phi = h * randn(size(u_phi));
    end
    u_r = h_r * randn(size(u_r));
    u_r0 = h_r * randn;

    for ll = 1:Leapfrog
        
        Phi_ = Phi_ + u_phi; assert(isfinite(sum(Phi_(:))));

        if iter < update_r
            r_k_ = r_k_ + u_r; assert(isfinite(sum(r_k_)));
            gamma0_ = gamma0_ + u_r0;
        end
        
        u_phi = (1 - ALPHA_phi) .* u_phi + gradPhi * eta_s + ...
            sqrt(2 * C * eta_s) * randn(size(u_phi));
        
        if iter < update_r
            u_r = (1 - ALPHA_r) .* u_r + gradr * eta_r + ...
                sqrt(2 * C * eta_r) * randn(size(u_r)); assert(isfinite(sum(u_r(:))));
            
            u_r0 = (1 - ALPHA_r0) .* u_r0 + gradr0 * eta_r + ...
                sqrt(2*C*eta_r) * randn(size(u_r0));
        end
        
        tmp = u_phi .* u_phi;
        ALPHA_phi = ALPHA_phi + tmp - eta_s;
        
        if iter < update_r
            tmp = u_r .* u_r;
            ALPHA_r = ALPHA_r + tmp - eta_r;
            
            tmp = u_r0 .* u_r0;
            ALPHA_r0 = ALPHA_r0 + tmp - eta_r;
        end
        
    end
    
    Phi = bsxfun(@minus, Phi_, max(Phi_));
    Phi = exp(Phi);
    Phi = bsxfun(@rdivide, Phi, sum(Phi, 1));
    r_k = exp(r_k_);
    gamma0 = exp(gamma0_);
    
    if dotest == 1

        [~, x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);
    
        % go up
        for l = 2:nlayer-1
            ph = logistic(Wavg{l-1}' * Htest{l-1} + Wavg{l} * Htest{l+1} + repmat(cavg{l}, 1, Ntest));
            Htest{l} = +(ph > rand(K(l), Ntest));
        end
        ph = logistic(Wavg{nlayer-1}' * Htest{nlayer-1} + repmat(bavg, 1, Ntest));
        Htest{nlayer} = +(ph > rand(K(nlayer), Ntest));

        % go down
        for l = nlayer-1:-1:2
        	ph = logistic(Wavg{l-1}' * Htest{l-1} + Wavg{l} * Htest{l+1} + repmat(cavg{l}, 1, Ntest));
            Htest{l} = +(ph > rand(K(l), Ntest));
        end
        negdata_test = logistic(Wavg{1} * Htest{2} + repmat(cavg{1}, 1, Ntest));

        % 2. Sample H1
        lix = (x_kntest==0);
        [rix,cix] = find(x_kntest==0);
        p1 = negdata_test(lix).*((1-p_i_test(cix)').^r_k(rix));
        p0 = 1-negdata_test(lix);
        Htest{1} = ones(K(1),Ntest);
        Htest{1}(lix) = +(p1./(p1+p0))>rand(size(rix)); 
        
        ThetaTest = gamrnd(x_kntest+(r_k*ones(1,Ntest)).*Htest{1},ones(K(1),1)*p_i_test);    
    end
    
%     % Now, collect results
    if iter <= burnin
        if dotest == 1
            X1 = Phi*(ThetaTest);  
            mid.PhiThetaTest = mid.PhiThetaTest + X1;
            mid.Count = mid.Count+1;
        end
        
        tempTrain = Phi*(Theta);
        tempTrain= bsxfun(@rdivide, tempTrain,sum(tempTrain,1));
        mid.loglikeTrain(end+1) = sum(Xtrain_s(Yflagtrain_s).*log(tempTrain(Yflagtrain_s)))/sum(Xtrain_s(:));
        
        if dotest == 1
            tempTest = mid.PhiThetaTest/mid.Count;
            tempTest= bsxfun(@rdivide, tempTest,sum(tempTest,1));
            mid.loglikeHeldout(end+1) = sum(wordsHeldout(YflagHeldout).*log(tempTest(YflagHeldout)))/sum(wordsHeldout(:)); 
            mid.loglike(end+1) = sum(wordsTest(YflagTest).*log(tempTest(YflagTest)))/sum(wordsTest(:));
        end
        
    elseif (iter>burnin && mod(iter,step)==0)
        if dotest == 1
            X1 = Phi*(ThetaTest);
            result.PhiThetaTest = result.PhiThetaTest + X1;
            result.Count = result.Count+1;
        end
        
        tempTrain = Phi*(Theta);
        tempTrain= bsxfun(@rdivide, tempTrain,sum(tempTrain,1));
        result.loglikeTrain(end+1) = sum(Xtrain_s(Yflagtrain_s).*log(tempTrain(Yflagtrain_s)))/sum(Xtrain_s(:));
        
        if dotest == 1
            tempTest = result.PhiThetaTest/result.Count;
            tempTest= bsxfun(@rdivide, tempTest,sum(tempTest,1));
            result.loglikeHeldout(end+1) = sum(wordsHeldout(YflagHeldout).*log(tempTest(YflagHeldout)))/sum(wordsHeldout(:)); 
            result.loglike(end+1) = sum(wordsTest(YflagTest).*log(tempTest(YflagTest)))/sum(wordsTest(:));
        end
          
    end;
    
    if mod(iter,1)==0
       if iter <= burnin
           if dotest == 1
               disp(['Burnin: ' num2str(iter) ' Train: ' num2str(exp(-mid.loglikeTrain(end)))...
                   ' Held out: ' num2str(exp(-mid.loglikeHeldout(end))) ' Test: ' num2str(exp(-mid.loglike(end)))]);
           else
               disp(['Burnin: ' num2str(iter) ' Train: ' num2str(exp(-mid.loglikeTrain(end)))]);
           end
       else
           if dotest == 1
               disp(['Collection: ' num2str(iter) ' Train: ' num2str(exp(-result.loglikeTrain(end)))...
                   ' Held out: ' num2str(exp(-result.loglikeHeldout(end))) ' Test: ' num2str(exp(-result.loglike(end)))]);
           else
               disp(['Collection: ' num2str(iter) ' Train: ' num2str(exp(-result.loglikeTrain(end)))]);
           end
       end;
        
        disp(['min_r ' num2str(min(r_k)) ' max_r ' num2str(max(r_k)) ' r0 ' num2str(gamma0)]);
        
        if do_fig == 1
            figure(1);
            for nl = 1:nlayer
                subplot(1,nlayer,nl); imagesc(H{nl});colorbar; title(strcat('H', num2str(nl)));
            end
            %subplot(1,2,2); imagesc(H{2}); colorbar; title('H2');
            figure(2);
            subplot(1,3,1); bar(Phi(:, 1)); title('Phi1');set(gca, 'xlim', [0 P]);%imagesc(Theta); colorbar; title('Theta');
            subplot(1,3,2); bar(r_k); title('r_k');set(gca, 'xlim', [0 K(1)]);
            subplot(1,3,3); imagesc(W{1}); colorbar; title('W');
            drawnow;
        end
    end;

end

W_outputN=10;
[Topics]=OutputTopics(Phi,vocabulary,W_outputN);

