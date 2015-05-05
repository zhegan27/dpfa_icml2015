% Deep Poisson Factor Analysis for Topic Modeling
% using Deep Belief Network
% for 20NEWS dataset
% 
% Written by Changyou Chen (cchangyou@gmail.com)

clear all; clc; close all;
seed = 0; rand('state',seed); randn('state',seed);

load '20news_data';

% Setup
[P,N] = size(wordsTrain); [~,Ntest] = size(wordsTest); wordsTest0 = wordsTest;
Ns = 50; %%% batch size
K = [128, 64, 32]; %%% Network configulation

nlayer = length(K);

Phi_ = rand(P,K(1)); Phi = bsxfun(@rdivide,Phi_,sum(Phi_,1));
Theta = 1/K(1)*ones(K(1), Ns);
ThetaTest = 1/K(1)*ones(K(1), Ntest);

H = cell(nlayer, 1);
for k = 1:nlayer
    H{k} = ones(K(k), Ns);
end

Htest = cell(nlayer, 1);
for k = 1:nlayer-1
    Htest{k} = zeros(K(k), Ntest);
end
Htest{nlayer} = ones(K(nlayer), Ntest);

b = 0*randn(K(nlayer), 1);%-10*ones(K(nlayer), 1);
W = cell(1, nlayer-1);
c = cell(1, nlayer-1);
for i = 1:nlayer-1
    W{i} = 0.1 * randn(K(i), K(i+1));
    c{i} = 0 * randn(K(i), 1);
end

burnin_s = 20; %%% burnin before sampling
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
gamma0 = 1; 
gamma0_ = log(gamma0);
eta = 0.05;
r_k = 20*ones(K(1),1);
r_k_ = log(r_k);
p_ii = 0.5*ones(1, N); %%% p_i/(1-p_i) = 1 --> p_i=0.5;
p_i_test = 0.5*ones(1, Ntest);
e0 = 1e-2; f0 = 1e-2;

C = 0.0008;
eta_s = 4e-08
Leapfrog = 50;
sigma = 1;
h = sqrt(eta_s);%5e-6
eta_w = eta_s;
h_w = sqrt(eta_w);
eta_r = eta_s * 100;
h_r = sqrt(eta_r);

u_w = cell(1, nlayer-1);
u_c = cell(1, nlayer-1);
for i = 1:nlayer-1
    u_w{i} = h_w * randn(size(W{i}));
    u_c{i} = h_w * randn(size(c{i}));
end
u_b = h_w * randn(size(b));

ALPHA_w = cell(1, nlayer-1);
ALPHA_c = cell(1, nlayer-1);
for i = 1:nlayer-1
    ALPHA_w{i} = C * ones(size(u_w{i}));
    ALPHA_c{i} = C * ones(size(u_c{i}));
end
ALPHA_b = C * ones(size(u_b));

gammaTrain = cell(nlayer-1, 1);
gammaTest = cell(nlayer-1, 1);

gradw = cell(nlayer-1, 1);

% collect results
result.loglike=[]; result.loglikeTrain=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTrain = 0; result.PhiThetaTest = 0; result.Count = 0; 
mid.loglike=[]; mid.loglikeTrain=[]; mid.loglikeHeldout=[];
mid.K=[]; mid.PhiThetaTrain = 0; mid.PhiThetaTest = 0; mid.Count = 0;

update_r = burnin + collection;%1000;
rand_phi = burnin-100;

dotest = 1;
do_fig = 0;

p = randperm(N);
select_doc = 1;

for iter = 1:burnin + collection
    
    if iter < burnin
        dotest = 0;
    else
        dotest = 1;
    end
	if iter < burnin - 10
		sample_test = 0;
	else
		sample_test = 1;
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
    
    prob = 1./(1+exp(-b));
    H{nlayer} = +(repmat(prob, 1, Ns) > rand(K(nlayer), Ns));%zeros(K(nlayer), Ns);
    for nl = nlayer-1:-1:3
        X = bsxfun(@plus, W{nl} * (H{nl+1}), c{nl}); 
        prob = 1 ./ (1+exp(-X));
        H{nl} = +(prob >= rand(K(nl), Ns));
    end
	H{2} = ones(K(2), Ns);
    H{1} = ones(K(1), Ns);
        
    X = repmat(Xtrain_s, 1, ns);
    pi_repmat = repmat(log(p_i), K(1), ns);
    
    for ss = 1:maxiter

        % 1. Sample x_pnk
    	[x_pk,x_kn] = mult_rand(Xtrain_s,Phi,Theta);
        
        % 3. inference of sbn
        % (1). update gamma0
        for nl = 1:nlayer-1
            Xmat = bsxfun(@plus, W{nl} * (H{nl+1}), c{nl});
        	Xvec = reshape(Xmat, K(nl) * Ns, 1);
        	gamma0vec = PolyaGamRndTruncated(ones(K(nl) * Ns, 1), Xvec, 20);
        	gammaTrain{nl} = reshape(gamma0vec, K(nl), Ns);
        end

        % 2. Sample H1
        lix = (x_kn==0);
        [rix,cix] = find(x_kn==0);
        T = bsxfun(@plus,W{1}*H{2},c{1});
        prob = 1./(1+exp(-T));
        p1 = prob(lix).*((1-p_i(cix)').^r_k(rix));
        p0 = 1-prob(lix);
        H{1} = ones(K(1),Ns);
        H{1}(lix) = (p1./(p1+p0))>rand(size(rix));

        % (3). update H2
        for nl = nlayer:-1:2
        	res = W{nl-1} * (H{nl});
            for k = 1:K(nl)
                res = res - W{nl-1}(:, k) * (H{nl}(k, :));
            	mat1 = bsxfun(@plus, res, c{nl-1});
                vec1 = sum(bsxfun(@times, H{nl-1} - 0.5 - gammaTrain{nl-1} .* mat1, W{nl-1}(:, k)));
                vec2 = sum(bsxfun(@times, gammaTrain{nl-1}, W{nl-1}(:, k).^2)) / 2;
                if nl == nlayer
                    logz = vec1 - vec2 + b(k);
                else
                    logz = vec1 - vec2 + W{nl}(k, :) * (H{nl+1}) + c{nl}(k);
                end
                probz = 1 ./ (1 + exp(-logz));
                H{nl}(k, :) = (probz > rand(1, Ns));
                res = res + W{nl-1}(:, k) * (H{nl}(k, :));
            end
        end
        
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
            pg_s{ii} = gammaTrain{1};
            xpk_s{ii} = x_pk;
            xkn_s{ii} = x_kn;
        end
    end

    xpk_m = xpk_s{1};
    for k = 2:ns
        xpk_m = xpk_m + xpk_s{k};
    end
    gradPhi = xpk_m - repmat(sum(xpk_m, 1), P, 1) .* Phi;
    gradPhi = N * gradPhi / Ns / ns + (eta - Phi_);%(eta-1)*(1 - P*Phi);
    
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

        gradr = (gamma0 + L_k - 1) + (sumbpi - c0) .* r_k;

        p_prime_k = -sumbpi./(c0-sumbpi);
        gradr0 = (e0 + CRT(ceil(L_k), gamma0) - 1) / gamma0 + sum(log(max(1-p_prime_k,realmin))) - f0;
        gradr0 = gradr0 * gamma0;
        
    end

    for nl = 1:nlayer-1
        gradw{nl} = zeros(K(nl), K(nl+1) + 1);

        mat_h2 = cell2mat(H_s{nl+1});
        mat = [mat_h2; ones(1, ns*Ns)]';
        
        for j = 1:K(nl)
            prodw = mat * [W{nl}(j, :), c{nl}(j)]';
            idx = prodw > 10;
            prodw(idx) = 1 ./ (1 + exp(-prodw(idx)));
            idx = ~idx;
            prodw(idx) = exp(prodw(idx));
            prodw(idx) = prodw(idx) ./ (1 + prodw(idx));
            gradw{nl}(j, :) = gradw{nl}(j, :) - sum(repmat(prodw, 1, K(nl+1) + 1) .* mat, 1);
            idx = (mat_h1(j, :) == 1)';
            gradw{nl}(j, :) = gradw{nl}(j, :) + sum(mat(idx, :), 1);
        end
        mat_h1 = mat_h2;

        gradw{nl} = gradw{nl} * N / ns / Ns - sigma * [W{nl}, c{nl}];
        %assert(isfinite(sum(gradw{nl}(:))));
    end
    
    tmp = mean(mat, 1);
	idxb = b > 10;
	tmpb = zeros(size(b));
	tmpb(idxb) = 1 ./ (1 + exp(-b(idxb)));
	idxb = ~idxb;
	tmpb(idxb) = exp(b(idxb));
	tmpb(idxb) = tmpb(idxb) ./ (1 + tmpb(idxb));
    gradb = N * (tmp(1:K(end))' - tmpb) - sigma * b;
    
	steps = max(1e-8, (1 + iter)^(-0.33))/10;
    steps1 = steps / 100;

    for nl = 1:nlayer-1
	   u_w{nl} = h_w * randn(size(u_w{nl}));
	   u_c{nl} = h_w * randn(size(u_c{nl}));
    end
    u_b = h_w * randn(size(u_b));

    for ll = 1:Leapfrog
        for nl = 1:nlayer-1
            W{nl} = W{nl} + u_w{nl}; %assert(isfinite(sum(W{nl}(:))));
            c{nl} = c{nl} + u_c{nl}; assert(isfinite(sum(c{nl}(:))));

            u_w{nl} = (1 - ALPHA_w{nl}) .* u_w{nl} + gradw{nl}(:, 1:K(nl+1)) * eta_w + ...
                sqrt(2 * C) * h_w * randn(size(u_w{nl}));
            u_c{nl} = (1 - ALPHA_c{nl}) .* u_c{nl} + gradw{nl}(:, K(nl+1)+1) * eta_w + ...
                sqrt(2 * C) * h_w * randn(size(u_c{nl}));

            tmp = u_w{nl} .* u_w{nl};
            ALPHA_w{nl} = ALPHA_w{nl} + tmp - eta_w; %assert(isfinite(sum(ALPHA_w(:))));

            tmp = u_c{nl} .* u_c{nl};
            ALPHA_c{nl} = ALPHA_c{nl} + tmp - eta_w;
        end

        b = b + u_b;

        u_b = (1 - ALPHA_b) .* u_b + gradb * eta_w + ...
            sqrt(2 * C) * h_w * randn(size(u_b));

        tmp = u_b .* u_b;
        ALPHA_b = ALPHA_b + tmp - eta_w;
	end

    if iter < update_r
        r_k_ = r_k_ + gradr * steps1 + ...
            sqrt(2 * steps1) * randn(size(r_k)); assert(isfinite(sum(r_k_)));
        r_k = exp(r_k_);

        gamma0_ = gamma0_ + gradr0 * steps1 + ...
            sqrt(2*steps1) * randn(size(gamma0));
        gamma0 = exp(gamma0_);
    end

	Phi_ = abs(Phi_ + gradPhi * steps + sqrt(2 * steps) * sqrt(Phi_) .* randn(size(Phi_)));
        
    Phi = bsxfun(@rdivide, Phi_, sum(Phi_, 1));
    
    if sample_test == 1
        [~, x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);
        
        % (1). update gamma0
        for nl = 1:nlayer-1
            Xmat = bsxfun(@plus, W{nl} * (Htest{nl+1}), c{nl});
        	Xvec = reshape(Xmat, K(nl) * Ntest, 1);
        	gamma0vec = PolyaGamRndTruncated(ones(K(nl) * Ntest, 1), Xvec, 20);
        	gammaTest{nl} = reshape(gamma0vec, K(nl), Ntest);
        end
        
        % (3). update H2
        for nl = nlayer:-1:2
        	res = W{nl-1} * (Htest{nl});
            for k = 1:K(nl)
                res = res - W{nl-1}(:, k) * (Htest{nl}(k, :));
            	mat1 = bsxfun(@plus, res, c{nl-1});
                vec1 = sum(bsxfun(@times, Htest{nl-1} - 0.5 - gammaTest{nl-1} .* mat1, W{nl-1}(:, k)));
                vec2 = sum(bsxfun(@times, gammaTest{nl-1}, W{nl-1}(:, k).^2)) / 2;
                if nl == nlayer
                    logz = vec1 - vec2 + b(k);
                else
                    logz = vec1 - vec2 + W{nl}(k, :) * (Htest{nl+1}) + c{nl}(k);
                end
                probz = 1 ./ (1 + exp(-logz));
                Htest{nl}(k, :) = (probz > rand(1, Ntest));
                res = res + W{nl-1}(:, k) * (Htest{nl}(k, :));
            end
        end

        % 2. Sample H1
        lix = (x_kntest==0);
        [rix,cix] = find(x_kntest==0);
        T = bsxfun(@plus,W{1}*Htest{2},c{1});
        prob = 1./(1+exp(-T));
        p1 = prob(lix).*((1-p_i_test(cix)').^r_k(rix));
        p0 = 1-prob(lix);
        Htest{1} = ones(K(1), Ntest);
        Htest{1}(lix) = (p1./(p1+p0))>rand(size(rix));
        
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


