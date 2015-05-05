function [result,Topics] = pfa_sbn_gibbs(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K1, K2, burnin, collection, step)

% Setup 
[P,Ntrain] = size(wordsTrain); [~,Ntest] = size(wordsTest);
Phi = rand(P,K1); Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
ThetaTrain = 1/K1*ones(K1,Ntrain); ThetaTest = 1/K1*ones(K1,Ntest);
H1train = ones(K1,Ntrain); H1test = ones(K1,Ntest);
Pi = 1/K2*ones(K2,1);
H2train = +(repmat(Pi,1,Ntrain) > rand(K2,Ntrain));
H2test = +(repmat(Pi,1,Ntest) > rand(K2,Ntest));
W = 0.1*randn(K1,K2); c = 0.1*randn(K1,1);

YflagTrain = wordsTrain>0;
YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

% initialization
c0 = 1; gamma0 = 1; eta = 0.05;
r_k = 50/K1*ones(K1,1); p_i_train = 0.5*ones(1,Ntrain); p_i_test = 0.5*ones(1,Ntest);
e0 = 1e-2; f0 = 1e-2;

% collect results
result.loglike=[]; result.loglikeTrain=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTrain = 0; result.PhiThetaTest = 0; result.Count = 0; 
mid.loglike=[]; mid.loglikeTrain=[]; mid.loglikeHeldout=[];
mid.K=[]; mid.PhiThetaTrain = 0; mid.PhiThetaTest = 0; mid.Count = 0;

result.Phi = zeros(P,K1); 
result.Pi = zeros(K2,1); result.r_k = zeros(K1,1);
result.W = zeros(K1,K2); result.c = zeros(K1,1);

result.ThetaTrain = zeros(K1,Ntrain); result.ThetaTest = zeros(K1,Ntest);
result.H1train = zeros(K1,Ntrain); result.H1test = zeros(K1,Ntest); 
result.H2train = zeros(K2,Ntrain); result.H2test = zeros(K2,Ntest);
result.x_kntrain = zeros(K1,Ntrain); result.x_kntest = zeros(K1,Ntest);

for iter = 1:burnin + collection
	
    % 1. Sample x_pnk
	[x_pk,x_kntrain] = mult_rand(wordsTrain,Phi,ThetaTrain);
    [~,x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);
    
    if (iter <=50)
        ThetaTrain = gamrnd(x_kntrain+(r_k*ones(1,Ntrain)).*H1train,ones(K1,1)*p_i_train);
        ThetaTest = gamrnd(x_kntest+(r_k*ones(1,Ntest)).*H1test,ones(K1,1)*p_i_test);
        Phi = gamrnd(eta+x_pk,1);
        Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
    else
    % 2. Sample H1
    lix = (x_kntrain==0);
    [rix,cix] = find(x_kntrain==0);
    T = bsxfun(@plus,W*H2train,c);
    prob = 1./(1+exp(-T));
    p1 = prob(lix).*((1-p_i_train(cix)').^r_k(rix));
    p0 = 1-prob(lix);
    H1train = ones(K1,Ntrain);
    H1train(lix) = (p1./(p1+p0))>rand(size(rix));
    
    lix = (x_kntest==0);
    [rix,cix] = find(x_kntest==0);
    T = bsxfun(@plus,W*H2test,c);
    prob = 1./(1+exp(-T));
    p1 = prob(lix).*((1-p_i_train(cix)').^r_k(rix));
    p0 = 1-prob(lix);
    H1test = ones(K1,Ntest);
    H1test(lix) = (p1./(p1+p0))>rand(size(rix));
    
    % 3. inference of sbn	
    % (1). update gamma0
    Xmat = bsxfun(@plus,W*H2train,c); % K1*n
    Xvec = reshape(Xmat,K1*Ntrain,1);
    gamma0vec = PolyaGamRndTruncated(ones(K1*Ntrain,1),Xvec,20);
    gamma0Train = reshape(gamma0vec,K1,Ntrain);
    
    Xmat = bsxfun(@plus,W*H2test,c); % K1*n
    Xvec = reshape(Xmat,K1*Ntest,1);
    gamma0vec = PolyaGamRndTruncated(ones(K1*Ntest,1),Xvec,20);
    gamma0Test = reshape(gamma0vec,K1,Ntest);
    
    % (2). update W
    for j = 1:K1        
        Hgam = bsxfun(@times,H2train,gamma0Train(j,:));
        invSigmaW = eye(K2) + Hgam*H2train';
        MuW = invSigmaW\(sum(bsxfun(@times,H2train,H1train(j,:)-0.5-c(j)*gamma0Train(j,:)),2));
        R = choll(invSigmaW); 
        W(j,:) = (MuW + R\randn(K2,1))';
    end;

    % (3). update H2
    res = W*H2train;
    for k = 1:K2
        res = res-W(:,k)*H2train(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,H1train-0.5-gamma0Train.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Train,W(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 ; % 1*n
        probz = exp(logz)*Pi(k)./(exp(logz)*Pi(k)+1-Pi(k)); % 1*n 
        H2train(k,:) = (probz>rand(1,Ntrain));
        res = res+W(:,k)*H2train(k,:);
    end;
    
    res = W*H2test;
    for k = 1:K2
        res = res-W(:,k)*H2test(k,:);
        mat1 = bsxfun(@plus,res,c);
        vec1 = sum(bsxfun(@times,H1test-0.5-gamma0Test.*mat1,W(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,W(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 ; % 1*n
        probz = exp(logz)*Pi(k)./(exp(logz)*Pi(k)+1-Pi(k)); % 1*n 
        H2test(k,:) = (probz>rand(1,Ntest));
        res = res+W(:,k)*H2test(k,:);
    end;
    
    Znz = sum(H2train,2);
    Pi = betarnd(c0/K2+Znz,c0-c0/K2+Ntrain-Znz);
    
    % (4). update c
    sigmaC = 1./(sum(gamma0Train,2)+1);
    muC = sigmaC.*sum(H1train-0.5-gamma0Train.*(W*H2train),2);
    c = normrnd(muC,sqrt(sigmaC));
    %-----------------------------------

    % 4. Sample r_k
    [kk,~,counts] = find(x_kntrain);
    ll = zeros(size(counts));
    L_k = zeros(K1,1);
    for k=1:K1
        [L_k(k),ll(kk==k)] = CRT(counts(kk==k),r_k(k));
    end;
    sumbpi = sum(bsxfun(@times,H1train,log(max(1-p_i_train,realmin))),2);
    p_prime_k = -sumbpi./(c0-sumbpi);
    gamma0 = gamrnd(e0 + CRT(L_k,gamma0),1/(f0-sum(log(max(1-p_prime_k,realmin)))) );
    r_k = gamrnd(gamma0+L_k, 1./(-sumbpi+c0));
		
    % 5. Sample Theta
    ThetaTrain = gamrnd(x_kntrain+(r_k*ones(1,Ntrain)).*H1train,ones(K1,1)*p_i_train);
    ThetaTest = gamrnd(x_kntest+(r_k*ones(1,Ntest)).*H1test,ones(K1,1)*p_i_test);
	
    % 6. Sample Phi
	Phi = gamrnd(eta+x_pk,1);
	Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
    end;
    
    % Now, collect results
    if iter <= burnin
        X1train = Phi*ThetaTrain;  
        X1test = Phi*ThetaTest;
        mid.PhiThetaTrain = mid.PhiThetaTrain + X1train;
        mid.PhiThetaTest = mid.PhiThetaTest + X1test;
        mid.Count = mid.Count+1;
        tempTrain = mid.PhiThetaTrain/mid.Count;
        tempTrain= bsxfun(@rdivide, tempTrain,sum(tempTrain,1));
        mid.loglikeTrain(end+1) = sum(wordsTrain(YflagTrain).*log(tempTrain(YflagTrain)))/sum(wordsTrain(:)); 
        tempTest = mid.PhiThetaTest/mid.Count;
        tempTest= bsxfun(@rdivide, tempTest,sum(tempTest,1));
        mid.loglikeHeldout(end+1) = sum(wordsHeldout(YflagHeldout).*log(tempTest(YflagHeldout)))/sum(wordsHeldout(:)); 
        mid.loglike(end+1) = sum(wordsTest(YflagTest).*log(tempTest(YflagTest)))/sum(wordsTest(:));  
        mid.K(end+1) = nnz(sum(x_kntrain,2)); 
    elseif (iter>burnin && mod(iter,step)==0)
        X1train = Phi*ThetaTrain;  
        X1test = Phi*ThetaTest;
        result.PhiThetaTrain = result.PhiThetaTrain + X1train;
        result.PhiThetaTest = result.PhiThetaTest + X1test;
        result.Count = result.Count+1;
        tempTrain = result.PhiThetaTrain/result.Count;
        tempTrain= bsxfun(@rdivide, tempTrain,sum(tempTrain,1));
        result.loglikeTrain(end+1) = sum(wordsTrain(YflagTrain).*log(tempTrain(YflagTrain)))/sum(wordsTrain(:)); 
        tempTest = result.PhiThetaTest/result.Count;
        tempTest= bsxfun(@rdivide, tempTest,sum(tempTest,1));
        result.loglikeHeldout(end+1) = sum(wordsHeldout(YflagHeldout).*log(tempTest(YflagHeldout)))/sum(wordsHeldout(:)); 
        result.loglike(end+1) = sum(wordsTest(YflagTest).*log(tempTest(YflagTest)))/sum(wordsTest(:));  
        result.K(end+1) = nnz(sum(x_kntrain,2));  
        
        result.Phi = result.Phi + Phi/collection;
        result.Pi = result.Pi + Pi/collection;
        result.r_k = result.r_k + r_k/collection;
        result.W = result.W + W/collection;
        result.c = result.c + c/collection;
        
        result.ThetaTrain = result.ThetaTrain + ThetaTrain/collection;
        result.ThetaTest = result.ThetaTest + ThetaTest/collection;
        result.H1train = result.H1train + H1train/collection;
        result.H1test = result.H1test + H1test/collection;
        result.H2train = result.H2train + H2train/collection;
        result.H2test = result.H2test + H2test/collection;
        result.x_kntrain = result.x_kntrain + x_kntrain/collection;
        result.x_kntest = result.x_kntest + x_kntest/collection;

    end;
    
    if mod(iter,1)==0
        if iter <= burnin
            disp(['Burnin: ' num2str(iter) ' Train: ' num2str(exp(-mid.loglikeTrain(end)))...
                ' Heldout: ' num2str(exp(-mid.loglikeHeldout(end)))...
                ' Test: ' num2str(exp(-mid.loglike(end))) ' Topic Num: ' num2str(mid.K(end))]);
        else
            disp(['Collection: ' num2str(iter) ' Train: ' num2str(exp(-result.loglikeTrain(end)))...
                ' Heldout: ' num2str(exp(-result.loglikeHeldout(end)))...
                ' Test: ' num2str(exp(-result.loglike(end))) ' Topic Num: ' num2str(result.K(end))]);
        end;
%         figure(111);
%         subplot(1,2,1); imagesc(H1train);colorbar; title('H1');
%         subplot(1,2,2); imagesc(H2train); colorbar; title('H2');
%         figure(222);
%         subplot(1,2,1); imagesc(ThetaTrain); colorbar; title('Theta');
%         subplot(1,2,2); imagesc(W); colorbar; title('W');
%         drawnow;
    end;

end

W_outputN=10;
[Topics]=OutputTopics(result.Phi,vocabulary,W_outputN);


