function [result,Topics] = pfa_gibbs(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K, burnin, collection, step)

% Setup 
[P,Ntrain] = size(wordsTrain); [~,Ntest] = size(wordsTest);
Phi = rand(P,K); Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
ThetaTrain = 1/K*ones(K,Ntrain); ThetaTest = 1/K*ones(K,Ntest); 
Ztrain = ones(K,Ntrain); Ztest = ones(K,Ntest);
Pi = 0.01*zeros(K,1);

YflagTrain = wordsTrain>0;
YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

% initialization
c = 1; gamma0 = 1; eta = 0.05;
r_k = 50/K*ones(K,1); p_i_train = 0.5*ones(1,Ntrain); p_i_test = 0.5*ones(1,Ntest); 
e0 = 1e-2; f0 = 1e-2;

% collect results
result.loglike=[]; result.loglikeTrain=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTrain = 0; result.PhiThetaTest = 0; result.Count = 0; 
mid.loglike=[]; mid.loglikeTrain=[]; mid.loglikeHeldout=[];
mid.K=[]; mid.PhiThetaTrain = 0; mid.PhiThetaTest = 0; mid.Count = 0;

result.Phi = zeros(P,K); result.Pi = zeros(K,1); result.r_k = zeros(K,1);
result.ThetaTrain = zeros(K,Ntrain); result.ThetaTest = zeros(K,Ntest);
result.Ztrain = zeros(K,Ntrain); result.Ztest = zeros(K,Ntest);
result.x_kntrain = zeros(K,Ntrain);
result.x_kntest = zeros(K,Ntest);

for iter = 1:burnin + collection
	
    % 1. Sample x_pnk
	[x_pk,x_kntrain] = mult_rand(wordsTrain,Phi,ThetaTrain);
    [~,x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);
    
    if (iter <=50)
        ThetaTrain = gamrnd(x_kntrain+(r_k*ones(1,Ntrain)).*Ztrain,ones(K,1)*p_i_train);
        ThetaTest = gamrnd(x_kntest+(r_k*ones(1,Ntest)).*Ztest,ones(K,1)*p_i_test);
        Phi = gamrnd(eta+x_pk,1);
        Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
    else
    % 2. Sample Z
    lix = (x_kntrain==0);
    [rix,cix] = find(x_kntrain==0);
    p1 = Pi(rix).*((1-p_i_train(cix)').^r_k(rix));
    p0 = 1-Pi(rix);
    Ztrain = ones(K,Ntrain);
    Ztrain(lix) = (p1./(p1+p0))>rand(size(rix));
    
    lix = (x_kntest==0);
    [rix,cix] = find(x_kntest==0);
    p1 = Pi(rix).*((1-p_i_test(cix)').^r_k(rix));
    p0 = 1-Pi(rix);
    Ztest = ones(K,Ntest);
    Ztest(lix) = (p1./(p1+p0))>rand(size(rix));
    
    % 3. Sample Pi	
    Znz = sum(Ztrain,2);
    Pi = betarnd(c/K+Znz,c-c/K+Ntrain-Znz);

    % 4. Sample r_k
    [kk,~,counts] = find(x_kntrain);
    ll = zeros(size(counts));
    L_k = zeros(K,1);
    for k=1:K
        [L_k(k),ll(kk==k)] = CRT(counts(kk==k),r_k(k));
    end;
    sumbpi = sum(bsxfun(@times,Ztrain,log(max(1-p_i_train,realmin))),2);
    p_prime_k = -sumbpi./(c-sumbpi);
    gamma0 = gamrnd(e0 + CRT(L_k,gamma0),1/(f0-sum(log(max(1-p_prime_k,realmin)))) );
    r_k = gamrnd(gamma0+L_k, 1./(-sumbpi+c));
		
    % 5. Sample Theta
    ThetaTrain = gamrnd(x_kntrain+(r_k*ones(1,Ntrain)).*Ztrain,ones(K,1)*p_i_train);
    ThetaTest = gamrnd(x_kntest+(r_k*ones(1,Ntest)).*Ztest,ones(K,1)*p_i_test);
	
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
        
        % collect posterior mean
        result.Phi = result.Phi + Phi/collection; 
        result.Pi = result.Pi + Pi/collection; 
        result.r_k = result.r_k + r_k/collection;
        result.ThetaTrain = result.ThetaTrain + ThetaTrain/collection; 
        result.ThetaTest = result.ThetaTest + ThetaTest/collection;
        result.Ztrain = result.Ztrain + Ztrain/collection; 
        result.Ztest = result.Ztest + Ztest/collection;
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
%         figure(11);
%         imagesc(Ztrain);colorbar;
%         figure(22); imagesc(ThetaTrain);colorbar;
%         drawnow;
    end;

end

W_outputN=10;
[Topics]=OutputTopics(result.Phi,vocabulary,W_outputN);



