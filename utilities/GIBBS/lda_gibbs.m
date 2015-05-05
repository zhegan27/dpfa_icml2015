function [result,Topics] = lda_gibbs(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K, burnin, collection, step)

% Setup 
[P,Ntrain] = size(wordsTrain); [~,Ntest] = size(wordsTest);
Phi = rand(P,K); Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
ThetaTrain = 1/K*ones(K,Ntrain); ThetaTest = 1/K*ones(K,Ntest); 

YflagTrain = wordsTrain>0;
YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

% initialization
eta = 0.05;

% collect results
result.loglike=[]; result.loglikeTrain=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTrain = 0; result.PhiThetaTest = 0; result.Count = 0; 
mid.loglike=[]; mid.loglikeTrain=[]; mid.loglikeHeldout=[];
mid.K=[]; mid.PhiThetaTrain = 0; mid.PhiThetaTest = 0; mid.Count = 0;

result.Phi = zeros(P,K);
result.ThetaTrain = zeros(K,Ntrain); result.ThetaTest = zeros(K,Ntest);
result.x_kntrain = zeros(K,Ntrain);
result.x_kntest = zeros(K,Ntest);

for iter = 1:burnin + collection
	
    % 1. Sample x_pnk
	[x_pk,x_kntrain] = mult_rand(wordsTrain,Phi,ThetaTrain);
    [~,x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);
		
    % 2. Sample Theta
    ThetaTrain = gamrnd(x_kntrain+50/K,1);
    ThetaTrain = bsxfun(@rdivide,ThetaTrain,sum(ThetaTrain,1));
    
    ThetaTest = gamrnd(x_kntest+50/K,1);
    ThetaTest = bsxfun(@rdivide,ThetaTest,sum(ThetaTest,1));
	
    % 3. Sample Phi
	Phi = gamrnd(eta+x_pk,1);
	Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
    
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
        result.ThetaTrain = result.ThetaTrain + ThetaTrain/collection; 
        result.ThetaTest = result.ThetaTest + ThetaTest/collection;
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
%         imagesc(Phi);colorbar;
%         figure(22); imagesc(ThetaTrain);colorbar;
%         drawnow;
    end;

end

W_outputN=10;
[Topics]=OutputTopics(result.Phi,vocabulary,W_outputN);









