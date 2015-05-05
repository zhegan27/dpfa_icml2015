
clear all; clc; close all;
seed = 100; rand('state',seed); randn('state',seed);

% load dataset
load 'rcv2_data';

% Setup 
[P,Ntrain] = size(wordsTrain);
K = 128;

Phi_curr = rand(P,K); Phi_curr = bsxfun(@rdivide,Phi_curr,sum(Phi_curr,1));

% options
burnin = 60; collection = 40; step = 1;

% initialization
c = 1; gamma0 = 1; eta = 0.05;
e0 = 1e-2; f0 = 1e-2;

x_pk_acc_curr = zeros(P,K);

batchsize = 100;
p_i_train = 0.5*ones(1,batchsize);

[~,Ntest] = size(wordsTest);
p_i_test = 0.5*ones(1,Ntest);
ThetaTest = 1/K*ones(K,Ntest);

YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

resultTest.loglike=[]; resultTest.loglikeHeldout=[];
resultTest.K=[]; resultTest.PhiThetaTest = 0; resultTest.Count = 0;

perpHeldout= []; perpTest = [];

for batch = 1:(2000 + 1500)
	
    vSS = subsampleData(wordsTrain,batchsize);
    
    ThetaTrain = 1/K*ones(K,batchsize);
    Phi = Phi_curr; Phi_curr = zeros(P,K);
    
    x_pk_acc_prev = x_pk_acc_curr;
    x_pk_acc_curr = zeros(P,K);
    
    result.loglikeTrain=[]; 
    result.K=[]; result.PhiThetaTrain = 0; result.Count = 0;
    
    for sweep = 1:burnin + collection
        % 1. Sample x_mnk
        [x_pk,x_kntrain] = mult_rand(vSS,Phi,ThetaTrain);
        x_pk_acc = x_pk_acc_prev + x_pk;

        % 2. Sample Theta
        ThetaTrain = gamrnd(x_kntrain+50/K,1);
        ThetaTrain = bsxfun(@rdivide,ThetaTrain,sum(ThetaTrain,1));
        
        % 3. Sample Phi
        Phi = gamrnd(eta+x_pk,1);
        Phi = bsxfun(@rdivide,Phi,sum(Phi,1));

        % Now, collect results
        YflagTrain = vSS>0;
        if (sweep>burnin && mod(sweep,step)==0)
            X1train = Phi*ThetaTrain;  
            result.PhiThetaTrain = result.PhiThetaTrain + X1train;
            result.Count = result.Count+1;
            tempTrain = result.PhiThetaTrain/result.Count;
            tempTrain= bsxfun(@rdivide, tempTrain,sum(tempTrain,1));
            result.loglikeTrain(end+1) = sum(vSS(YflagTrain).*log(tempTrain(YflagTrain)))/sum(vSS(:));  
            result.K(end+1) = nnz(sum(x_kntrain,2)); 

            Phi_curr = Phi_curr + Phi/collection;
            x_pk_acc_curr = x_pk_acc_curr + x_pk_acc/collection;
        end;
    end;
    
    if batch <=2000
        disp(['batch: ' num2str(batch) ' Train: ' num2str(exp(-result.loglikeTrain(end)))]);
    end;
    
    % collect inferred parameters now
    param.Phi = Phi_curr; 
    Phi = param.Phi;
    
    % test
    if (batch>2000)
        [~,x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);
  
        ThetaTest = gamrnd(x_kntest+50/K,1);
        ThetaTest = bsxfun(@rdivide,ThetaTest,sum(ThetaTest,1));
        
        X1test = Phi*ThetaTest;  
        resultTest.PhiThetaTest = resultTest.PhiThetaTest + X1test;
        resultTest.Count = resultTest.Count+1;
        tempTest = resultTest.PhiThetaTest/resultTest.Count;
        tempTest= bsxfun(@rdivide, tempTest,sum(tempTest,1));
        resultTest.loglikeHeldout(end+1) = sum(wordsHeldout(YflagHeldout).*log(tempTest(YflagHeldout)))/sum(wordsHeldout(:)); 
        resultTest.loglike(end+1) = sum(wordsTest(YflagTest).*log(tempTest(YflagTest)))/sum(wordsTest(:));  
        resultTest.K(end+1) = nnz(sum(x_kntest,2)); 
    
        perpHeldout = exp(-resultTest.loglikeHeldout(end));
        perpTest = exp(-resultTest.loglike(end));
        disp(['Heldout: ' num2str(perpHeldout) ' Test: ' num2str(perpTest)]);
    end;
end;

W_outputN=10;
[Topics]=OutputTopics(param.Phi,vocabulary,W_outputN);

save('rcv2_lda_bcdf_K128','param','perpHeldoutT','perpTestT','resultTest','Topics');



