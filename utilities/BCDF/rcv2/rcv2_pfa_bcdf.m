
clear all; clc; close all;
seed = 100; rand('state',seed); randn('state',seed);

% load dataset
load 'rcv2_data';

% Setup 
[P,Ntrain] = size(wordsTrain); [~,Ntest] = size(wordsTest);
K = 128;

% global variable
Phi_curr = rand(P,K); Phi_curr = bsxfun(@rdivide,Phi_curr,sum(Phi_curr,1));
Pi_curr = 0.01*zeros(K,1); r_k = 200/K*ones(K,1);

% options
burnin = 60; collection = 40; step = 1;

% initialization
c = 1; gamma0 = 1; eta = 0.05;
e0 = 1e-2; f0 = 1e-2;

x_pk_acc_curr = zeros(P,K);
Znz_acc_curr = zeros(K,1);
    
batchsize = 100;
p_i_train = 0.5*ones(1,batchsize);

perpHeldout= []; perpTest = [];

for batch = 1:2000
    tic;
    vSS = subsampleData(wordsTrain,batchsize);

    ThetaTrain = 1/K*ones(K,batchsize);
    Ztrain = ones(K,batchsize);
        
    Phi = Phi_curr; Pi = Pi_curr;
    Phi_curr = zeros(P,K);
    Pi_curr = zeros(K,1);
    
    x_pk_acc_prev = x_pk_acc_curr;
    Znz_acc_prev = Znz_acc_curr;
    x_pk_acc_curr = zeros(P,K);
    Znz_acc_curr = zeros(K,1);
    
    result.loglikeTrain=[]; 
    result.K=[]; result.PhiThetaTrain = 0; result.Count = 0;

    for sweep = 1:burnin + collection
        % 1. Sample x_mnk
        [x_pk,x_kntrain] = mult_rand(vSS,Phi,ThetaTrain);
        x_pk_acc = x_pk_acc_prev + x_pk;

        % 2. Sample Z
        lix = (x_kntrain==0);
        [rix,cix] = find(x_kntrain==0);
        p1 = Pi(rix).*((1-p_i_train(cix)').^r_k(rix));
        p0 = 1-Pi(rix);
        Ztrain = ones(K,batchsize);
        Ztrain(lix) = (p1./(p1+p0))>rand(size(rix));
    
        % 3. Sample Pi	
        Znz = sum(Ztrain,2);
        Znz_acc = Znz_acc_prev + Znz;
        Pi = betarnd(c/K+Znz_acc,c-c/K+batchsize*batch-Znz_acc);
		
        % 5. Sample Theta
        ThetaTrain = gamrnd(x_kntrain+(r_k*ones(1,batchsize)).*Ztrain,ones(K,1)*p_i_train);
	
        % 6. Sample Phi
        Phi = gamrnd(eta+x_pk_acc,1);
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
            Pi_curr = Pi_curr + Pi/collection;
            x_pk_acc_curr = x_pk_acc_curr + x_pk_acc/collection;
            Znz_acc_curr = Znz_acc_curr + Znz_acc/collection;
        end;
    end;
    timespent = toc;
    disp(['batch: ' num2str(batch) ' Train: ' num2str(exp(-result.loglikeTrain(end))) ...
        ' timespent: ' num2str(timespent)]);
    
    % collect inferred parameters now
    param.Phi = Phi_curr; param.Pi = Pi_curr;
    
    param.K = K; param.r_k = r_k;
    param.burnin = burnin; param.collection = collection; param.step = step;
        
    % test
    if mod(batch,1) == 0
        [vHeldoutSS,ndx]=subsampleData(wordsHeldout,batchsize);
        vTestSS=wordsTest(:,ndx);
        [perpHeldout(end+1), perpTest(end+1), ~] = pfa_bcdf_test(vHeldoutSS,vTestSS,param);
        disp(['Heldout: ' num2str(perpHeldout(end)) ' Test: ' num2str(perpTest(end))]);
    end;
end;

% test on the whole test dataset
param.Phi = Phi_curr; param.Pi = Pi_curr;
param.K = K; param.r_k = r_k;
param.burnin = 2000; param.collection = 1500; param.step = 1;
    
[perpHeldoutT, perpTestT, resultTest] = pfa_bcdf_test(wordsHeldout,wordsTest,param);

W_outputN=10;
[Topics]=OutputTopics(param.Phi,vocabulary,W_outputN);

save('rcv2_pfa_bcdf_K128','param','perpHeldoutT','perpTestT','resultTest','Topics');

