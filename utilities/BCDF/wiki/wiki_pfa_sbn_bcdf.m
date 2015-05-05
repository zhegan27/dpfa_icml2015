
clear all; clc; close all;
seed = 100; rand('state',seed); randn('state',seed);

load 'wiki_test';
fid_train = fopen('wiki_train.txt', 'r');

% Setup 
P = 7702;
K1 = 128; K2 = 64;

% global variable
Phi_curr = rand(P,K1); Phi_curr = bsxfun(@rdivide,Phi_curr,sum(Phi_curr,1));
Pi_curr = 1/K2*ones(K2,1); W_curr = 0.1*randn(K1,K2); c_curr = 0.1*randn(K1,1);

r_k = 1/K1*ones(K1,1);

% options
burnin = 60; collection = 40; step = 1;

% initialization
c0 = 1; gamma0 = 1; eta = 0.05;
e0 = 1e-2; f0 = 1e-2;

x_pk_acc_curr = zeros(P,K1); Znz_acc_curr = zeros(K2,1);
W_suffstat1_curr = zeros(K1,K2); W_suffstat2_curr = zeros(K1,K2);
c_suffstat1_curr = zeros(K1,1); c_suffstat2_curr = zeros(K1,1);
    
batchsize = 100;
p_i_train = 0.5*ones(1,batchsize);

perpHeldout= []; perpTest = [];

for batch = 1:3500

    vSS = GetBatch(fid_train, batchsize, 'wiki_train.txt');

    % initialize global variables
    Phi = Phi_curr; Pi = Pi_curr;
    W = W_curr; c = c_curr;
    Phi_curr = zeros(P,K1); Pi_curr = zeros(K2,1);
    W_curr = zeros(K1,K2); c_curr = zeros(K1,1);
    
    % initialize local variables
    ThetaTrain = 1/K1*ones(K1,batchsize);
    H1train = ones(K1,batchsize);
    H2train = +(repmat(Pi,1,batchsize) > rand(K2,batchsize));
    
    % store previous sufficient statistics
    x_pk_acc_prev = x_pk_acc_curr; Znz_acc_prev = Znz_acc_curr;
    W_suffstat1_prev = W_suffstat1_curr; W_suffstat2_prev = W_suffstat2_curr;
    c_suffstat1_prev = c_suffstat1_curr; c_suffstat2_prev = c_suffstat2_curr;
    
    x_pk_acc_curr = zeros(P,K1); Znz_acc_curr = zeros(K2,1);
    W_suffstat1_curr = zeros(K1,K2); W_suffstat2_curr = zeros(K1,K2); W_suffstat2 = zeros(K1,K2);
    c_suffstat1_curr = zeros(K1,1); c_suffstat2_curr = zeros(K1,1);
    
    result.loglikeTrain=[]; 
    result.K=[]; result.PhiThetaTrain = 0; result.Count = 0;

    for sweep = 1:burnin + collection
        % 1. Sample x_mnk
        [x_pk,x_kntrain] = mult_rand(vSS,Phi,ThetaTrain);
        x_pk_acc = x_pk_acc_prev + x_pk;

        % 2. Sample H1
        lix = (x_kntrain==0);
        [rix,cix] = find(x_kntrain==0);
        T = bsxfun(@plus,W*H2train,c);
        prob = 1./(1+exp(-T));
        p1 = prob(lix).*((1-p_i_train(cix)').^r_k(rix));
        p0 = 1-prob(lix);
        H1train = ones(K1,batchsize);
        H1train(lix) = (p1./(p1+p0))>rand(size(rix));
    
        % 3. inference of sbn	
        % (1). update gamma0
        Xmat = bsxfun(@plus,W*H2train,c); % K1*n
        Xvec = reshape(Xmat,K1*batchsize,1);
        gamma0vec = PolyaGamRndTruncated(ones(K1*batchsize,1),Xvec,20);
        gamma0Train = reshape(gamma0vec,K1,batchsize);
        
        
        % (2). update W
        mat = gamma0Train*H2train';
        W_suffstat1 = W_suffstat1_prev + mat;
        SigmaW = 1./(W_suffstat1+1);
        
        res = W*H2train;
        for k = 1:K2
            res = res-W(:,k)*H2train(k,:);
            mat = bsxfun(@plus,res,c);
            vec = (H1train-mat.*gamma0Train-1/2)*H2train(k,:)';
            W_suffstat2(:,k) = W_suffstat2_prev(:,k) + vec;
            muW = SigmaW(:,k).*W_suffstat2(:,k);
            W(:,k) = normrnd(muW,SigmaW(:,k));            
            res = res+W(:,k)*H2train(k,:);
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
            H2train(k,:) = (probz>rand(1,batchsize));
            res = res+W(:,k)*H2train(k,:);
        end;
        
        Znz = sum(H2train,2);
        Znz_acc = Znz_acc_prev + Znz;
        Pi = betarnd(c0/K2+Znz_acc,c0-c0/K2+batchsize*batch-Znz_acc);

        % (4). update c
        vec = sum(gamma0Train,2);
        c_suffstat1 = c_suffstat1_prev + vec;
        sigmaC = 1./(c_suffstat1+1);
        
        vec = sum(H1train-0.5-gamma0Train.*(W*H2train),2);
        c_suffstat2 = c_suffstat2_prev + vec;
        muC = sigmaC.*c_suffstat2;    
        c = normrnd(muC,sqrt(sigmaC));
		
        % 5. Sample Theta
        ThetaTrain = gamrnd(x_kntrain+(r_k*ones(1,batchsize)).*H1train,ones(K1,1)*p_i_train);
	
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
            W_curr = W_curr + W/collection;
            c_curr = c_curr + c/collection;
            x_pk_acc_curr = x_pk_acc_curr + x_pk_acc/collection;
            Znz_acc_curr = Znz_acc_curr + Znz_acc/collection;
            W_suffstat1_curr = W_suffstat1_curr + W_suffstat1/collection;
            W_suffstat2_curr = W_suffstat2_curr + W_suffstat2/collection;
            c_suffstat1_curr = c_suffstat1_curr + c_suffstat1/collection;
            c_suffstat2_curr = c_suffstat2_curr + c_suffstat2/collection;
        end;
    end;
%     figure(111);
%     subplot(1,2,1); imagesc(H1train); subplot(1,2,2); imagesc(H2train);
%     drawnow;
    disp(['batch: ' num2str(batch) ' Train: ' num2str(exp(-result.loglikeTrain(end)))]);
    
    
    % collect inferred parameters now
    param.Phi = Phi_curr; param.Pi = Pi_curr;
    param.W = W_curr; param.c = c_curr;
    
    param.K1 = K1; param.K2 = K2; param.r_k = r_k;
    param.burnin = burnin; param.collection = collection; param.step = step;
        
    % test
    if mod(batch,1) == 0
        [vHeldoutSS,ndx]=subsampleData(wordsHeldout,batchsize);
        vTestSS=wordsTest(:,ndx);
        [perpHeldout(end+1), perpTest(end+1)] = pfa_sbn_bcdf_test(vHeldoutSS,...
            vTestSS,param);
        disp(['Heldout: ' num2str(perpHeldout(end)) ' Test: ' num2str(perpTest(end))]);
    end;
end;
fclose(fid_train);

% test on the whole test dataset
param.Phi = Phi_curr; param.Pi = Pi_curr;
param.W = W_curr; param.c = c_curr;
    
param.K1 = K1; param.K2 = K2; param.r_k = r_k;
param.burnin = 2000; param.collection = 1500; param.step = 1;

[perpHeldoutT, perpTestT, resultTest] = pfa_sbn_bcdf_test(wordsHeldout,...
    wordsTest,param);

W_outputN=10;
[Topics]=OutputTopics(param.Phi,vocabulary,W_outputN);


