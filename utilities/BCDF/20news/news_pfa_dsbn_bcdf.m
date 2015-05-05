
clear all; clc; close all;
seed = 100; rand('state',seed); randn('state',seed);

% load dataset
load '20news_data';

% Setup 
[P,Ntrain] = size(wordsTrain); [~,Ntest] = size(wordsTest);
K1 = 128; K2 = 64; K3 = 32;

% global variable
Phi_curr = rand(P,K1); Phi_curr = bsxfun(@rdivide,Phi_curr,sum(Phi_curr,1));
W1_curr = 0.1*randn(K1,K2); c1_curr = 0.1*randn(K1,1);
W2_curr = 0.1*randn(K2,K3); c2_curr = 0.1*randn(K2,1);
Pi_curr = 1/K3*ones(K3,1);

r_k = 200/K1*ones(K1,1);

% options
burnin = 60; collection = 40; step = 1;

% initialization
c0 = 1; gamma0 = 1; eta = 0.05;
e0 = 1e-2; f0 = 1e-2;

x_pk_acc_curr = zeros(P,K1); Znz_acc_curr = zeros(K3,1);
W1_suffstat1_curr = zeros(K1,K2); W1_suffstat2_curr = zeros(K1,K2);
c1_suffstat1_curr = zeros(K1,1); c1_suffstat2_curr = zeros(K1,1);
W2_suffstat1_curr = zeros(K2,K3); W2_suffstat2_curr = zeros(K2,K3);
c2_suffstat1_curr = zeros(K2,1); c2_suffstat2_curr = zeros(K2,1);
    
batchsize = 100;
p_i_train = 0.5*ones(1,batchsize);

perpHeldout= []; perpTest = [];

for batch = 1:2000

    vSS = subsampleData(wordsTrain,batchsize);
      
    % initialize global variables
    Phi = Phi_curr; Pi = Pi_curr;
    W1 = W1_curr; c1 = c1_curr;
    W2 = W2_curr; c2 = c2_curr;
    Phi_curr = zeros(P,K1); Pi_curr = zeros(K3,1);
    W1_curr = zeros(K1,K2); c1_curr = zeros(K1,1);
    W2_curr = zeros(K2,K3); c2_curr = zeros(K2,1);
    
    % initialize local variables
    ThetaTrain = 1/K1*ones(K1,batchsize);
    H1train = ones(K1,batchsize);
    H3train = +(repmat(Pi,1,batchsize) > rand(K3,batchsize));
    prob = 1./(1+exp(-W2*H3train));
    H2train = +(prob > rand(K2,batchsize));
    
    % store previous sufficient statistics
    x_pk_acc_prev = x_pk_acc_curr; Znz_acc_prev = Znz_acc_curr;
    W1_suffstat1_prev = W1_suffstat1_curr; W1_suffstat2_prev = W1_suffstat2_curr;
    c1_suffstat1_prev = c1_suffstat1_curr; c1_suffstat2_prev = c1_suffstat2_curr;
    W2_suffstat1_prev = W2_suffstat1_curr; W2_suffstat2_prev = W2_suffstat2_curr;
    c2_suffstat1_prev = c2_suffstat1_curr; c2_suffstat2_prev = c2_suffstat2_curr;
    
    x_pk_acc_curr = zeros(P,K1); Znz_acc_curr = zeros(K3,1);
    W1_suffstat1_curr = zeros(K1,K2); W1_suffstat2_curr = zeros(K1,K2); W1_suffstat2 = zeros(K1,K2);
    c1_suffstat1_curr = zeros(K1,1); c1_suffstat2_curr = zeros(K1,1);
    W2_suffstat1_curr = zeros(K2,K3); W2_suffstat2_curr = zeros(K2,K3); W2_suffstat2 = zeros(K2,K3);
    c2_suffstat1_curr = zeros(K2,1); c2_suffstat2_curr = zeros(K2,1);
    
    result.loglikeTrain=[]; 
    result.K=[]; result.PhiThetaTrain = 0; result.Count = 0;

    for sweep = 1:burnin + collection
        % 1. Sample x_mnk
        [x_pk,x_kntrain] = mult_rand(vSS,Phi,ThetaTrain);
        x_pk_acc = x_pk_acc_prev + x_pk;

        % 2. Sample H1
        lix = (x_kntrain==0);
        [rix,cix] = find(x_kntrain==0);
        T = bsxfun(@plus,W1*H2train,c1);
        prob = 1./(1+exp(-T));
        p1 = prob(lix).*((1-p_i_train(cix)').^r_k(rix));
        p0 = 1-prob(lix);
        H1train = ones(K1,batchsize);
        H1train(lix) = (p1./(p1+p0))>rand(size(rix));
    
        % 3. inference of sbn	
        % (1). update gamma0
        Xmat = bsxfun(@plus,W1*H2train,c1); % K1*n
        Xvec = reshape(Xmat,K1*batchsize,1);
        gamma0vec = PolyaGamRndTruncated(ones(K1*batchsize,1),Xvec,20);
        gamma0Train = reshape(gamma0vec,K1,batchsize);
        
        % (2). update W
        mat = gamma0Train*H2train';
        W1_suffstat1 = W1_suffstat1_prev + mat;
        SigmaW = 1./(W1_suffstat1+1);
        
        res = W1*H2train;
        for k = 1:K2
            res = res-W1(:,k)*H2train(k,:);
            mat = bsxfun(@plus,res,c1);
            vec = (H1train-mat.*gamma0Train-1/2)*H2train(k,:)';
            W1_suffstat2(:,k) = W1_suffstat2_prev(:,k) + vec;
            muW = SigmaW(:,k).*W1_suffstat2(:,k);
            W1(:,k) = normrnd(muW,SigmaW(:,k));            
            res = res+W1(:,k)*H2train(k,:);
        end;
        
        % (3). update H2
        res = W1*H2train;
        for k = 1:K2
            res = res-W1(:,k)*H2train(k,:);
            mat1 = bsxfun(@plus,res,c1);
            vec1 = sum(bsxfun(@times,H1train-0.5-gamma0Train.*mat1,W1(:,k))); % 1*n
            vec2 = sum(bsxfun(@times,gamma0Train,W1(:,k).^2))/2; % 1*n
            logz = vec1 - vec2 + W2(k,:)*H3train+c2(k); % 1*n
            probz = 1./(1+exp(-logz)); % 1*n
            H2train(k,:) = (probz>rand(1,batchsize));
            res = res+W1(:,k)*H2train(k,:);
        end;
        
        % (4). update c
        vec = sum(gamma0Train,2);
        c1_suffstat1 = c1_suffstat1_prev + vec;
        sigmaC = 1./(c1_suffstat1+1);
        
        vec = sum(H1train-0.5-gamma0Train.*(W1*H2train),2);
        c1_suffstat2 = c1_suffstat2_prev + vec;
        muC = sigmaC.*c1_suffstat2;    
        c1 = normrnd(muC,sqrt(sigmaC));
        
        % (5). update gamma1
        Xmat = bsxfun(@plus,W2*H3train,c2); % K2*n
        Xvec = reshape(Xmat,K2*batchsize,1);
        gamma1vec = PolyaGamRndTruncated(ones(K2*batchsize,1),Xvec,20);
        gamma1Train = reshape(gamma1vec,K2,batchsize);
        
        % (6). update W2
        mat = gamma1Train*H3train';
        W2_suffstat1 = W2_suffstat1_prev + mat;
        SigmaW = 1./(W2_suffstat1+1);
        
        res = W2*H3train;
        for k = 1:K3
            res = res-W2(:,k)*H3train(k,:);
            mat = bsxfun(@plus,res,c2);
            vec = (H2train-mat.*gamma1Train-1/2)*H3train(k,:)';
            W2_suffstat2(:,k) = W2_suffstat2_prev(:,k) + vec;
            muW = SigmaW(:,k).*W2_suffstat2(:,k);
            W2(:,k) = normrnd(muW,SigmaW(:,k));            
            res = res+W2(:,k)*H3train(k,:);
        end;
        
        % (7). update H3
        res = W2*H3train;
        for k = 1:K3
            res = res-W2(:,k)*H3train(k,:); % k1*n
            mat1 = bsxfun(@plus,res,c2);
            vec1 = sum(bsxfun(@times,H2train-0.5-gamma1Train.*mat1,W2(:,k))); % 1*n
            vec2 = sum(bsxfun(@times,gamma1Train,W2(:,k).^2))/2; % 1*n
            logz = vec1 - vec2 ; % 1*n
            probz = exp(logz)*Pi(k)./(exp(logz)*Pi(k)+1-Pi(k)); % 1*n
            H3train(k,:) = (probz>rand(1,batchsize));
            res = res+W2(:,k)*H3train(k,:); % k1*n
        end;
        
        Znz = sum(H3train,2);
        Znz_acc = Znz_acc_prev + Znz;
        Pi = betarnd(c0/K3+Znz_acc,c0-c0/K3+batchsize*batch-Znz_acc);

        % (8). update c2
        vec = sum(gamma1Train,2);
        c2_suffstat1 = c2_suffstat1_prev + vec;
        sigmaC = 1./(c2_suffstat1+1);
        
        vec = sum(H2train-0.5-gamma1Train.*(W2*H3train),2);
        c2_suffstat2 = c2_suffstat2_prev + vec;
        muC = sigmaC.*c2_suffstat2;    
        c2 = normrnd(muC,sqrt(sigmaC));
		
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
            W1_curr = W1_curr + W1/collection;
            c1_curr = c1_curr + c1/collection;
            W2_curr = W2_curr + W2/collection;
            c2_curr = c2_curr + c2/collection;
            
            x_pk_acc_curr = x_pk_acc_curr + x_pk_acc/collection;
            Znz_acc_curr = Znz_acc_curr + Znz_acc/collection;
            W1_suffstat1_curr = W1_suffstat1_curr + W1_suffstat1/collection;
            W1_suffstat2_curr = W1_suffstat2_curr + W1_suffstat2/collection;
            c1_suffstat1_curr = c1_suffstat1_curr + c1_suffstat1/collection;
            c1_suffstat2_curr = c1_suffstat2_curr + c1_suffstat2/collection;
            W2_suffstat1_curr = W2_suffstat1_curr + W2_suffstat1/collection;
            W2_suffstat2_curr = W2_suffstat2_curr + W2_suffstat2/collection;
            c2_suffstat1_curr = c2_suffstat1_curr + c2_suffstat1/collection;
            c2_suffstat2_curr = c2_suffstat2_curr + c2_suffstat2/collection;
        end;
    end;
%     figure(111);
%     subplot(1,2,1); imagesc(H1train); subplot(1,2,2); imagesc(H2train);
%     drawnow;
    disp(['batch: ' num2str(batch) ' Train: ' num2str(exp(-result.loglikeTrain(end)))]);
    
    % collect inferred parameters now
    param.Phi = Phi_curr; param.Pi = Pi_curr;
    param.W1 = W1_curr; param.c1 = c1_curr;
    param.W2 = W2_curr; param.c2 = c2_curr;
    
    param.K1 = K1; param.K2 = K2; param.K3 = K3; param.r_k = r_k;
    param.burnin = burnin; param.collection = collection; param.step = step;
        
    % test
    if mod(batch,1) == 0
        [vHeldoutSS,ndx]=subsampleData(wordsHeldout);
        vTestSS=wordsTest(:,ndx);
        [perpHeldout(end+1), perpTest(end+1),~] = pfa_dsbn_bcdf_test(vHeldoutSS,...
            vTestSS,param);
        disp(['Heldout: ' num2str(perpHeldout(end)) ' Test: ' num2str(perpTest(end))]);
    end;
end;

% test on the whole test dataset
param.Phi = Phi_curr; param.Pi = Pi_curr;
param.W1 = W1_curr; param.c1 = c1_curr;
param.W2 = W2_curr; param.c2 = c2_curr;
    
param.K1 = K1; param.K2 = K2; param.K3 = K3; param.r_k = r_k;
param.burnin = 2000; param.collection = 1500; param.step = 1;

[perpHeldoutT, perpTestT, resultTest] = pfa_dsbn_bcdf_test(wordsHeldout,...
    wordsTest,param);

W_outputN=10;
[Topics]=OutputTopics(param.Phi,vocabulary,W_outputN);

save('20news_pfa_dsbn_bcdf_K128','param','perpHeldoutT','perpTestT','resultTest','Topics');
