function [perpHeldout, perpTest, result] = pfa_dsbn_bcdf_test(wordsHeldout,wordsTest,...
    param)

% Setup 
Phi = param.Phi; Pi = param.Pi;
W1 = param.W1; c1 = param.c1;
W2 = param.W2;c2 =  param.c2;

K1 = param.K1; K2 = param.K2; K3 = param.K3; r_k = param.r_k;
burnin = param.burnin; collection = param.collection; step = param.step;

[~,Ntest] = size(wordsTest);

p_i_test = 0.5*ones(1,Ntest);

ThetaTest = 1/K1*ones(K1,Ntest);
H1test = ones(K1,Ntest);
H3test = +(repmat(Pi,1,Ntest) > rand(K3,Ntest));
prob = 1./(1+exp(-W2*H3test));
H2test = +(prob > rand(K2,Ntest));

YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

result.loglike=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTest = 0; result.Count = 0;


for sweep = 1:burnin + collection
    tic;
    % 1. Sample x_mnk
    [~,x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);

    % 2. Sample H1
    lix = (x_kntest==0);
    [rix,cix] = find(x_kntest==0);
    T = bsxfun(@plus,W1*H2test,c1);
    prob = 1./(1+exp(-T));
    p1 = prob(lix).*((1-p_i_test(cix)').^r_k(rix));
    p0 = 1-prob(lix);
    H1test = ones(K1,Ntest);
    H1test(lix) = (p1./(p1+p0))>rand(size(rix));

    % 3. inference of sbn	
    % (1). update gamma0
    Xmat = bsxfun(@plus,W1*H2test,c1); % K1*n
    Xvec = reshape(Xmat,K1*Ntest,1);
    gamma0vec = PolyaGamRndTruncated(ones(K1*Ntest,1),Xvec,20);
    gamma0Test = reshape(gamma0vec,K1,Ntest);

    % (3). update H2
    res = W1*H2test;
    for k = 1:K2
        res = res-W1(:,k)*H2test(k,:);
        mat1 = bsxfun(@plus,res,c1);
        vec1 = sum(bsxfun(@times,H1test-0.5-gamma0Test.*mat1,W1(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma0Test,W1(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 + W2(k,:)*H3test+c2(k); % 1*n
        probz = 1./(1+exp(-logz)); % 1*n
        H2test(k,:) = (probz>rand(1,Ntest));
        res = res+W1(:,k)*H2test(k,:);
    end;

    % (5). update gamma1
    Xmat = bsxfun(@plus,W2*H3test,c2); % K2*n
    Xvec = reshape(Xmat,K2*Ntest,1);
    gamma1vec = PolyaGamRndTruncated(ones(K2*Ntest,1),Xvec,20);
    gamma1Test = reshape(gamma1vec,K2,Ntest);

    % (7). update H3
    res = W2*H3test;
    for k = 1:K3
        res = res-W2(:,k)*H3test(k,:); % k1*n
        mat1 = bsxfun(@plus,res,c2);
        vec1 = sum(bsxfun(@times,H2test-0.5-gamma1Test.*mat1,W2(:,k))); % 1*n
        vec2 = sum(bsxfun(@times,gamma1Test,W2(:,k).^2))/2; % 1*n
        logz = vec1 - vec2 ; % 1*n
        probz = exp(logz)*Pi(k)./(exp(logz)*Pi(k)+1-Pi(k)); % 1*n
        H3test(k,:) = (probz>rand(1,Ntest));
        res = res+W2(:,k)*H3test(k,:); % k1*n
    end;

    % 5. Sample Theta
    ThetaTest = gamrnd(x_kntest+(r_k*ones(1,Ntest)).*H1test,ones(K1,1)*p_i_test);

    % Now, collect results
    if (sweep>burnin && mod(sweep,step)==0)
        X1test = Phi*ThetaTest;  
        result.PhiThetaTest = result.PhiThetaTest + X1test;
        result.Count = result.Count+1;
        tempTest = result.PhiThetaTest/result.Count;
        tempTest= bsxfun(@rdivide, tempTest,sum(tempTest,1));
        result.loglikeHeldout(end+1) = sum(wordsHeldout(YflagHeldout).*log(tempTest(YflagHeldout)))/sum(wordsHeldout(:)); 
        result.loglike(end+1) = sum(wordsTest(YflagTest).*log(tempTest(YflagTest)))/sum(wordsTest(:));  
        result.K(end+1) = nnz(sum(x_kntest,2)); 
    end;
%     timespent = toc;
%     if (sweep <= burnin)
%         disp(['sweep: ' num2str(sweep) ' Timespent: ' num2str(timespent)]);
%     else
%         disp(['sweep: ' num2str(sweep) ' Collection: ' num2str(sweep) ' Heldout: '...
%             num2str(exp(-result.loglikeHeldout(end))) ' Test: '...
%             num2str(exp(-result.loglike(end)))...
%             ' Timespent: ' num2str(timespent)]);
%     end;
end;

perpHeldout = exp(-result.loglikeHeldout(end));

perpTest = exp(-result.loglike(end));



