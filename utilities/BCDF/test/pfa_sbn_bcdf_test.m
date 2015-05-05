function [perpHeldout, perpTest, result] = pfa_sbn_bcdf_test(wordsHeldout,...
    wordsTest,param)

% Setup 
K1 = param.K1; K2 = param.K2;
Phi = param.Phi; Pi = param.Pi;
W = param.W; c = param.c;
   
r_k = param.r_k; burnin = param.burnin; 
collection = param.collection; step = param.step;

[~,Ntest] = size(wordsTest);


p_i_test = 0.5*ones(1,Ntest);
ThetaTest = 1/K1*ones(K1,Ntest);
H1test = ones(K1,Ntest);
H2test = +(repmat(Pi,1,Ntest) > rand(K2,Ntest));

YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

result.loglike=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTest = 0; result.Count = 0;


for sweep = 1:burnin + collection
    % 1. Sample x_mnk
    [~,x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);
    
    % 2. Sample H1
    lix = (x_kntest==0);
    [rix,cix] = find(x_kntest==0);
    T = bsxfun(@plus,W*H2test,c);
    prob = 1./(1+exp(-T));
    p1 = prob(lix).*((1-p_i_test(cix)').^r_k(rix));
    p0 = 1-prob(lix);
    H1test = ones(K1,Ntest);
    H1test(lix) = (p1./(p1+p0))>rand(size(rix));

    % 3. inference of sbn	
    % (1). update gamma0
    Xmat = bsxfun(@plus,W*H2test,c); % K1*n
    Xvec = reshape(Xmat,K1*Ntest,1);
    gamma0vec = PolyaGamRndTruncated(ones(K1*Ntest,1),Xvec,20);
    gamma0Test = reshape(gamma0vec,K1,Ntest);
    
    % (3). update H2
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

    % 3. Sample Theta
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
%     if (sweep <= burnin)
%         disp(['sweep: ' num2str(sweep)]);
%     else
%         disp(['sweep: ' num2str(sweep) ' Collection: ' num2str(sweep) ' Heldout: '...
%             num2str(exp(-result.loglikeHeldout(end))) ' Test: '...
%             num2str(exp(-result.loglike(end)))...
%                 ' Topic Num: ' num2str(result.K(end))]);
%     end;
end;

perpHeldout = exp(-result.loglikeHeldout(end));

perpTest = exp(-result.loglike(end));



