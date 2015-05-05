function [perpHeldout, perpTest, result] = pfa_bcdf_test(wordsHeldout,wordsTest,param)

% Setup 
K = param.K;
Phi = param.Phi;
Pi = param.Pi;
r_k = param.r_k;
burnin = param.burnin;
collection = param.collection;
step = param.step;

[~,Ntest] = size(wordsTest);

p_i_test = 0.5*ones(1,Ntest);
ThetaTest = 1/K*ones(K,Ntest);
Ztest = ones(K,Ntest);

YflagHeldout = wordsHeldout>0;
YflagTest = wordsTest>0;

result.loglike=[]; result.loglikeHeldout=[];
result.K=[]; result.PhiThetaTest = 0; result.Count = 0;

for sweep = 1:burnin + collection
    % 1. Sample x_mnk
    [~,x_kntest] = mult_rand(wordsHeldout,Phi,ThetaTest);

    % 2. Sample Z
    lix = (x_kntest==0);
    [rix,cix] = find(x_kntest==0);
    p1 = Pi(rix).*((1-p_i_test(cix)').^r_k(rix));
    p0 = 1-Pi(rix);
    Ztest = ones(K,Ntest);
    Ztest(lix) = (p1./(p1+p0))>rand(size(rix));

    % 3. Sample Theta
    ThetaTest = gamrnd(x_kntest+(r_k*ones(1,Ntest)).*Ztest,ones(K,1)*p_i_test);

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



