
% Written by David Carlson, (david.carlson@duke.edu), Duke ECE

function [params,performance]=RBM_RS_SGD_wiki(opts,vtestObs,vtestHO)
%% Init;
performance=[];
% [M,N]=size(v); 
M = 7702;
J=opts.J;
%% starting params
params.W=.1*randn(M,J);
params.sb=.5*ones(J,1);
params.b=isigmoid(params.sb);
params.c=zeros(M,1);

%%
fid_train = fopen('wiki_train.txt', 'r');

for iter=1:opts.stopping.maxIters
    if mod(iter,100)==0
        fprintf('Iteration %d\n',iter)
    end
%     vSS=subsampleData(v,opts.gradient.batchSize);
    vSS = GetBatch(fid_train, opts.gradient.batchSize, 'wiki_train.txt');
    [vSSCD,hSSCD,ph1]=RBM_RS_CD(params,vSS,opts.gradient.CD);
    %% Estimate gradients
    scal=opts.gradient.batchSize;
    dW=(vSS*ph1'-vSSCD*hSSCD')./scal-opts.penalties.penW*params.W;
    dc=sum((vSS-vSSCD),2)./scal;
    DSS=sum(vSS);
    db=(ph1-hSSCD)*(DSS(:))./sum(DSS);
    %% Update parameters
    params.W=params.W+...
        opts.stepsizes.stepW*iter^-opts.stepsizes.decay*dW;
    params.c=params.c+...
        opts.stepsizes.stepc*iter^-opts.stepsizes.decay*dc;
    params.b=params.b+...
        opts.stepsizes.stepb*iter^-opts.stepsizes.decay*db;
    %% Estimate hold-out performance
    if mod(iter,opts.test.iter)==0
        [vtestObsSS,ndx]=subsampleData(vtestObs);
        vtestHOSS=vtestHO(:,ndx);
        [MLLK_test_holdout,MLLK_test_in]=RBM_RS_Test(params,vtestObsSS,vtestHOSS,opts);
        a=exp(-[MLLK_test_holdout,MLLK_test_in]);
        fprintf(...
            'Hold-out training data perplexity is %0.3e.\n Hold-out testing data perplexity is %0.3e.\n',...
            full(a(2)),full(a(1)))
    end
    
end
fclose(fid_train);
%%
[MLLK_test_holdout,MLLK_test_in]=RBM_RS_Test(params,vtestObs,vtestHO,opts);
performance.perplexity.holdout= exp(-[MLLK_test_holdout]);
performance.perplexity.test_in= exp(-[MLLK_test_in]);
performance.llk.holdout= [MLLK_test_holdout];
performance.llk.test_in= [MLLK_test_in];
