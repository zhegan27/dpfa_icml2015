
% Written by David Carlson, (david.carlson@duke.edu), Duke ECE

clear all; clc; close all;
seed = 100; rand('state',seed); randn('state',seed);

load 'wiki_test';
%%
K=128; % number of topics/hidden nodes.
%%
% number of hidden units
optimOpts.J=K;
optimOpts.M=7702;
% stopping parameters
optimOpts.stopping.maxIters=10000;
% gradient estimation parameters
optimOpts.gradient.batchSize=100;
optimOpts.gradient.CD=5; % Higher is better, but slower
% stepsize parameters
ff=100;
optimOpts.stepsizes.stepc=.1;%ff/optimOpts.M;
optimOpts.stepsizes.stepW=.1;%ff/(optimOpts.M*optimOpts.J);
optimOpts.stepsizes.stepb=.1;%ff/optimOpts.J;
optimOpts.stepsizes.decay=.2;
optimOpts.penalties.penW=1e-4;
% testing parameters
optimOpts.test.iter=1000;
optimOpts.test.collect=20;
optimOpts.test.subset=500;
%%
%(v,opts,vtestObs,vtestHO)
% v=wordsTrain;
% opts=optimOpts;
% vtestObs=wordsHeldout;
% vtestHO=wordsTest;
% RBM_RS_SGD
[params,performance]=RBM_RS_SGD_wiki(optimOpts,wordsHeldout,wordsTest);
% return
save(['wiki_RBM_SGD_K' num2str(K)],'params','performance');
