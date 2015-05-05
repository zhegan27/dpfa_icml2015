
clear all; clc; close all;
seed = 100; rand('state',seed); randn('state',seed);
load '20news_data';

%% LDA
K = 128; burnin = 2000; collection = 1500; step = 1;

[result,Topics] = lda_gibbs(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K, burnin, collection, step);

save(['20news_LDA_Gibbs_K' num2str(K)],'result','Topics');

%% PFA 
K = 128; burnin = 2000; collection = 1500; step = 1;

[result,Topics] = pfa_gibbs(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K, burnin, collection, step);

save(['20news_PFA_Gibbs_K' num2str(K)],'result','Topics');

%% PFA + SBN
K1 = 128; K2 = K1/2; burnin = 2000; collection = 1500; step = 1;

[result,Topics] = pfa_sbn_gibbs(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K1, K2, burnin, collection, step);

save(['20news_PFA_SBN_Gibbs_K' num2str(K1) '_' num2str(K2)],'result','Topics');

%% PFA + DSBN
K1 = 128;  K2 = K1/2; K3 = K1/4;
burnin = 2000; collection = 1500; step = 1;

[result,Topics] = pfa_dsbn_gibbs(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K1, K2, K3, burnin, collection, step);

save(['20news_PFA_DSBN_Gibbs_K' num2str(K1) '_' num2str(K2) '_' num2str(K3)],'result','Topics');

%% PFA + DSBN + shrinkage
K1 = 128; K2 = K1/2; K3 = K1/4;
burnin = 2000; collection = 1500; step = 1;

[result,Topics] = pfa_dsbn_gibbs_shrinkage(wordsTrain,wordsHeldout,wordsTest,vocabulary,...
    K1, K2, K3, burnin, collection, step);

save(['20news_PFA_DSBN_Gibbs_K' num2str(K1) '_' num2str(K2) '_' num2str(K3) '_shrinkage'],'result','Topics');



