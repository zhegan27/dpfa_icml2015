
% Deep Poisson Factor Analysis

% The Matlab Code for the ICML 2015 paper 
% "Scalable Deep Poisson Factor Analysis for Topic Modeling"
% Code written by: Zhe Gan (zhe.gan@duke.edu), 
% Changyou Chen (cchangyou@gmail.com), Ricardo Henao, 
% David Carlson, Duke University, ECE department, 5.4.2015.

% License
% Please note that this code should be used at your own risk. 
% There is no implied guarantee that it will
% not do anything stupid. Permission is granted to use and modify the code.

% Citing DPFA
% Please cite our ICML paper in your publications if it helps your research:
% 
%     @inproceedings{Gan15dpfa,
%       Author = {Z. Gan, C. Chen, R. Henao, D. Carlson, and L. Carin},
%       Title = {Scalable Deep Poisson Factor Analysis for Topic Modeling},
%       booktitle={ICML},
%       Year  = {2015}
%     }

addpath(genpath('.'));

%% inference method 1: SGNHT
% (1) model: DPFA using SBN
wiki_dpfa_sbn_sgnht;

% (2) model: DPFA using RBM
wiki_dpfa_rbm_sgnht;

%% inference method 2: BCDF
wiki_lda_bcdf; % lda
wiki_pfa_bcdf; % pfa
wiki_pfa_sbn_bcdf; % pfa + sbn
wiki_pfa_dsbn_bcdf; % pfa + dsbn
wiki_pfa_dsbn_bcdf_v2; % pfa + dsbn

%% RSM: Replicated Softmax
wiki_rsm_sgd;

%% nHDP
% the code can be downloaded from John Paisley's homepage
wiki_nHDP;
