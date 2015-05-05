clear all; clc; close all;

% initialize / use a large subset of documents (e.g., 10,000) contained in Xid and Xcnt to initialize
load 'rcv2_data';
[P,Ntrain] = size(wordsTrain); [~,Ntest] = size(wordsTest);
Xid = cell(Ntrain,1);
Xcnt = cell(Ntrain,1);
for n=1:Ntrain
    [q1,q2,val]=find(wordsTrain(:,n));
    Xid{n}=q1';
    Xcnt{n}=val';
end

%%
% num_topics = [20 10 5];
num_topics = [10 5 5];

scale = 100000;
index = randperm(Ntrain);
Tree = nHDP_init(Xid(index(1:10000)),Xcnt(index(1:10000)),num_topics,scale);
for i = 1:length(Tree)
    if Tree(i).cnt == 0
        Tree(i).beta_cnt(:) = 0;
    end
    vec = gamrnd(ones(1,length(Tree(i).beta_cnt)),1);
    Tree(i).beta_cnt = .95*Tree(i).beta_cnt + .05*scale*vec/sum(vec);
end
disp(['finished initialization !']);

% main loop / to modify this, at each iteration send in a new subset of docs
% contained in Xid_batch and Xcnt_batch
beta0 = .1; % this parameter is the Dirichlet base distribution and can be played with
batchsize=100;
for i = 1:2
    disp(['mini batch: ' num2str(i)]);
    [a,b] = sort(rand(1,length(Xid)));
    rho = (1+i)^-.75; % step size can also be played with
    Xid_batch = Xid(b(1:batchsize));
    Xcnt_batch = Xcnt(b(1:batchsize));
    Tree = nHDP_step(Xid_batch,Xcnt_batch,Tree,scale,rho,beta0);
end

%% test
% Ntest=100;
Xid=cell(Ntest,1);
Xcnt=cell(Ntest,1);
Xid_test=cell(Ntest,1);
Xcnt_test=cell(Ntest,1);
for n=1:Ntest
    [q1,q2,val]=find(wordsHeldout(:,n));
    Xid{n}=q1';
    Xcnt{n}=val';
    [q1,q2,val]=find(wordsTest(:,n));
    Xid_test{n}=q1';
    Xcnt_test{n}=val';
end
%%
[llik_mean,C_d] = nHDP_test(Xid_test,Xcnt_test,Xid,Xcnt,Tree,beta0);
loglike = llik_mean./sum(sum(wordsTest(:,1:Ntest)));
perp = exp(-loglike);

% save('rcv2_nHDP_02022015','Tree','perp');

