function Tree = nHDP_step(Xid,Xcnt,Tree,scale,rho,beta0)
% NHDP_STEP performs one step of stochastic variational inference 
% for the nested hierarchical Dirichlet process.
%
% *** INPUT (mini-batch) ***
% Xid{d} : contains vector of word indexes for document d
% Xcnt{d} : contains vector of word counts corresponding to Xid{d}
% Tree : current top-level of nHDP
% rho : step size
%
% Written by John Paisley, jpaisley@berkeley.edu

Voc = length(Tree(1).beta_cnt);
tot_tops = length(Tree);
D = length(Xid);
size_subtree = zeros(1,D);
hist_levels = zeros(1,10);

% collects statistics for updating the tree
B_up = zeros(tot_tops,Voc);
weight_up = zeros(tot_tops,1);

gamma1 = 5; % top-level DP concentration
gamma2 = 1; % second-level DP concentration
gamma3 = 2*(1/3); % beta switches
gamma4 = 2*(2/3); %

% put info from Tree struct into matrix and vector form
[ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,gamma1);

% Family tree indicator matrix. Tree_mat(j,i) = 1 indicates that node j is 
% along the path from node i to the root node
Tree_mat = zeros(tot_tops);
for i = 1:tot_tops
    bool = 1;
    idx = i;
    while bool
        idx = find(id_me==id_parent(idx));
        if ~isempty(idx) %id_me(idx) ~= log(2)
            Tree_mat(idx,i) = 1;
        else
            bool = 0;
        end        
    end
end
level_penalty = psi(gamma3) - psi(gamma3+gamma4) + sum(Tree_mat,1)'*(psi(gamma4) - psi(gamma3+gamma4));

% main loop
for d = 1:D
    ElnB_d = ElnB(:,Xid{d});                                            % pick out words in document for penalty
    ElnV = psi(1) - psi(1+gamma2);
    Eln1_V = psi(gamma2) - psi(1+gamma2);
    ElnP_d = zeros(tot_tops,1) - inf;                                   % -inf removes non-activated topics by giving them zero probability
    ElnP_d(id_parent==log(2)) = ElnV+psi(gamma3)-psi(gamma3+gamma4);    % activate first level of topics

  % select subtree
    bool = 1;
    idx_pick = [];
    Lbound = [];
    vec_DPweights = zeros(tot_tops,1);                                  % ElnP_d minus the level penalty
    while bool
        idx_active = find(ElnP_d > -inf);                                             % index of active (selected and potential) nodes
        penalty = ElnB_d(idx_active,:) + repmat(ElnP_d(idx_active),1,length(Xid{d}));
        C_act = penalty;
        penalty = penalty.*repmat(Xcnt{d},size(penalty,1),1);
        ElnPtop_act = ElnPtop(idx_active);
        if isempty(idx_pick)
            score = sum(penalty,2) + ElnPtop_act;
            [temp,idx_this] = max(score);
            idx_pick = idx_active(idx_this);                                          % index of selected nodes
            Lbound(end+1) = temp - ElnPtop_act(idx_this);
        else
            temp = zeros(tot_tops,1); 
            temp(idx_active) = (1:length(idx_active))';
            idx_clps = temp(idx_pick);                                                % index of selected nodes within active nodes
            num_act = length(idx_active);
            vec = max(penalty(idx_clps,:),[],1);
            C_act = C_act - repmat(vec,num_act,1);
            C_act = exp(C_act);
            numerator = C_act.*penalty;
            numerator = numerator + repmat(sum(numerator(idx_clps,:),1),num_act,1);
            denominator = C_act + repmat(sum(C_act(idx_clps,:),1),num_act,1);
            vec = sum(C_act(idx_clps,:).*log(eps+C_act(idx_clps,:)),1);
            H = log(denominator) - (C_act.*log(C_act+eps) + repmat(vec,num_act,1))./denominator;
            score = sum(numerator./denominator,2) + ElnPtop_act + H*Xcnt{d}';
            score(idx_clps) = -inf;          
            [temp,idx_this] = max(score);
            idx_pick(end+1) = idx_active(idx_this);
            Lbound(end+1) = temp - ElnPtop_act(idx_this);
        end
        idx_this = find(id_parent == id_parent(idx_pick(end)));
        [t1,t2] = intersect(idx_this,idx_pick);
        idx_this(t2) = [];
        vec_DPweights(idx_this) = vec_DPweights(idx_this) + Eln1_V;
        ElnP_d(idx_this) = ElnP_d(idx_this) + Eln1_V;
        idx_add = find(id_parent == id_me(idx_pick(end)));
        vec_DPweights(idx_add) = ElnV;
        ElnP_d(idx_add) = ElnV + level_penalty(idx_add);
        bool2 = 1;
        idx = idx_pick(end);
        while bool2
            if ~isempty(idx) %id_me(idx) ~= log(2)
                ElnP_d(idx_add) = ElnP_d(idx_add) + vec_DPweights(idx);
                idx = find(id_me == id_parent(idx));
            else
                bool2 = 0;
            end
        end
        if length(Lbound) > 1
            if abs(Lbound(end)-Lbound(end-1))/abs(Lbound(end-1)) < 10^-3 || length(Lbound) == 20
                bool = 0;
            end
        end
        hist_levels(length(Tree(idx_pick(end)).me)-1) = hist_levels(length(Tree(idx_pick(end)).me)-1) + 1;
%         plot(Lbound); title(num2str(length(Tree(idx_pick(end)).me))); pause(.1);
    end
    size_subtree(d) = length(idx_pick);

  % learn document parameters for subtree
    T = length(idx_pick);
    ElnB_d = ElnB(idx_pick,Xid{d});
    ElnP_d = 0*ElnP_d(idx_pick) - 1;
    cnt_old = zeros(length(idx_pick),1);
    bool_this = 1;
    num = 0;
    while bool_this
        num = num+1;
        C_d = ElnB_d + repmat(ElnP_d,1,length(Xid{d}));
        C_d = C_d - repmat(max(C_d,[],1),T,1);
        C_d = exp(C_d);
        C_d = C_d./repmat(sum(C_d,1),T,1);
        cnt = C_d*Xcnt{d}';
        ElnP_d = func_doc_weight_up(cnt,id_parent(idx_pick),gamma2,gamma3,gamma4,Tree_mat(idx_pick,idx_pick));
        if sum(abs(cnt-cnt_old))/sum(cnt) < 10^-3 || num == 50
            bool_this = 0;
        end
        cnt_old = cnt;
%         stem(cnt); title(num2str(num)); pause(.1);
    end
    B_up(idx_pick,Xid{d}) = B_up(idx_pick,Xid{d}) + C_d.*repmat(Xcnt{d},length(idx_pick),1);
    weight_up(idx_pick) = weight_up(idx_pick) + 1;
end
subplot(2,2,[1 3]); stem(sum(B_up,2));
subplot(2,2,2); bar(hist(size_subtree,1:20));
subplot(2,2,4); bar(hist_levels/D);

% update tree
for i = 1:tot_tops
    if rho == 1
        Tree(i).beta_cnt = scale*B_up(i,:)/D;
    else
        vec = ones(1,size(B_up,2));
        vec = vec/sum(vec);
        vec = sum(B_up(i,:))*vec;
        Tree(i).beta_cnt = (1-rho)*Tree(i).beta_cnt + rho*((1-rho/10)*scale*B_up(i,:)/D + (rho/10)*scale*vec/D);
    end
    Tree(i).cnt = (1-rho)*Tree(i).cnt + rho*scale*weight_up(i)/D;
end