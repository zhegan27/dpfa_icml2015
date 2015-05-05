function [ElnB,ElnPtop,id_parent,id_me] = func_process_tree(Tree,beta0,gamma1)
% process the tree for the current batch

godel = log([2 3 5 7 11 13 17 19 23 29 31 37 41 43 47]);

Voc = length(Tree(1).beta_cnt);
tot_tops = length(Tree);

id_parent = zeros(tot_tops,1);
id_me = zeros(tot_tops,1);
ElnB = zeros(tot_tops,Voc);
count = zeros(tot_tops,1);
for i = 1:length(Tree)
    id_parent(i) = Tree(i).parent*godel(1:length(Tree(i).parent))';
    id_me(i) = Tree(i).me*godel(1:length(Tree(i).me))';
    ElnB(i,:) = psi(Tree(i).beta_cnt + beta0) - psi(sum(Tree(i).beta_cnt + beta0));
    count(i) = Tree(i).cnt;
end

ElnPtop = zeros(tot_tops,1);
groups = unique(id_parent);
for g = 1:length(groups)
    group_idx = find(id_parent==groups(g));
    this = count(group_idx);
    [group_count,sort_group_idx] = sort(this,'descend');
    a = group_count + 1;
    b = [rev_cumsum(group_count(2:end)) ; 0] + gamma1;
    ElnV = psi(a) - psi(a+b);
    Eln1_V = psi(b) - psi(a+b);
    vec = ElnV + [0 ; cumsum(Eln1_V(1:end-1))];
    ElnPtop(group_idx(sort_group_idx)) = vec;
end
    
    
    