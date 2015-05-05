function ElnP_d = func_doc_weight_up(cnt,id_parent,gamma2,gamma3,gamma4,Tree_mat)
% update expected log probability of each topic selected for this document

T = length(cnt);
ElnP_d = zeros(T,1);

bin_cnt1 = cnt;
bin_cnt0 = Tree_mat*cnt;
Elnbin1 = psi(bin_cnt1+gamma3) - psi(bin_cnt1+bin_cnt0+gamma3+gamma4);
Elnbin0 = psi(bin_cnt0+gamma4) - psi(bin_cnt1+bin_cnt0+gamma3+gamma4);

% % don't re-order weights
% stick_cnt = bin_cnt1+bin_cnt0;
% partition = unique(id_parent);
% for i = 1:length(partition)
%     idx = find(id_parent==partition(i));
%     t1 = stick_cnt(idx);
%     t3 = rev_cumsum(t1);
%     if length(t3) > 1
%         t4 = [t3(2:end) ; 0];
%         t5 = [0 ; psi(t4(1:end-1)+gamma2) - psi(t1(1:end-1)+t4(1:end-1)+1+gamma2)];
%     else
%         t4 = 0;
%         t5 = 0;
%     end
%     ElnP_d(idx) =  psi(t1+1) - psi(t1+t4+1+gamma2) + cumsum(t5);
% end
% this = ElnP_d + Elnbin1 + Tree_mat'*(Elnbin0 + ElnP_d);
% ElnP_d = this;

% re-order weights
stick_cnt = bin_cnt1+bin_cnt0;
partition = unique(id_parent);
for i = 1:length(partition)
    idx = find(id_parent==partition(i));
    t1 = stick_cnt(idx);
    [t1,idx_sort] = sort(t1,'descend');
    t3 = rev_cumsum(t1);
    if length(t3) > 1
        t4 = [t3(2:end) ; 0];
        t5 = [0 ; psi(t4(1:end-1)+gamma2) - psi(t1(1:end-1)+t4(1:end-1)+1+gamma2)];
    else
        t4 = 0;
        t5 = 0;
    end
    weights = psi(t1+1) - psi(t1+t4+1+gamma2) + cumsum(t5);
    ElnP_d(idx(idx_sort)) = weights;
end
this = ElnP_d + Elnbin1 + Tree_mat'*(Elnbin0 + ElnP_d);
ElnP_d = this;