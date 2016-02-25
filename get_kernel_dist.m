%get kernel distances to query points

function PHI = get_kernel_dist(X,K,sigmaa,mu_ks)
M = size(X,1);
PHI = nan(K,M);
for ii = 1:M
    PHI(:,ii) = exp(- sum((mu_ks - repmat(X(ii,:),K,1)).*  ...
        (mu_ks - repmat(X(ii,:),K,1)),2) / (2*(sigmaa^2)));
end