%get_RBF_activation
%for any given stimulus vector s (1xD) , and a set of RBF centers mu
%output is phi, the RBF activations

function phi = get_RBF_activation(s,mu,sigmaa)

%dimensionality
D = size(mu,2);
%no. of rbfs
K = size(mu,1);

S = repmat(s,K,1);
dist = sum((S - mu).*(S - mu),2); %distance of s to each RBF kernel center
dist = -dist/sigmaa; %scaled by RBF scale param
phi = exp(dist);

end