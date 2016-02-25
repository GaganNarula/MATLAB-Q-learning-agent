%IRL Neu and Szepesvari
%Discrete with RPROP

clc

clear Ns Na P D pid_as Q theta phi mu 
Ns = 10;
Na = 2;
D = 10; %size of expansion
beta = 2;
Nepochs = 2000;

%Target function

%initialize demonstrators policy 
%For two action case : index 1 is stay, 2 is leave
pid_as = nan(Na,Ns);
pid_as(1,:) = [0.9*ones(1,Ns/2),0.5*ones(1,Ns/2)];
pid_as(2,:) = 1 - pid_as(1,:);

%probability of a stimulus
mu = nan(1,Ns);
mu = [(0.75*2/Ns)*ones(1,Ns/2),(0.25*2/Ns)*ones(1,Ns/2)];

%choosing random basis for ( Ns x Na, D) representation
phi = randn(Na,Ns,D);
theta = randn(D,1); %parameters to optimize 
alpha = rand(D,1);
minal = 0.001;
maxal = 0.8; etaplus = 1.5; etaneg = 0.5;
prevalpha = alpha;

%posterior for every action and stimulus 
Q = zeros(Na,Ns);
P = zeros(Na,Ns);

cost = nan(1,Nepochs);
GRAD = nan(D,Nepochs);

for ii = 1:Nepochs
    
    for jj = 1:Ns
        x = reshape(phi(:,jj,:),Na,D);
        Q(:,jj) = theta'*x';
        
        P(:,jj) = exp(beta*Q(:,jj))/sum(exp(beta*Q(:,jj)));
    end
    
    %gradient
    cost(ii) = sum(sum(repmat(mu,Na,1).*( (P - pid_as).^2 )));
    grad = zeros(D,1);
    for kk = 1:D        
        for jj = 1:Na
            
            for ll = 1:Ns
                xk =  reshape(phi(jj,ll,kk),1,1);
                grad(kk)  = grad(kk) + mu(ll)*(P(jj,ll) - pid_as(jj,ll)) *P(jj,ll) *beta* ...
                    ( xk - sum(P(:,ll).*reshape(phi(:,ll,kk),Na,1)) );
            end
            
        end  
    end
    
    %rprop
    if ii>1
        g = GRAD(:,ii-1).*grad;
        pos = g > 0; neg = g < 0; eq = g == 0;
        alpha(pos) = etaplus*prevalpha(pos);
        alpha(neg) = etaneg*prevalpha(neg);
        alpha(eq) = prevalpha(eq);
        alpha(alpha > maxal) = maxal;
        alpha(alpha < minal) = minal;
        prevalpha = alpha;
    end
    
    %update weights
    theta = theta - alpha.*grad;
    GRAD(:,ii) = grad;
    
end

for jj = 1:Ns
    Q(:,jj) = theta'*reshape(phi(:,jj,:),Na,D)';
    
    P(:,jj) = exp(beta*Q(:,jj))/sum(exp(beta*Q(:,jj)));
end
figure(1111);plot(pid_as(2,:),'r'); hold on; plot(P(2,:),'b');
figure(1112);plot(cost)
figure(1113);plot(Q(2,:));
hold on;plot(Q(1,:),'k'); title 'Final Q'; ylabel 'Action value for escape';