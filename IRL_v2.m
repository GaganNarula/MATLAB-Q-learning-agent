%IRL Neu and Szepesvari

clc
close all

clear Ns Na P D pid_as Q theta phi mu 
Ns = 10;
Na = 2;
D = 10; %size of expansion
beta = 1;
Nepochs = 3000;
alpha = 0.2;
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


%posterior for every action and stimulus 
Q = zeros(Na,Ns);
P = zeros(Na,Ns);

cost = nan(1,Nepochs);
GRAD = nan(Nepochs,D);

for ii = 1:Nepochs
    
    for jj = 1:Ns
        x = reshape(phi(:,jj,:),Na,D);
        Q(:,jj) = theta'*x';
        
        P(:,jj) = exp(beta*Q(:,jj))/sum(exp(beta*Q(:,jj)));
    end
    
    %gradient
    cost(ii) = sum(sum(repmat(mu,Na,1).*( (P - pid_as).^2 )));
    
    for kk = 1:D
        grad = 0;
        for jj = 1:Na
            
            for ll = 1:Ns
                xk =  reshape(phi(jj,ll,kk),1,1);
                grad  = grad + mu(ll)*(P(jj,ll) - pid_as(jj,ll)) *P(jj,ll) *beta* ...
                    ( xk - sum(P(:,ll).*reshape(phi(:,ll,kk),Na,1)) );
            end
            
        end
        %update weights
        theta(kk) = theta(kk) - alpha*grad;    
        GRAD(ii,kk) = grad;
    end
    
end

for jj = 1:Ns
    Q(:,jj) = theta'*reshape(phi(:,jj,:),Na,D)';
    
    P(:,jj) = exp(beta*Q(:,jj))/sum(exp(beta*Q(:,jj)));
end
figure(1111);plot(pid_as(2,:),'-dr','LineWidth',2.4); hold on; plot(P(2,:),'b','LineWidth',2.4);
legend('Dem','Obs'); ylabel '\Delta Pesc';
figure(1112);plot(cost,'LineWidth',2.4);title 'IRL cost';
figure(1113);plot(Q(2,:),'-ob','LineWidth',2.4);
hold on;plot(Q(1,:),'-ok','LineWidth',2.4); title 'Final Q'; ylabel 'Action value';
xlabel 'Stimuli'; 
hold on;line([5.5 5.5],[-2 3],'Color','g');
legend('Q(escape,x)','Q(stay,x)','Class boundary');