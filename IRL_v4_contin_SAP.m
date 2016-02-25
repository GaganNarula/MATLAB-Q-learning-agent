%IRL Neu and Szepesvari
%with continuous input 
%input mapped to kernel space

clc

clear Ns Na P D pid_as Q theta phi mu 


load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect.mat')
X = [X1;X2]; X = zscore(X);

Ns = size(X,1);
Na = 2;
%Projection to higher dimension?
K = 100;
J = (1/sqrt(Ns))*randn(K,size(X,2));
X = J*X'; X = X';

Ns = size(X,1);

D = size(X,2) + 1; %size of theta
beta = 1;
Nepochs = 3000;
alpha = 0.5;




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
phi = cell(Na,1); phi{1,1} = [X,ones(Ns,1)]; 
phi{2,1} = [X,2*ones(Ns,1)];

%stimulus
theta = randn(D,1); %parameters to optimize 


%posterior for every action and stimulus 
Q = zeros(Na,Ns);
P = zeros(Na,Ns);

cost = nan(1,Nepochs);
GRAD = nan(Nepochs,D);

for ii = 1:Nepochs
    
    for jj = 1:Ns
        x = [phi{1,1}(jj,:);phi{2,1}(jj,:)];
        Q(:,jj) = theta'*x';
        
        P(:,jj) = exp(beta*Q(:,jj))/sum(exp(beta*Q(:,jj)));
    end
    
    %gradient
    cost(ii) = sum(sum(repmat(mu,Na,1).*( (P - pid_as).^2 )));
    
    for kk = 1:D
        grad = 0;
        for jj = 1:Na
            
            for ll = 1:Ns
                xk = phi{jj,1}(Ns,kk);
                grad  = grad + mu(ll)*(P(jj,ll) - pid_as(jj,ll)) *P(jj,ll) *beta* ...
                    ( xk - sum(P(:,ll).*  [phi{1,:}(jj,kk);phi{2,:}(jj,kk)]) );
            end
            
        end
        
        %update weights
        
        theta(kk) = theta(kk) - alpha*grad;    
        GRAD(ii,kk) = grad;
    end
    fprintf('\n ... %f percent done ... \n',ii*100/Nepochs); 
end

for jj = 1:Ns
    x = [phi{1,1}(jj,:);phi{2,1}(jj,:)];
    Q(:,jj) = theta'*x';
    
    P(:,jj) = exp(beta*Q(:,jj))/sum(exp(beta*Q(:,jj)));
end
figure(1111);plot(pid_as(2,:),'r'); hold on; plot(P(2,:),'b');
figure(1112);plot(cost)