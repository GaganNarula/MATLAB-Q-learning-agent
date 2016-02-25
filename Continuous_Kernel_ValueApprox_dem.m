%Continuous case
%Gaussian kernel regression for Q approximation for demonstrator
%One can also choose to have no kernel.
clc
close all
clear 

usekernel  = 1; %if zero, the data is kept as it is 
WTA  = 0;
%% create input
artific = 0;
if artific
M = 40; D = 2;
m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
[x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
    X = [X1;X2]; X = zscore(X); 
    x1 = X(1:size(X,1)/2,:); x2 = X(size(X,1)/2 + 1:end,:); %from my data
    M = size(x1,1); D = size(x1,2);
end

%% parameters
K = 70; %number of kernels 
sigmaa = 2; %RBF kernel length parameter
alpha = 0.3; %learning rate for function approximation
beta = 0.6; %inverse temperature

ntrials = 10000;
binsize = 500;

Pescape = [0.7;0.3]; %prior probability of escape

%Prior probability of puff class
Ppuff = 0.5;

%% transform data
[PHI,mu_ks] = get_kernel_rep(X,K,sigmaa);   %PHI is K x 2M matrix
%normalize?
PHI = PHI./repmat(sum(PHI),K,1);
%WTA?
if WTA
[~,I] = max(PHI);
for ii = 1:2*M
    pp = zeros(K,1);
    pp(I(ii)) = 1;
    PHI(:,ii) = pp;
end
end
figure(98);imagesc(PHI);ylabel 'Kernels';xlabel 'Stimuli';
%% useful data arrays
% N actions
Na = 2;

%Q function parameterized by theta
if usekernel
    theta = randn(Na,K);
else
    theta = randn(Na,D);
end
%theta = randn(Na,D);

%Probability of actions 
P = nan(Na,ntrials);

%rewards per trial
R = nan(1,ntrials);
S = zeros(2,M);
A = zeros(2,M); %count of escapes per stimulus

PESC = nan(ntrials/binsize,2*M); %Pescape matrix per binsize
dpnp = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken


%stimulus presented
stim = nan(2,ntrials);

l = 1; error = zeros(1,ntrials); 

for ii = 1:ntrials
    
    h = randi(M);
    %choose stimulus
     if rand > Ppuff
        %this is No puff (1) class
        c = 1; label = -1;
        %choose a stimulus s in Rd
        stim(1,ii) = h;

        s = x1(h,:);
        if ~usekernel
            %phi = [s,1;s,2]';
            phi = s';
        else
%             phi1 = [PHI(:,h);1]; phi2 = [PHI(:,h);2];   
%             phi = [phi1,phi2];  
            phi = PHI(:,h);
        end
       
    else
        c = 2; label = 1;
        stim(2,ii) = h;
                
        s = x2(h,:);
        if ~usekernel
            %phi = [s,1;s,2]';
            phi = s';
            
        else
%             phi1 = [PHI(:,h+M);1]; phi2 = [PHI(:,h+M);2];   
%             phi = [phi1,phi2];
           phi = PHI(:,h+M); 
        end

     end
    S(c,h) = S(c,h) + 1;
    
    
    %% create policy
    Qpred = theta*phi;
    %Qpred = sum(repmat(theta,2,1).*[[phi;1],[phi;2]]',2);
    P = exp(beta*Qpred).*Pescape;
    P = P/sum(P);
    
    %% take actions according to policy
    a(ii) = 1 + double(P(2) > rand);
    
    if a(ii)==2 
        A(c,h) = A(c,h)+1;
    end
    
    %% collect reward
    if (c==1 && a(ii)==2)  %this is escape when no puff
        
        r = -1;
    elseif (c==1 && a(ii)==1) % this is stay when no puff
        r = 1;
    elseif (c==2 && a(ii)==2) %this is escape when puff
        r = 1;
    elseif (c==2 && a(ii)==1)
        r = -1;             % this is stay when puff 
    end
    
    %rewards
    R(ii) = r;
    
    
    %% update theta
    %theta = theta + alpha*( r - Qpred(a(ii)) )*phi(:,a(ii));
    %theta = theta + alpha*( r - Qpred(a(ii)) )*[phi;a(ii)]';
    theta(a(ii),:) = theta(a(ii),:) + alpha*(r  - Qpred(a(ii))) *phi';
    error(ii) = abs(r - Qpred(a(ii)));
    %% for delpesc calculation
    if rem(ii,binsize)==0
        PESC(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l,1) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        A = zeros(2,M); S = A;
        l=l+1;
    end
        
end

figure(11);hold on;plot(cumsum(R),'r'); title 'Cumulative Rewards earned';
%figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Actions'; 
%xlabel 'Classes';
figure(33);imagesc(PESC');title 'Pesc(s)'; ylabel 'Stimulus ID'
figure(44);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc';
figure(55);plot(cumsum(error),'k','LineWidth',2.4); title 'Bellman error';

figure(33);imagesc(PESC'); ylabel 'Stimulus ID'; xlabel([num2str(binsize) ' trial bins']);
title(['Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ...
    ' P(a=1): ' num2str(Pescape(2)) 'K = ' num2str(K) '\sigma = ' num2str(sigmaa)]);

figure(44);hold on;plot(dpnp,'-dk','LineWidth',2.4);xlabel([num2str(binsize) ' trial bins']); ylim([-0.2 1]);
title(['\Delta Pesc  alpha: ' num2str(alpha) ' beta: ' num2str(beta) ...
    ' P(a=1): ' num2str(Pescape(2)) 'K = ' num2str(K) '\sigma = ' num2str(sigmaa)]);
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(66);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
title(['Pesc(stimulus) at end, alpha: ' num2str(alpha) ' beta: ' ...
    num2str(beta) ' P(a=1): ' num2str(Pescape(2)) ' K = ' num2str(K) '\sigma = ' num2str(sigmaa)]);


%% generalization

load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
Xgen = [X1;X2]; Xgen = zscore(Xgen);
x1v = Xgen(1:size(Xgen,1)/2,:); x2v = Xgen(size(Xgen,1)/2 +1 :end, :);
%get kernel activation
PHIgen = get_kernel_dist(Xgen,K,sigmaa,mu_ks);%normalize?
PHIgen = PHIgen./repmat(sum(PHIgen),K,1);
%WTA?
if WTA
[~,I] = max(PHIgen);
for ii = 1:2*M
    pp = zeros(K,1);
    pp(I(ii)) = 1;
    PHIgen(:,ii) = pp;
end
end
figure(99);imagesc(PHIgen);ylabel 'Kernels';xlabel 'Stimuli';

%rewards per trial
R = nan(1,ntrials);
S = zeros(2,M);
A = zeros(2,M); %count of escapes per stimulus

PESCgen = nan(ntrials/binsize,2*M); %Pescape matrix per binsize
dpnpgen = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken


%stimulus presented
stim = nan(2,ntrials);

l = 1; error = zeros(1,ntrials); 

for ii = 1:ntrials
    
    h = randi(M);
    %choose stimulus
     if rand > Ppuff
        %this is No puff (1) class
        c = 1; label = -1;
        %choose a stimulus s in Rd
        stim(1,ii) = h;

        s = x1v(h,:);
        if ~usekernel
            %phi = [s,1;s,2]';
            phi = s';
        else
%             phi1 = [PHI(:,h);1]; phi2 = [PHI(:,h);2];   
%             phi = [phi1,phi2];  
            phi = PHIgen(:,h);
        end
       
    else
        c = 2; label = 1;
        stim(2,ii) = h;
                
        s = x2v(h,:);
        if ~usekernel
            %phi = [s,1;s,2]';
            phi = s';
            
        else
%             phi1 = [PHI(:,h+M);1]; phi2 = [PHI(:,h+M);2];   
%             phi = [phi1,phi2];
           phi = PHIgen(:,h+M); 
        end

     end
    S(c,h) = S(c,h) + 1;
    
    
    %% create policy
    Qpred = theta*phi;
    %Qpred = sum(repmat(theta,2,1).*[[phi;1],[phi;2]]',2);
    P = exp(beta*Qpred).*Pescape;
    P = P/sum(P);
    
    %% take actions according to policy
    a(ii) = 1 + double(P(2) > rand);
    
    if a(ii)==2 
        A(c,h) = A(c,h)+1;
    end
    
    %% collect reward
    if (c==1 && a(ii)==2)  %this is escape when no puff
        
        r = -1;
    elseif (c==1 && a(ii)==1) % this is stay when no puff
        r = 1;
    elseif (c==2 && a(ii)==2) %this is escape when puff
        r = 1;
    elseif (c==2 && a(ii)==1)
        r = -1;             % this is stay when puff 
    end
    
    %rewards
    R(ii) = r;
    
    
    %% update theta
    %theta = theta + alpha*( r - Qpred(a(ii)) )*phi(:,a(ii));
    %theta = theta + alpha*( r - Qpred(a(ii)) )*[phi;a(ii)]';
    %theta(a(ii),:) = theta(a(ii),:) + alpha*(r  - Qpred(a(ii))) *phi';
    %theta(a(ii),:) = theta(a(ii),:) + alpha*(r  - Qpred(a(ii))) *phi';
    %regularized theta
    
    errorgen(ii) = abs(r - Qpred(a(ii)));
    %% for delpesc calculation
    if rem(ii,binsize)==0
        PESCgen(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnpgen(l,1) = nanmean(PESCgen(l,M + 1:end)) - nanmean(PESCgen(l,1:M));
        A = zeros(2,M); S = A;
        l=l+1;
    end
        
end

figure(111);hold on;plot(cumsum(R),'r'); title 'Generalization Cumulative Rewards earned';
%figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Actions'; 
%xlabel 'Classes';
%figure(133);imagesc(PESC');title 'Generalization  Pesc(s)'; ylabel 'Stimulus ID'
% figure(144);plot(dpnpgen,'-dk','LineWidth',2.4);title 'Generalization \Delta Pesc';
figure(155);plot(cumsum(errorgen),'k','LineWidth',2.4); title 'Generalization Bellman error';

figure(133);imagesc(PESCgen'); ylabel 'Stimulus ID'; xlabel([num2str(binsize) ' trial bins']);
title(['Generalization Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

figure(441);hold on;plot(dpnpgen,'-dk','LineWidth',2.4);xlabel([num2str(binsize) ' trial bins']); ylim([-0.2 1]);
title(['Generalization \Delta Pesc  alpha: ' num2str(alpha) ' beta: ' ...
    num2str(beta) ' P(a=1): ' num2str(Pescape(2))...
    'K = ' num2str(K) '\sigma = ' num2str(sigmaa)]);
tt = mean(PESCgen(end-5:end,:)); tt = tt(:);
figure(661);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
title(['Generalization Pesc(stimulus) at end, alpha: ' num2str(alpha) ...
    ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))...
    'K = ' num2str(K) '\sigma = ' num2str(sigmaa)]);

figure(662);plot([dpnp; dpnpgen],'-dk','LineWidth',2.4);
hold on;line([length(dpnp) length(dpnp)],[0 1],'Color','g');
title(['\Delta Pesc, training to generalization switch']);