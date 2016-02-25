%Continuous case mapping to Kernel
%Testing generalization
% s -> phi(s) where phi(s) is RBF kernel placed on the number of training
% stimuli
% phi(s) -> normalized Winner Take All psi(s)
% a ~  exp(beta*Q(a,psi(s))*Prior(a) / Z

clc
clear
close all

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

sigma = 8; %RBF kernel length parameter
alpha = 0.1; %learning rate for function approximation
beta = 0.7; %inverse temperature

ntrials = 10000;
binsize = 500;

Pescape = [0.7;0.3]; %prior probability of escape

%Prior probability of puff class
Ppuff = 0.5;
K = 100;
PHI = get_kernel_rep(X,K,sigma);


%% useful data arrays
% N actions
Na = 2;

%Q function parameterized by theta
%theta = randn(2,M);
%theta = randn(Na,D);
Qpred = zeros(Na,M);

%Probability of actions 
P = nan(Na,ntrials);

%rewards per trial
R = nan(1,ntrials);
S = zeros(2,M);
A = zeros(2,M); %count of escapes per stimulus

PESC = nan(ntrials/binsize,2*M); %Pescape matrix per binsize
dpnp = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken

%PHI MATRIX
PHI1 = nan(M,ntrials);
PHI2 = nan(M,ntrials);

l = 1; error = zeros(1,ntrials); g=1; j = 1;

%% loop over trials
for ii = 1:ntrials
    
    h = randi(M);
    %choose stimulus
     if rand > Ppuff
        %this is No puff (1) class
        c = 1; label = -1;

        s = x1(h,:);
        %phi = s';
        
        %%measure similarity of the point to each of the other points       
        phi = exp(-(x1 - repmat(s,size(x1,1),1)).^2 / 2*sigma^2);
        phi = phi/sum(phi);
        %make it a WTA
        [~,I] = min(phi);
        psi = phi*0; psi(I) = 1; 
        PHI1(:,g) = psi;
        g =g+1;
    else
        c = 2; label = 1;
        
        s = x2(h,:);
        %phi = s';
        %%measure similarity of the point to each of the other points
        phi = exp(-(x1 - repmat(s,size(x1,1),1)).^2 / 2*sigma^2);
        phi = phi/sum(phi);
        %make it a WTA
        [~,I] = min(phi);
        psi = phi*0; psi(I) = 1; 
        PHI2(:,j) = psi;
        j = j+1;
         
     end
    S(c,h) = S(c,h) + 1;
    
    
    %% create policy
    %Qpred = theta*phi;
    %Qpred = sum(repmat(theta,2,1).*[[phi;1],[phi;2]]',2);
    P = exp(beta*Qpred(:,I)).*Pescape;
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
    %theta(a(ii),:) = theta(a(ii),:) + alpha*( r - Qpred(a(ii)) )*phi';
    %theta = theta + alpha*( r - Qpred(a(ii)) )*[phi;a(ii)]';
    Qpred(a(ii),I) = Qpred(a(ii),I) + alpha*(r - Qpred(a(ii),I));
    %error(ii) = abs(r - Qpred(a(ii)));
    %% for delpesc calculation
    if rem(ii,binsize)==0
        PESC(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l,1) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        A = zeros(2,M); S = A;
        l=l+1;
    end
        
end

PHI1 = PHI1(:,1:g-1); PHI2 = PHI2(:,1:j-1);


figure(11);hold on;plot(cumsum(R),'r'); title 'Cumulative Rewards earned';
figure(22);hold on;imagesc(Qpred);colormap(hot); title 'final Q'; ylabel 'Actions'; 
%xlabel 'Classes';
figure(33);imagesc(PESC');title 'Pesc(s)'; ylabel 'Stimulus ID'
figure(44);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc';
%figure(55);plot(cumsum(error),'k','LineWidth',2.4); title 'Bellman error';

figure(33);imagesc(PESC'); ylabel 'Stimulus ID'; xlabel([num2str(binsize) ' trial bins']);
title(['Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

figure(44);hold on;plot(dpnp,'-dk','LineWidth',2.4);xlabel([num2str(binsize) ' trial bins']); ylim([-0.2 1]);
title(['\Delta Pesc  alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(66);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
title(['Pesc(stimulus) at end, alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);




%generalization

load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
X = [X1;X2]; X = zscore(X);
    
for ii = 1:ntrials
    
    h = randi(M);
    %choose stimulus
     if rand > Ppuff
        %this is No puff (1) class
        c = 1; label = -1;

        s = x1(h,:);
        %phi = s';
        
        %%measure similarity of the point to each of the other points       
        phi = exp(-(x1 - repmat(s,size(x1,1),1)).^2 / 2*sigma^2);
        phi = phi/sum(phi);
        %make it a WTA
        [~,I] = min(phi);
        psi = phi*0; psi(I) = 1; 
        PHI1(:,g) = psi;
        g =g+1;
    else
        c = 2; label = 1;
        
        s = x2(h,:);
        %phi = s';
        %%measure similarity of the point to each of the other points
        phi = exp(-(x1 - repmat(s,size(x1,1),1)).^2 / 2*sigma^2);
        phi = phi/sum(phi);
        %make it a WTA
        [~,I] = max(phi);
        psi = phi*0; psi(I) = 1; 
        PHI2(:,j) = psi;
        j = j+1;
         
     end
    S(c,h) = S(c,h) + 1;
    
    
    %% create policy
    %Qpred = theta*phi;
    %Qpred = sum(repmat(theta,2,1).*[[phi;1],[phi;2]]',2);
    P = exp(beta*Qpred(:,I)).*Pescape;
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
    
 
    %% for delpesc calculation
    if rem(ii,binsize)==0
        PESC(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l,1) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        A = zeros(2,M); S = A;
        l=l+1;
    end
        
end

PHI1 = PHI1(:,1:g-1); PHI2 = PHI2(:,1:j-1);


figure(111);hold on;plot(cumsum(R),'r'); title 'Cumulative Rewards earned';
%figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Actions'; 
%xlabel 'Classes';
figure(333);imagesc(PESC'); ylabel 'Stimulus ID'; xlabel([num2str(binsize) ' trial bins']);
title(['Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

figure(444);hold on;plot(dpnp,'-dk','LineWidth',2.4);xlabel([num2str(binsize) ' trial bins']); ylim([-0.2 1]);
title(['\Delta Pesc  alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(666);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
title(['Pesc(stimulus) at end, alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);


    