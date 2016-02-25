%Demonstrator logistic regression on puff no puff
clc
clear
close all;


artific = 0;
eta = 0.2; alpha = 0.5; beta = 0.6;
sigma = 4;
if artific
MM = 80; D = 5;
m1 = 4; m2 = 6; shiftt = 3; s1 = 3; s2 = 4; correlatee = 0;
[x1,x2,~,~,~] = create_class_input(MM,D,m1,m2,shiftt,s1,s2,correlatee);
x1gen = x1(MM/2 + 1:MM,:); x2gen = x2(MM/2 + 1:MM,:);
x1 = x1(1:MM/2,:); x2 = x2(1:MM/2,:);
M = MM/2;
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
    %load('/Users/GaganN/Dropbox/PCA - LDA analysis/Training_X1X2_dursorted_SAP_ptracecorrect');
    X = [X1;X2];
    X = zscore(X); %X = [X,ones(size(X,1),1)];
    Xorig = X;
    M = size(X,1)/2; D = size(X,2);
    %Convert to radial basis function?
    if 1
    Xout = nan(2*M,2*M);
    for kk = 1:2*M
        %distance to each RBF kernel
        phi = exp( -( sum((X - repmat(X(kk,:),2*M,1)).* ...
            (X - repmat(X(kk,:),2*M,1)),2)) /(2*(sigma^2)) );
        Xout(kk,:) = phi;
    end
    X = Xout;
    end
    %Random high dimensional expansion?
    X = [X,ones(size(X,1),1)];
    M = size(X,1)/2; D = size(X,2);
end


%demonstrator policy (only for escapes)
Na = 2; Nclass = 2;
%logistic regression parameters
W = 0.5*randn(D,1); %last dimension is for bias

ntrials = 4000; Ppuff = 0.5;
binsize = 400;

Q = zeros(Na,Nclass); %Q function
Pa = [0.7;0.3]; % prior probability  of leaving 
A = zeros(1,2*M);
S = zeros(1,2*M);
R = nan(1,ntrials);
error = nan(1,ntrials);
a = nan(1,ntrials);
PESC = nan(ntrials/binsize,2*M);
dpnp = nan(1,ntrials/binsize);

l = 1;
for ii = 1:ntrials
    %choose a class 
    if rand > Ppuff
        %no puff
        h = randi(M);
        x = X(h,:);
        c = 1; %class
    else
        %puff
        h = M + randi(M);
        x = X(h,:);
        c = 2;
    end
    S(h) = S(h) + 1;
    
    %make a class prediction
    P_chat = 1/(1 + exp(-W'*x'));
    chat = 1 + double(P_chat > 0.5); %deterministic choice of class, 2: Puff
    
    %make an action
    P = exp(beta*Q(:,chat)).*Pa;
    P = P/sum(P);
    a(ii) = 1 + double(P(2) > rand); %stochastic action 
    
    if a(ii)==2
        A(h) = A(h) + 1;
    end
    
    %collect reward
      %get reward
    if (c==1 && a(ii)==2)  %this is escape when no puff
        
        r = -1;
    elseif (c==1 && a(ii)==1) % this is stay when no puff
        r = 1;
    elseif (c==2 && a(ii)==2) %this is escape when puff
        r = 1;
    elseif (c==2 && a(ii)==1)
        r = -1;             % this is stay when puff 
    end
    
    %save rewards
    R(ii) = r;
       
    %update classifier weights
    W = W + eta*((c-1) - P_chat)*P_chat*(1 - P_chat)*x';
    error(ii) = (c - chat)^2;
    
    %update Q
    Q(a(ii),chat) = Q(a(ii),chat) + alpha*(r - Q(a(ii),chat));
    
    if rem(ii,binsize)==0
        PESC(l,:) = A./S;  %probability of escape
        dpnp(l) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        A = zeros(1,2*M); S = A;
        l=l+1;
    end
    fprintf('\n .... %f Percent done .... \n',ii*100/ntrials);
end

figure(334);imagesc(PESC');title 'Training Pesc(s)'; ylabel 'Stimulus ID';
figure(444);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc Train';
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(666);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
title 'Pesc Train';

%generalization
if artific
% M = 40; D = 2;
% m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
% [x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
    %load('/Users/GaganN/Dropbox/PCA - LDA analysis/Generalize_X1X2_dursorted_SAP_ptracecorrect');
    Xgen = [X1;X2]; Xgen = zscore(Xgen); 
    %Convert to radial basis function?
    if 1
    Xout = nan(2*M,2*M);
    for kk = 1:2*M
        %distance to each RBF kernel 
        phi = exp( -( sum((Xorig - repmat(Xgen(kk,:),2*M,1)).* ...
            (Xorig - repmat(Xgen(kk,:),2*M,1)),2)) /(2*(sigma^2)) );
        Xout(kk,:) = phi;
    end
    Xgen = Xout; 
    end
    Xgen = [Xgen,ones(size(Xgen,1),1)];
    M = size(Xgen,1)/2; D = size(Xgen,2);
end

ntrials = 4000;
binsize = 400;
Ppuff = 0.5;

A = zeros(1,2*M);
S = zeros(1,2*M);
a = nan(1,ntrials);
PESC = nan(ntrials/binsize,2*M);
dpnp = nan(1,ntrials/binsize);

l = 1;
for ii = 1:ntrials
    %choose a class 
    if rand > Ppuff
        %no puff
        h = randi(M);
        x = Xgen(h,:);
        c = 1; %class
    else
        %puff
        h = M + randi(M);
        x = Xgen(h,:);
        c = 2;
    end
    S(h) = S(h) + 1;
    
    %choose action
    %make a class prediction
    P_chat = 1/(1 + exp(-W'*x'));
    chat = 1 + double(P_chat > 0.5); %deterministic choice of class, 2: Puff
    
    %make an action
    P = exp(beta*Q(:,chat)).*Pa;
    P = P/sum(P);
    a(ii) = 1 + double(P(2) > rand);
    
    if a(ii) == 2
        A(h) = A(h) + 1;
    end
    
    if rem(ii,binsize)==0
        PESC(l,:) = A./S;%[A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        A = zeros(1,2*M); S = A;
        l=l+1;
    end
    
    
end


figure(33);imagesc(PESC');title 'Generalization Pesc(s)'; ylabel 'Stimulus ID'
figure(44);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc Gen';
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(66);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
