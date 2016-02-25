%observer logistic regression on demonstrator 
clc
clear
close all;
artific = 0;

sigma = 4;   %Kernel variance
alpha = 0.5;  %learning rate of batch logistic regression 
beta = 0.6;   %inverse temperature of Observer policy
cycles = 300; %number of epochs of batch logistic regression

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
    X = zscore(X); % X = [X,ones(size(X,1),1)];
    Xorig = X;
    M = size(X,1)/2; D = size(X,2);
    
    %Convert to radial basis function?
    if 0
    Xout = nan(2*M,2*M);
    for kk = 1:2*M
        %distance to each RBF kernel 
        phi = exp( -( sum((X - repmat(X(kk,:),2*M,1)).* ...
            (X - repmat(X(kk,:),2*M,1)),2)) /(2*(sigma^2)) );
        Xout(kk,:) = phi;
    end
    X = Xout;
    end
    % 
    X = [X,ones(size(X,1),1)];
    M = size(X,1)/2; D = size(X,2);
end


%demonstrator policy (only for escapes)
%Pdem_as = [0.1*ones(1,M),0.5*ones(1,M)];
Pdem_as = [(0.1 + 0.0*randn(1,M)),(0.5 + 0.0*randn(1,M))];
%regression
W = 0.5*randn(D,1); %last dimension is for bias

error = nan(1,cycles);
for ii = 1:cycles
        Phat_as = 1./(1+exp(-beta*(W'*X')));
        gradterm = (Pdem_as - Phat_as).*Phat_as.*(1-Phat_as);
        grad = repmat(gradterm',1,D).*X;
        totgrad = mean(grad);
        %update W
        W = W + alpha*beta*totgrad';
        error(ii) = mean((Pdem_as - Phat_as).^2);
end
Phat_as = 1./(1+exp(-beta*(W'*X')));
figure(11);plot(error,'LineWidth',2.4); title 'Mean Squared regression error';
figure(22);plot(Pdem_as,'r','LineWidth',2.4);
hold on;plot(Phat_as,'b','LineWidth',2.4);
legend('Demonstrator','Observer'); title 'Pesc curves';


%training time
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
        x = X(h,:);
        c = 1; %class
    else
        %puff
        h = M + randi(M);
        x = X(h,:);
        c = 2;
    end
    S(h) = S(h) + 1;
    
    %choose action
    P = 1/(1+exp(-beta*W'*x'));
    %figure(88);hold on;plot(P);
    a(ii) = 1 + double(P > rand);
    
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

figure(3);imagesc(PESC');title 'Generalization Obs Pesc(s)'; ylabel 'Stimulus ID'
figure(4);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc Obs';
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(6);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);



%% generalization
if artific
% M = 40; D = 2;
% m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
% [x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    %load('/Users/GaganN/Dropbox/PCA - LDA analysis/Generalize_X1X2_dursorted_SAP_ptracecorrect');
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
    Xgen = [X1;X2]; Xgen = zscore(Xgen); 
    %Xgen = [Xgen,ones(size(Xgen,1),1)];
    M = size(Xgen,1)/2; D = size(Xgen,2);
    %Convert to radial basis function?
    if 0
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
    M = size(X,1)/2; D = size(Xgen,2);
end

ntrials = 4000;
binsize = 400;
Ppuff = 0.5;
alpha2 = 0.2;
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
    P = 1/(1+exp(-beta*W'*x'));
    %figure(88);hold on;plot(P);
    a(ii) = 1 + double(P > rand);
    
    if a(ii) == 2
        A(h) = A(h) + 1;
    end
    
    W = W + alpha2*beta*(c-1 - P)*P*(1-P)*x';
    
    if rem(ii,binsize)==0
        PESC(l,:) = A./S;%[A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        A = zeros(1,2*M); S = A;
        l=l+1;
    end
    
    
end


figure(335);imagesc(PESC');title 'Generalization Obs Pesc(s)'; ylabel 'Stimulus ID'
figure(445);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc Obs';
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(665);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
