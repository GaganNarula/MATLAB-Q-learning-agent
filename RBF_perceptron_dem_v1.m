%This script does the following transformation :
% x -> phi(x) -> prediction of puff class through a perceptron
% Specifics:
% phi(x) is a Kernel function. An RBF network where each component basis of
% phi(x) {phi1, phi2 .. phiK} is an RBF kernel with particular mean, but
% fixed and constant scale parameter sigma 

% the bird uses reward to change the centers of the kernels 

clc
close all
clear 

usekernel  = 1;
rbf_norm = 1;
WTA = 0;
%% create input
artific = 0;
if artific
M = 40; D = 2;
m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
[x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
    %load('/Users/GaganN/Dropbox/PCA - LDA analysis/Training_X1X2_dursorted_SAP_ptracecorrect.mat');
    
    %standardize X
    X = [X1;X2]; 
    X = zscore(X); 
    
    x1 = X(1:size(X,1)/2,:); x2 = X(size(X,1)/2 + 1:end,:); %from my data
    M = size(x1,1); D = size(x1,2);
end


%% parameters
K = 30; %number of kernels 
sigmaa = 4; %RBF kernel scale parameter (like standard deviation)
mu = randn(K,D); %initialize kernel centers in D dimensions 
alpha = 0.1; %learning rate for TD learning 
beta = 0.65; %inverse temperature of policy
eta = 0.1; %learning rate for perceptron
ntrials = 10000; %total number of trials
binsize = 200;

Pescape = [0.5;0.5]; %Prior of escape
Ppuff = 0.25; %probability of puff 

%% useful data arrays
% N actions
Na = 2;

%parameters for perceptron (+1 dimension for bias)
theta = randn(K+1,1);

%Q function
Q = zeros(Na,2); % 2 class function

%Probability of actions 
P = nan(Na,ntrials);


%rewards per trial
R = nan(1,ntrials);
S = zeros(2,M);
A = zeros(2,M); %count of escapes per stimulus

PESC = nan(ntrials/binsize,2*M); %Pescape matrix per binsize
dpnp = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken
Pval = nan(ntrials/binsize,1); % P values of significance test between puff and no puff 

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
        stim(c,ii) = h;

        s = x1(h,:); %stimulus from class no puff
        
        if ~usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,sigmaa);
            %normalize activation?
            if rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if WTA 
                [~,I] = max(phi);
                phi = zeros(K,1);
                phi(I) = 1;                
            end
            
        end
       
    else
        c = 2; label = 1;
        stim(c,ii) = h;
                
        s = x2(h,:);
        
        if ~usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,sigmaa);
            %normalize activation?
            if rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if WTA 
                [~,I] = max(phi);
                phi = zeros(K,1);
                phi(I) = 1;                
            end
            
        end

     end
    S(c,h) = S(c,h) + 1;
    
    
    %predicted class of stimulus
    y = sign(theta'*[phi;1]);
    if y>=0
        Chat(ii) = 1; %predicted label
        Qind = 2;
    else
        Chat(ii) = -1; %predicted label
        Qind = 1;
    end
    
    %use Qind to make action
     %use predicted class to make probability of action estimate
    
    Rterm =  exp(beta*Q(:,Qind));
    p = Rterm.*Pescape;
    P(:,ii) = p/sum(p);
    %choose an action
    %deterministic policy
    %[~,a(ii)] = max(P(:,ii));
    %stochastic action
    a(ii) = 1 + double(P(2,ii) > rand);
    
    if a(ii)==2 
        A(c,h) = A(c,h)+1;
    end
        
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
    
    %update weights if error
    if (Chat(ii)~=label)
        error(ii) = 1;
        theta = theta + eta*[phi;1]*label;
    end
    
    %update Q
    Q(a(ii),Qind) = Q(a(ii),Qind) + alpha*(r - Q(a(ii),Qind));
    
    
    if rem(ii,binsize)==0
        PESC(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l,1) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        PuffProb = nanmean(PESC(l,M + 1:end)); NoPuffProb = nanmean(PESC(l,1:M));
        [~,~,~,Pval(l),~] = gn_fa_meta_airpuff_sigtest_binom(PuffProb, ...
            NoPuffProb,[sum(S(1,:)),sum(S(2,:))],0.01);
        A = zeros(2,M); S = A;
        l=l+1;
    end
   fprintf('\n .. Training %f percent done .. \n',ii*100/ntrials);
   if ii > 50
   fprintf('\n .. Perceptron loss : %f .. \n',mean(error(ii-50)));
   end
end
%smooth error
binsiz2 = 50;
Err = nan(1,floor(ntrials/binsiz2));
l = 1;
for kk = 1:length(Err);
    Err(kk) = mean(error(l:l+binsiz2-1));
    l = l+binsiz2;
end

figure(11);hold on;plot(cumsum(R),'r'); title 'Cumulative Rewards earned';
%figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Actions'; 
%xlabel 'Classes';
figure(33);imagesc(PESC');title 'Pesc(s)'; ylabel 'Stimulus ID'
figure(44);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc';
figure(55);plot(cumsum(error),'k','LineWidth',2.4); title 'cumulative Label error';


%% get criterion on training delpesc
PVAL_sig = Pval < 0.01; %significant p values
windoww = 8;
thresh = 0.9;
binsize = 200;
gn_get_bird_out_trialstocrit;
Tcrit_train = trialstocrit;

%% generalization


% Mgen = 20;
% [x1v,x2v] = create_class_input(Mgen,D);
if 0
M = 40; D = 2;
m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
[x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
    %load('/Users/GaganN/Dropbox/PCA - LDA analysis/Generalize_X1X2_dursorted_SAP_ptracecorrect.mat');
    
    Xgen = [X1;X2]; Xgen = zscore(Xgen); 
    x1v = Xgen(1:size(Xgen,1)/2,:); x2v = Xgen(size(Xgen,1)/2 + 1 :end,:); %from my data
    
    Mgen = size(x1v,1); D = size(x1,2);
end
ntrials = 5000;

%rewards per trial
Rgen = nan(1,ntrials); 

%predicted class 
Chat = nan(1,ntrials);
S = zeros(2,Mgen);
A = zeros(2,Mgen); %count of escapes per stimulus
PESCgen = nan(ntrials/binsize,2*Mgen);
dpnp_gen = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken
error_gen = zeros(1,ntrials);
Pval_gen = nan(ntrials/binsize,1);

l =1;
for ii = 1:ntrials
    
    h = randi(Mgen);
    %choose a class
    if rand > Ppuff
        %this is No puff (1) class
        c = 1; label = -1;
        %choose a stimulus s in Rd
        %X(1,ii) = h;

        s = x1v(h,:);
        if ~usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,sigmaa);
            %normalize activation?
            if rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if WTA 
                [~,I] = max(phi);
                phi = zeros(K,1);
                phi(I) = 1;                
            end
            
        end
    else
        c = 2; label = 1;
        %X(2,ii) = h;
                
        s = x2v(h,:);
        if ~usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,sigmaa);
            %normalize activation?
            if rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if WTA 
                [~,I] = max(phi);
                phi = zeros(K,1);
                phi(I) = 1;                
            end
            
        end
    end
    S(c,h) = S(c,h) + 1;
    
    
    %predicted class of stimulus
    y = sign(theta'*[phi;1]);
    if y>=0
        Chat(ii) = 1;
        Qind = 2;
    else
        Chat(ii) = -1;
        Qind = 1;
    end
    
    
    %use predicted class to make probability of action estimate
    P(:,ii) = exp(beta*Q(:,Qind))/sum(exp(beta*Q(:,Qind)));
    
    %choose an action
    %deterministic policy
    %[~,a(ii)] = max(P(:,ii));
    %stochastic action
    a(ii) = 1 + double(P(2,ii) > rand);
    
    if a(ii)==2 
        A(c,h) = A(c,h)+1;
    end
        
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
    Rgen(ii) = r;
    
    %errors
    if Chat(ii)~= label
        error_gen(ii) = 1;
        %W = W + eta*s'*label;
        theta = theta + eta*[phi;1]*label;
    end
        
    
    if rem(ii,binsize)==0
        PESCgen(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp_gen(l,1) = nanmean(PESCgen(l,Mgen + 1:end)) - nanmean(PESCgen(l,1:Mgen));
        PuffProb = nanmean(PESCgen(l,M + 1:end)); NoPuffProb = nanmean(PESCgen(l,1:M));
        [~,~,~,Pval_gen(l),~] = gn_fa_meta_airpuff_sigtest_binom(PuffProb, ...
            NoPuffProb,[sum(S(1,:)),sum(S(2,:))],0.01);
        A = zeros(2,Mgen); S = A;
        l=l+1;
    end
end
%smooth error
binsiz2 = 50;
Errgen = nan(1,floor(ntrials/binsiz2));
l = 1;
for kk = 1:length(Errgen);
    Errgen(kk) = mean(error_gen(l:l+binsiz2-1));
    l = l+binsiz2;
end
figure(111);hold on;plot(cumsum(Rgen),'r'); title 'Generalization Cumulative Rewards earned';
figure(133);imagesc(PESCgen');title 'Generalization Pesc(s)'; ylabel 'Stimulus ID'
xlabel(['Bins of ' num2str(binsize) ' trials']);
figure(144);plot(dpnp_gen,'-dk','LineWidth',2.4);title 'Generalization \Delta Pesc';
xlabel(['Bins of ' num2str(binsize) ' trials']);
figure(155);plot(Errgen,'k','LineWidth',2.4); 
title 'Generalization Classification error';
% tt = mean(PESCgen(end-1:end,:)); tt = tt(:);
% figure(166);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
% hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
% title(['Pesc(stimulus) at end, alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);
figure(199);plot([dpnp;dpnp_gen],'-dk');ylim([0 1])
hold on;line([length(dpnp) length(dpnp)],[0 1]); xlabel(['Bins of ' num2str(binsize) ' trials']);
ylabel '\Delta Pesc';


%% get criterion on generalization  delpesc
PVAL_sig = Pval_gen < 0.01; %significant p values
windoww = 8;
thresh = 0.9;
binsize = 200;
gn_get_bird_out_trialstocrit;
Tcrit_gen = trialstocrit;