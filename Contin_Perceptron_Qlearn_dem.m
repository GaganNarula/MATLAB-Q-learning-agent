%Q learning in continuous space with classification

% Demonstrator takes actions according too:
% a ~ P(r=1|a,C(s))P(a) / Z
% P(r=1|a,C(s)) = exp(beta*Q(a,C(s))
% C(s) = sign(W*s) class of stimulus s - {0,1}
%
% W = W + eta*s*label  if a mistake is made on the given trial

clc
close all

%% create stimulus

if 0
M = 40; D = 2;
m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
[x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
    X = [X1;X2]; X = zscore(X); X = [X,ones(size(X,1),1)];
    x1 = X(1:size(X,1)/2,:); x2 = X(size(X,1)/2 + 1 :end,:); %from my data
    
    M = size(x1,1); D = size(x1,2);
end
%% set params
%Prior probability of puff class
Ppuff = 0.5;

%prior probability of escape
Pescape = [0.7;0.3];

%RATES
alpha = 0.002; %Qlearning rate 
beta = 0.6; %inverse temperature
eta = 0.005; %learning rate of perceptron

ntrials = 8000;
binsize = 400;

% classes
Nc = 2;
% N actions
Na = 2;

%% initialize data arrays
%Q function
Q = zeros(Na,Nc);
%Probability of actions 
P = nan(Na,ntrials);
%rewards per trial
R = nan(1,ntrials);
S = zeros(2,M);
A = zeros(2,M); %count of escapes per stimulus
PESC = nan(ntrials/binsize,2*M);
dpnp = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken




%stimulus presented
X = nan(2,ntrials);
%predicted class 
Chat = nan(1,ntrials);

%initialize Weight vector for classifier
%W = -1 + 2*rand(D,1); 
W = randn(D,1);
l = 1; error = zeros(1,ntrials);
for ii = 1:ntrials
    
    %h = randi(M);
    tmp = randperm(M);
    h = tmp(1);
    %choose a class
    if rand > Ppuff
        %this is No puff (1) class
        c = 1; label = -1;
        %choose a stimulus s in Rd
        X(1,ii) = h;

        s = x1(h,:);
    else
        c = 2; label = 1;
        X(2,ii) = h;
                
        s = x2(h,:);
    end
    S(c,h) = S(c,h) + 1;
    
    
    %predicted class of stimulus
    y = sign(s*W);
    if y>=0
        Chat(ii) = 1;
        Qind = 2;
    else
        Chat(ii) = -1;
        Qind = 1;
    end
    
    
    %use predicted class to make probability of action estimate
    %P(:,ii) = exp(beta*Q(:,Qind))/sum(exp(beta*Q(:,Qind)));
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
        W = W + eta*s'*label;
    end
    
    %update Q
    Q(a(ii),Qind) = Q(a(ii),Qind) + alpha*(r - Q(a(ii),Qind));
    
    if rem(ii,binsize)==0
        PESC(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l,1) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        A = zeros(2,M); S = A;
        l=l+1;
    end
        
end

%smooth error
Err = nan(1,floor(ntrials/binsize));
l = 1;
for kk = 1:length(Err);
    Err(kk) = mean(error(l:l+binsize-1));
    l = l+binsize;
end
figure(11);hold on;plot(cumsum(R),'r'); title 'Cumulative Rewards earned';
figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Actions'; 
xlabel 'Classes';
figure(33);imagesc(PESC');title 'Pesc(s)'; ylabel 'Stimulus ID'
figure(44);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc';
figure(55);plot(Err,'k','LineWidth',2.4); title 'Classification error';

figure(33);imagesc(PESC'); ylabel 'Stimulus ID'; xlabel([num2str(binsize) ' trial bins']);
title(['Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

figure(44);hold on;plot(dpnp,'-dk','LineWidth',2.4);xlabel([num2str(binsize) ' trial bins']); ylim([-0.2 1]);
title(['\Delta Pesc  alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2)) ' eta: ' num2str(eta)]);
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(66);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
title(['Pesc(stimulus) at end, alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

%% generalization
if 1
% Mgen = 20;
% [x1v,x2v] = create_class_input(Mgen,D);
if 0
M = 40; D = 2;
m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
[x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
    Xgen = [X1;X2]; Xgen = zscore(Xgen); Xgen = [Xgen,ones(size(Xgen,1),1)];
    x1v = Xgen(1:size(Xgen,1)/2,:); x2v = Xgen(size(Xgen,1)/2 + 1 :end,:); %from my data
    
    Mgen = size(x1v,1); D = size(x1,2);
end
ntrials = 2000;

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
    else
        c = 2; label = 1;
        %X(2,ii) = h;
                
        s = x2v(h,:);
    end
    S(c,h) = S(c,h) + 1;
    
    
    %predicted class of stimulus
    y = sign(s*W);
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
    end
        
    
    if rem(ii,binsize)==0
        PESCgen(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp_gen(l,1) = nanmean(PESCgen(l,Mgen + 1:end)) - nanmean(PESCgen(l,1:Mgen));
        A = zeros(2,Mgen); S = A;
        l=l+1;
    end
end
%smooth error
Errgen = nan(1,floor(ntrials/binsize));
l = 1;
for kk = 1:length(Errgen);
    Errgen(kk) = mean(error_gen(l:l+binsize-1));
    l = l+binsize;
end
figure(111);hold on;plot(cumsum(Rgen),'r'); title 'Generalization Cumulative Rewards earned';
figure(133);imagesc(PESCgen');title 'Generalization Pesc(s)'; ylabel 'Stimulus ID'
figure(144);plot(dpnp_gen,'-dk','LineWidth',2.4);title 'Generalization \Delta Pesc';
figure(155);plot(Errgen,'k','LineWidth',2.4); title 'Generalization Classification error';
% tt = mean(PESCgen(end-1:end,:)); tt = tt(:);
% figure(166);plot([tt(1:M) ;  nan(M,1)],'-db','LineWidth',2.4); 
% hold on; plot([nan(M,1); tt(M:end)],'-dr','LineWidth',2.4); ylim([0 1]);
% title(['Pesc(stimulus) at end, alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

if 0
    h1 = figure(33);
    saveas(h1,'C:\Users\songbird\Dropbox\Q learning agent\PESC','bmp');
    h2 = figure(44);
    saveas(h2,'C:\Users\songbird\Dropbox\Q learning agent\dpnp','bmp');
    h3 = figure(55);
    saveas(h3,'C:\Users\songbird\Dropbox\Q learning agent\Pesc(stimulus)','bmp');
end
end