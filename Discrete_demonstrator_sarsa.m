%Discrete case
%Demonstrator learns by Sarsa Q(a,s) <- Q(a,s) + alpha.(r - Q(a,s)) 
%takes actions according to   a ~ P(r=1|a,s)P(a) / Z

clc
close all
clear

%number of stimuli
Ns = 10;
%number of actions
Na = 2;

alpha = 0.002; %learning rate
beta = 1; %inverse temperature
Pescape = [0.7 0.3]; %prior probability of escape

ntrials = 10000; binsize = 100;

%Action value function
Q = zeros(Ns,Na);  %this is the q function Q
x = nan(1,ntrials); %chosen stimulus
a = nan(1,ntrials); %chosen action 

%reward per trial
R = nan(1,ntrials);

S = zeros(1,Ns);
A = zeros(1,Ns); %count of escapes per stimulus
l=1;

%Pesc
PESC = nan(ntrials/binsize,Ns);
dpnp = nan(ntrials/binsize,2);

for ii = 1:ntrials
    
    %choose a stimulus
    h = randi(Ns);
    
    x(ii) = h;
    S(h) = S(h)+1;
    
    
    %choose an action
    %P = exp(beta*Q(h,:))/sum(exp(beta*Q(h,:)));
    P = exp(beta*Q(h,:)).*Pescape;
    P = P/sum(P);
    a(ii) = 1 + double(P(2)>rand); %stochastic action in binary action case 
    %[~,a(ii)] = max(P);
    
    if a(ii)==2
        A(h) =A(h)+1;
    end
    
    %collect reward
    if (h <= Ns/2 && a(ii)==2)  %this is escape when no puff
        
        r = -1;
    elseif (h <= Ns/2 && a(ii)==1) % this is stay when no puff
        r = 1;
    elseif (h > Ns/2 && a(ii)==2) %this is escape when puff
        r = 1;
    elseif (h > Ns/2 && a(ii)==1)
        r = -1;             % this is stay when puff 
    end
    
    %rewards
    R(ii) = r;
    
    %Q iteration
    Q(h,a(ii)) = Q(h,a(ii)) + alpha*(r - Q(h,a(ii)));
    
    if rem(ii,binsize)==0
        PESC(l,:) = A./S;  %probability of escape
        dpnp(l,1) = mean(PESC(l,Ns/2 + 1:end)) - mean(PESC(l,1:Ns/2)); %delPesc
        A = zeros(1,Ns); S = A;
        l=l+1;
    end
    %figure(11);plot(sum(R(1:ii)),'-*b');title 'accumulated rewards';
    
end
figure(11);hold on;plot(cumsum(R),'r');title 'cumulative Rewards earned'; xlabel 'trials';
figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Stimulus ID';
xlabel 'Actions, 2 = escape, 1 = stay';

figure(33);imagesc(PESC'); ylabel 'Stimulus ID'; xlabel([num2str(binsize) ' trial bins']);
title(['Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

figure(44);hold on;plot(dpnp,'-dk','LineWidth',2.4);xlabel([num2str(binsize) ' trial bins']); ylim([-0.2 1]);
title(['\Delta Pesc  alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(55);plot([tt(1:Ns/2) ;  nan(Ns/2,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(Ns/2,1); tt(Ns/2 + 1:end)],'-dr','LineWidth',2.4); ylim([0 1]);