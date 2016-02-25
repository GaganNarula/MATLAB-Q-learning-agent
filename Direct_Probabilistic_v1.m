%Bayesian apparoach
% P(a|r,s) = P(r|a,s)P(a)P(s)
% Model P(r|a,s) as a Gaussian distribution with a small variance
% and a mean = 
% Prior comes from a combination of two Beta distributions
clc;close all;clear
ntrials = 1000;
Ns = 10;
Na = 2;
beta =0.1; gamma  = 0.1;
%define prior of escape probability for observer
Pa = [0.5;0.5];

Q = zeros(Na,Ns);



%define demonstrators model
%Pad = [0.1*ones(1,Ns/2),0.5*ones(1,Ns/2)];


%initialize demonstrators policy
%For two action case : index 1 is stay, 2 is leave
pid_as(1,:) = [0.8*ones(1,Ns/2),0.5*ones(1,Ns/2)];
pid_as(2,:) = 1 - pid_as(1,:);

%calculate count values (for escapes) for this assuming 1000 trials observed 
m = 1000*pid_as(2,:);
l = 1000-m;
a = 400; b = 100; 

pid_2 = nan(Na,Ns);
pid_2(2,:) = (m+a)./(m+a+l+b);
pid_2(1,:) = 1 - pid_2(2,:);


%initialize data arrays
x = nan(1,ntrials); %stimuli
S = zeros(1,Ns); %count of trials per stimulus (gets refreshed every 100 trials)
A = zeros(1,Ns); %count of escapes per stimulus
R = nan(1,ntrials); P = nan(2,ntrials);
l=1;
%Pesc
PESC = nan(ntrials/100,Ns);
dpnp = nan(ntrials/100,2);

for ii = 1:ntrials
    
    %choose stimulus
    h = randi(Ns);
    x(ii) = h;        
    S(h) = S(h)+1;
    
    %generate posterior probability for actions for this stimulus
    Rterm = exp(beta*Q(:,h))/sum(exp(beta*Q(:,h)));
    
    P(:,ii) = Rterm.*pid_as(:,h).*Pa./sum(Rterm.*pid_as(:,h).*Pa);
    %take deterministic action
    %[~,a(ii)] = max(P(:,ii));
    %take stochastic action
    a(ii) = 1 + double(P(2,ii)>rand);
    
    if a(ii)==2
        A(h) = A(h)+1;
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
    
    R(ii) = r;
    %Q iteration
    Q(a(ii),h) = Q(a(ii),h) + gamma*(r - Q(a(ii),h));
    
    if rem(ii,100)==0
        PESC(l,:) = A./S;  %probability of escape
        dpnp(l,1) = mean(PESC(l,Ns/2 + 1:end)) - mean(PESC(l,1:Ns/2));
        A = zeros(1,Ns); S = A;
        l=l+1;
    end
    fprintf('\n %f percent done .... \n',ii*100/ntrials)
end

%figure;plot(cumsum(R));'cumulative rewards';
figure;imagesc(Q);colormap(hot); title 'final Q';
ylabel 'action 1 = stay, 2 = escape'; xlabel 'stimulus id';
figure;imagesc(PESC);ylabel 'Pesc(s)';
figure;plot(dpnp);ylabel '\Delta Pesc';

figure;plot(cumsum(R));    
    
    



