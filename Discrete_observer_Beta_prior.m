%Discrete case
%Observer evaluates policy with beta prior
%
clc
clear
close all

%number of stimuli
Ns = 10;
%number of actions
Na = 2;

ntrials = 1000;
binsize = 100;

%parameters
alpha = 0.01; %sarsa learning rate
beta = 0.5; %inverse temperature
Pescape = [0.65; 0.35]; %prior probability of escape

%policy from demonstrator  P(a|s)
pid_as = nan(Na,Ns);
%For two action case : index 1 is stay, 2 is leave
pid_as(1,:) = [0.9*ones(1,Ns/2),0.5*ones(1,Ns/2)];
pid_as(2,:) = 1 - pid_as(1,:);


%%FOR BETA PRIOR -- SWITCH OFF UPDATE
%initialize your policy to a beta reweighted dem policy
pio = nan(Na,Ns);
pio(2,:) = [betarnd(1,4,1,Ns/2) betarnd(4,1,1,Ns/2)];
pio(1,:) = 1 - pio(2,:);

initialdpnp = mean(pid_as(2,Ns/2 + 1:end)) - mean(pid_as(2,1:Ns/2));


%initialize data arrays
x = nan(1,ntrials); %stimuli
a = nan(1,ntrials); %action
S = zeros(1,Ns); %count of trials per stimulus (gets refreshed every 100 trials)
A = zeros(1,Ns); %count of escapes per stimulus
R = nan(1,ntrials); P = nan(2,ntrials);
K = S;

Q = zeros(Na,Ns);


%Pesc
PESC = nan(ntrials/100,Ns);
dpnp = nan(ntrials/100,2);
error = nan(1,ntrials);

updates=0; %number of times the policy is updated 
l = 1;

for ii = 1:ntrials
    
    %choose stimulus
    h = randi(Ns);
    x(ii) = h;     
    
    %dummy variables
    S(h) = S(h)+1;
    K(h) = K(h)+1; 
    
    %choose stochastic action according to pid
    %a(ii) = 1 + double(pio(2,h)> rand);
    
    %%%ACTING ACCORDING TO BETA PRIOR
    if h > Ns/2
        a(ii) = 1 + double(betarnd(4,1) > rand);
    else
        a(ii) = 1 + double(betarnd(1,4) > rand);
    end
    
    %for greedy action:
    %[~,a(ii)] = max(pio(:,h));
        
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
    Q(a(ii),h) = Q(a(ii),h) + alpha*(r - Q(a(ii),h));
    
    %squared bellman error
    error(ii) = (r-Q(a(ii),h)).^2;
    
    if rem(ii,100)==0
        PESC(l,:) = A./S;  %probability of escape
        dpnp(l,1) = mean(PESC(l,Ns/2 + 1:end)) - mean(PESC(l,1:Ns/2));
        A = zeros(1,Ns); S = A;
        l=l+1;
    end
    
            
    fprintf('\n %f percent done .... \n',ii*100/ntrials)
end

figure(11);hold on;plot(cumsum(R),'r');title 'cumulative Rewards earned'; xlabel 'trials';
figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Stimulus ID';
xlabel 'Actions, 2 = escape, 1 = stay';

figure(33);imagesc(PESC'); ylabel 'Stimulus ID'; xlabel 'binsize trial bins';
title(['Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

figure(44);hold on;plot(dpnp,'-dk','LineWidth',2.4);xlabel 'binsize trial bins';
tt = nan(length(dpnp),1); tt(1)= initialdpnp;
hold on;plot(tt,'*r');
title(['\Delta Pesc  alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);
legend('OBS','FROM DEM');
tt = mean(PESC(end-5:end,:)); tt = tt(:);
figure(55);plot([tt(1:Ns/2) ;  nan(Ns/2,1)],'-db','LineWidth',2.4); 
hold on; plot([nan(Ns/2,1); tt(Ns/2 + 1:end)],'-dr','LineWidth',2.4); ylim([0 1]);
title(['Pesc(stimulus) at end, alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

xlabel 'Stimulus ID';  ylabel 'Pesc';


if 0
    h1 = figure(33);
    saveas(h1,'C:\Users\songbird\Dropbox\Q learning agent\OBS_PESC','bmp');
    h2 = figure(44);
    saveas(h2,'C:\Users\songbird\Dropbox\Q learning agent\OBS_dpnp','bmp');
    h3 = figure(55);
    saveas(h3,'C:\Users\songbird\Dropbox\Q learning agent\OBS_Pesc(stimulus)','bmp');
end