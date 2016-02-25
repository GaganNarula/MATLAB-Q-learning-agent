%Using policy directly, then change policy based on earned rewards
%(Beta distribution type
clc
clear
close all


Ns = 10;
Na = 2;
ntrials = 1000;

%parameters
alpha = 0.005;
beta = 30;

%policy from demonstrator 
pid_as = nan(Na,Ns);
%For two action case : index 1 is stay, 2 is leave
pid_as(1,:) = [0.9*ones(1,Ns/2),0.5*ones(1,Ns/2)];
pid_as(2,:) = 1 - pid_as(1,:);
pio = pid_as; initialdpnp = mean(pio(2,Ns/2 + 1:end)) - mean(pio(2,1:Ns/2));
Pact = pio(2,:);
Pescape = [0.7; 0.3];
%initialize data arrays
x = nan(1,ntrials); %stimuli
a = nan(1,ntrials); %action
S = zeros(1,Ns); %count of trials per stimulus (gets refreshed every 100 trials)
A = zeros(1,Ns); %count of escapes per stimulus
R = nan(1,ntrials); P = nan(2,ntrials);
K = S;

Q = zeros(Na,Ns);

l=1;
%Pesc
PESC = nan(ntrials/100,Ns);
dpnp = nan(ntrials/100,2);
error = nan(1,ntrials);

%accumulated rewards per stimulus and action
S_count = zeros(1,Ns);
R_count = zeros(Na,Ns);

updates=0;
for ii = 1:ntrials
    
    %choose stimulus
    h = randi(Ns);
    x(ii) = h;        
    S(h) = S(h)+1;
    K(h) = K(h)+1;
    
    %choose stochastic action according to pid
    a(ii) = 1 + double(Pact(h)> (rand));
    [~,I] = max(pio(:,h));
    
%      %choose an action
%     if rand > sigma  %greedy action
%         a(ii) = I;
%     else
%         if I==2      %random action
%         a(ii) = 1;
%         elseif I==1
%             a(ii) = 2;
%         end
%     end
    
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
    error(ii) = (r-Q(a(ii),h)).^2;
    if rem(ii,100)==0
        PESC(l,:) = A./S;  %probability of escape
        dpnp(l,1) = mean(PESC(l,Ns/2 + 1:end)) - mean(PESC(l,1:Ns/2));
        A = zeros(1,Ns); S = A;
        l=l+1;
    end
    
    %update pid if each state is visited atleast once
%     if sum(K>=ones(size(K)))==length(K)
%         for jj=1:Ns
%             p = exp(beta*Q(:,jj)).*Pescape;
%             pio(:,jj) = p/sum(p);
%         end
%         for jj=1:Ns
%             p = s
%             
%         end
%         %K(:)=0;
%         updates = updates+1;
%     end
            
    fprintf('\n %f percent done .... \n',ii*100/ntrials)
end

figure(11);hold on;plot(cumsum(R),'r');title 'cumulative Rewards earned'; xlabel 'trials';
figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Stimulus ID';
xlabel 'Actions, 2 = escape, 1 = stay';

figure(33);imagesc(PESC'); ylabel 'Stimulus ID'; xlabel 'binsize trial bins';
title(['Pesc(stimulus) alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

figure(44);hold on;plot(dpnp,'-dk','LineWidth',2.4);xlabel 'binsize trial bins';
tt = nan(length(dpnp),1); tt(1)= initialdpnp;
hold on;plot(tt,'*k');
title(['\Delta Pesc  alpha: ' num2str(alpha) ' beta: ' num2str(beta) ' P(a=1): ' num2str(Pescape(2))]);

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