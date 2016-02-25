%MIXTURE DENSITY METHOD
%CORRECT VERSION
close all;
%softmax policy, 
ntrials = 1000;
Ns = 10;
Na = 2;

beta = 1; %inverse temperature 

%Confidence
alpha = 0.5;  
incalpha = 0:0.9/ntrials:1;


gamma = 0.1; %learning rate for Q

%initialize demonstrators policy
pid_as = nan(Na,Ns);
%For two action case
pid_as(1,1:Ns/2) = 0.9;  %action one is stay, action 2 is escape , stimuli 1 to Ns/2 are nopuff
pid_as(1,Ns/2 + 1:end) = 0.6;
pid_as(2,1:Ns/2) = 0.1;
pid_as(2,Ns/2 + 1:end) = 0.4;


pio_as = nan(Na,Ns);
Q = zeros(Na,Ns);

%actions taken per stimulus
a = nan(1,ntrials);


%actions as a function of stimulus
A = zeros(1,Ns);
S = zeros(1,Ns);

%stimuli chosen
x = a;


%Pesc
PESC = nan(ntrials/100,Ns);
dpnp = nan(ntrials/100,2);

l=1;
for ii = 1:ntrials
    
    %choose stimulus
    h = randi(Ns);
    x(ii) = h;        
    S(h) = S(h)+1;
    
    
    pio_as(:,h) = exp(beta*Q(:,h))./sum(exp(beta*Q(:,h)));
    
    %mix and create output posterior
    %FIXED ALPHA
    %P = alpha*pio_as(:,h).*(1-pio_as(:,h)) + (1-alpha)*pid_as(:,h).*(1-pid_as(:,h));
    %CHANGING ALPHA
    P = incalpha(ii)*pio_as(:,h)+ ...
        (1-incalpha(ii))*pid_as(:,h);

    %take deterministic action
    %[~,a(ii)] = max(P);
    %take stochastic action
    a(ii) = 1 + double(P(2)>rand);
    
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
    error(ii) = r - Q(a(ii),h);
    Q(a(ii),h) = Q(a(ii),h) + gamma*(error(ii));
    
    if rem(ii,100)==0
        PESC(l,:) = A./S;  %probability of escape
        dpnp(l,1) = mean(PESC(l,Ns/2 + 1:end)) - mean(PESC(l,1:Ns/2));
        A = zeros(1,Ns); S = A;
        l=l+1;
    end
    fprintf('\n %f percent done .... \n',ii*100/ntrials)
end

figure(11);plot(cumsum(R));'cumulative rewards';
figure(22);imagesc(Q);colormap(hot); title 'final Q';
ylabel 'action 1 = stay, 2 = escape'; xlabel 'stimulus id';
figure(33);imagesc(PESC); title 'Pesc(s) vs 100 trial bins';xlabel 'stimulus id';
figure(44);plot(dpnp); ylabel '\Delta Pesc';

figure(55);plot(error.^2);