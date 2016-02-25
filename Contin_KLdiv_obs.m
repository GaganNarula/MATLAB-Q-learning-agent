%Continuous case 
%Observer learns a distribution q_theta(a|s) to approximate p_theta(a|s)
%where p_theta(a|s) is the demonstrators policy for an s
%
% q_theta(a|s) = exp(-theta*phi(s))/ Z
% theta = theta - eta*KL(p||q)

clc 
clear
close all

%% create input 
M = 40; D = 2;
m1 = 4; m2 = 6; shiftt = 5; s1 = 2; s2 = 4;
[x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
x1 = zscore(x1);  x2 = zscore(x2);
%% set parameters
%Prior probability of puff class
Ppuff = 0.5;

%prior probability of escape
Pescape = [0.75;0.25];

%RATES
alpha = 0.02; %Qlearning rate 
beta = 1; %inverse temperature
eta = 0.03; %learning rate of perceptron


% N actions
Na = 2;

ntrials = 8000;
binsize = 200;

Ns = 2*M;
%policy from demonstrator 
pid_as = nan(Na,Ns);
%For two action case : index 1 is stay, 2 is leave
pid_as(1,:) = [0.9*ones(1,Ns/2),0.5*ones(1,Ns/2)];
pid_as(2,:) = 1 - pid_as(1,:);


%THETA
%theta = 0.5*randn(Na,D);
theta = 0.5*randn(1,D+1); % +1 for action

%Probability of actions 


%rewards per trial
R = nan(1,ntrials);
S = zeros(2,M);
A = zeros(2,M); %count of escapes per stimulus
PESC = nan(ntrials/binsize,2*M);
dpnp = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken

%g = [];
for ii = 1:ntrials
    
    %% stimulus selection
    h = randi(M);
    %tmp = randperm(M);
    %h = tmp(1);
    %choose a class
    if rand > Ppuff
        %this is No puff (1) class
        c = 1; label = -1;
        %choose a stimulus s in Rd
        %X(1,ii) = h;
    
        s = x1(h,:);
        sind = h;
    else
        c = 2; label = 1;
        %X(2,ii) = h;
                
        s = x2(h,:);
        sind = h+M;
    end
    S(c,h) = S(c,h) + 1;
    
    
    %% make q 
    %q_theta = exp(-beta*theta*s');
    
    q_theta(1) = exp(beta*theta*[s,1]'); %stay
    q_theta(2) = exp(beta*theta*[s,2]'); %escape
    q_theta = q_theta/sum(q_theta);
    
    %%update theta
    for jj = 1:Na
        
            %minimizing DKL
            theta = theta + eta*( pid_as(jj,sind)*(1 - q_theta(jj))*[s,jj] );
            %grad(jj,kk) = abs( pid_as(jj,sind)*(1 - q_theta(jj))*s(kk) );
            %minimizing squared error
            % theta(jj,kk) = theta(jj,kk) + eta*( (pid_as(jj,sind) - q_theta(jj)) ...
            %     *q_theta(jj)*(1 - q_theta(jj))*s(kk) );
        
    end
    %g = [g;mean(mean(grad))];
    if rem(ii,100)==0
        figure(44);plot(theta,'*k');
        %figure(99);hold on;plot(mean(mean(grad)),'-.b');
        %pause
    end
end

Q = nan(2*M,2);
Q(:,1) = exp(beta*theta*[[x1,ones(M,1)];[x2,ones(M,1)]]');
Q(:,2) = exp(beta*theta*[[x1,2*ones(M,1)];[x2,2*ones(M,1)]]');
Q = Q./repmat(sum(Q,2),1,Na);
figure(555);plot(pid_as(2,:),'b');
hold on;plot(Q(2,:),'r');
