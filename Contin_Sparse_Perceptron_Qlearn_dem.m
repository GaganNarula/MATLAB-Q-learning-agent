%Continuous case
%One-Hot mapping
%Demonstrator - classifies 
%Q Learning

%author: gagan narula 18/12/2015

clc
clear
close all

%All parameters
%input dimensionality
D = 10;

%input sample size (same size for each class)
M = 100;  %some proportion of this number is used as generalization set

%proportion training samples
ptrain = 0.75;

%number of cortical neurons
Ns = 300;

ntrials = 8000;
binsize = 100;

%probability of presenting a puff stimulus
Ppuff = 0.5;


%RATES
alpha = 0.2; %learning rate Q
beta = 0.7; %beta parameter
eta = 0.4; %learning rate perceptron

%prior probability of escape
Prior_a = [0.5; 0.5];% escape probability is second row


%% create input 

m1 = 4; m2 = 6; shiftt = 5; s1 = 2; s2 = 4;
[x1,x2,V] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
% x1 and x2 are size M x D matrices, V is set of Principal components of
% this data (in descending eigen value order from left to right column)
% x2 is PUFF CLASS!!

%% separate input into Training and Generalization set

%randomly permute
order = randperm(M);
x1 = x1(order,:); x2 = x2(order,:);

x1train = x1(1:floor(ptrain*M),:);
x1test = x1(size(x1train,1)+1:M,:);

x2train = x2(1:floor(ptrain*M),:);
x2test = x2(size(x1train,1)+1:M,:);
Mtrain = size(x1train,1);

%zscore data (optional)
x1train = zscore(x1train);
x2train = zscore(x2train);

x1test = zscore(x1test);
x2test = zscore(x2test);

%% create a sparse code from this stimulus set
mode1 = 'random';
mode2 = 'WTA';
displayy = 1;

[y1train,y2train,J] = create_sparse_rep(mode1,mode2,x1train,x2train, ...
    displayy,Ns);
[y1test,y2test] = create_sparse_rep(mode1,mode2,x1test,x2test, ...
    0,Ns,[],[],J);


%% setup learning experiment


%Q function
Na = 2; %number of actions
Nc = 2; %number of classes
Q = zeros(Na,Nc);

%Probability of actions over trials
P = nan(Na,ntrials);

%rewards per trial
R = nan(1,ntrials);

S = zeros(1,2*Mtrain); %count of number of times stimulus occurs
A = zeros(1,2*Mtrain); %count of escapes per stimulus

error = nan(1,ntrials); %classification error

PESCtrain = nan(ntrials/binsize,2*Mtrain); %Pesc
dpnp = nan(ntrials/binsize,1); %Delta Pesc

a = nan(1,ntrials); %actions taken per trial 1=stay, 2=escape

%initialize Weight vector for classifier (+1 for bias)
W = -0.5 + rand(Ns+1,1); 

%% TRAINING PERIOD: loop over trials
l = 1;%dummy count
for ii = 1:ntrials
    
    %choose a stimulus
    h = randi(Mtrain);
        
    %choose a class
    if rand>Ppuff
        %puff class        
        x = [y2train(h,:) 1]';
        indforS = h+Mtrain;
        labl = 1;         %class label
    else
        x = [y1train(h,:) 1]';
        indforS = h;
        labl = 0;         %class label
    end
    
    S(indforS) = +1;
    
    %classify the stimulus
    c = heaviside(W'*x);
    if c==0.5
      c = 1;  %class 1 or 0
    end
    
    %make a reward prediction
    Rhat = exp(beta*Q(:,c+1));
    
    %make a posterior over actions
    P(:,ii) = Rhat.*Prior_a/sum(Rhat.*Prior_a);
    
    %choose action
    %[~,a(ii)] = max(P(:,ii)); %greedy action
    a(ii) = 1 + (P(2,ii) > rand); %stochastic action 
        
    if a(ii)==2 
        A(indforS) = A(indforS)+1;
    end
    
    %get reward
    if (labl==0 && a(ii)==2)  %this is escape when no puff
        
        r = -1;
    elseif (labl==0 && a(ii)==1) % this is stay when no puff
        r = 1;
    elseif (labl==1 && a(ii)==2) %this is escape when puff
        r = 1;
    elseif (labl==1 && a(ii)==1)
        r = -1;             % this is stay when puff 
    end
    
    %save rewards
    R(ii) = r;
    
    %update classifier weights
    W = W + eta*(labl - c)*x;
    error(ii) = (labl - c)^2;
    
    %update Q
    Q(a(ii),c+1) = Q(a(ii),c+1) + alpha*(r - Q(a(ii),c+1));
    
    if rem(ii,binsize)==0
        PESCtrain(l,:) = A./S;  %probability of escape
        dpnp(l,1) = nanmean(PESCtrain(l,Mtrain + 1:end)) - nanmean(PESCtrain(l,1:Mtrain));
        A = zeros(1,2*Mtrain); S = A;
        l=l+1;
    end
    fprintf('\n .... %f Percent done .... \n',ii*100/ntrials);
    
end

%display
figure(11);hold on;plot(cumsum(R),'r'); title 'Cumulative Rewards earned';
figure(22);hold on;imagesc(Q);colormap(hot); title 'final Q'; ylabel 'Actions'; 
xlabel 'Classes';
figure(33);imagesc(PESCtrain');title 'Pesc(s)'; ylabel 'Stimulus ID'
figure(44);plot(dpnp,'-dk','LineWidth',2.4);title '\Delta Pesc';
figure(55);plot(cumsum(error),'k','LineWidth',2.4); title 'Classification error';

    
    
    




