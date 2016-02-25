%RBF_perceptron_ model for observers
%author: Gagan Narula 25/02/2016
%This function runs Nexp experiments, where each experiment is a bird
%learning how to do the task.

%inputs:  Xtrain, Xgen are stimulus matrices
%Pdem_as : the probability of escape of the demonstrator , a 1x2 vector

function [Tcrit_train,varargout] = gn_do_RBF_percept_experiments_obs(Xtrain,Xgen,Pdem_as,opt,Nexp)

%split data into two halves
x1 = Xtrain(1:size(Xtrain,1)/2,:); 
x2 = Xtrain(size(Xtrain,1)/2 + 1:end,:); 
x1v = Xgen(1:size(Xtrain,1)/2,:); 
x2v = Xgen(size(Xtrain,1)/2 + 1:end,:); 

M = size(x1,1); %nmb samples in training set
Mgen = size(x1v,1); %nmb samples in generalization set
D = size(x1,2); %data dimensionality

%output arrays from function
Tcrit_train = nan(1,Nexp); %trails to crit on training set
ED_train = nan(1,Nexp); %Escape Difference at end of training (avg 600 trials)
% for generalization set
Tcrit_gen= nan(1,Nexp);
ED_gen = nan(1,Nexp);
%Delta Pesc dynamics
Dpnp_train = nan(opt.train_ntrials/opt.binsize,Nexp);
Dpnp_gen = nan(opt.gen_ntrials/opt.binsize,Nexp);


%% do experiments (here the observer learns to predict demonstrators behavior)
for u = 1:Nexp
    
%intialize the kernel centers
mu = randn(opt.nmb_kernels,D);

%initialize the parameter vector 
theta = randn(opt.nmb_kernels+1,1);

%initialize useful data arrays
%Q function
Q = zeros(opt.Nactions,opt.Nclasses); % 2 class function

%rewards per trial
R = nan(1,opt.train_ntrials);
S = zeros(2,M);
A = zeros(2,M); %count of escapes per stimulus

PESC = nan(opt.train_ntrials/opt.binsize,2*M); %Pescape matrix per binsize
dpnp = nan(opt.train_ntrials/opt.binsize,1); %Delta Pesc
a = nan(1,opt.train_ntrials); %actions taken
PHI = nan(opt.nmb_kernels,opt.train_ntrials); %record of RBF activations
% P values of significance test between puff and no puff 
Pval = nan(opt.train_ntrials/opt.binsize,1); 
%stimulus presented
stim = nan(2,opt.train_ntrials);

l = 1; error = zeros(1,opt.train_ntrials); 

for ii = 1:opt.train_ntrials
   
        h = randi(M);
    %choose stimulus
     if rand > opt.Ppuff
        %this is No puff (1) class
        c = 1; label = -1;
        %choose a stimulus s in Rd
        stim(c,ii) = h;

        s = x1(h,:); %stimulus from class no puff
        
        if ~opt.usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,opt.sigmaa);
            %normalize activation?
            if opt.rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if opt.WTA 
                [~,I] = max(phi);
                phi = zeros(K,1);
                phi(I) = 1;                
            end
           
        end
        PHI(:,ii) = phi;
    else
        c = 2; label = 1;
        stim(c,ii) = h;
                
        s = x2(h,:);
        
        if ~opt.usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,opt.sigmaa);
            %normalize activation?
            if opt.rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if opt.WTA 
                [~,I] = max(phi);
                phi = zeros(K,1);
                phi(I) = 1;                
            end
            
        end
        PHI(:,ii) = phi;
     end
    S(c,h) = S(c,h) + 1;
    
    %generate a label from the demonstrators behavior by sampling
    demlabel = Pdem_as(2,c)
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
    
    Rterm =  exp(opt.beta*Q(:,Qind));
    p = Rterm.*opt.Pescape;
    P = p/sum(p);
    %choose an action
    %deterministic policy
    %[~,a(ii)] = max(P(:,ii));
    %stochastic action
    a(ii) = 1 + double(P(2) > rand);
    
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
        theta = theta + opt.eta*[phi;1]*label;
    end
    
    %update Q
    Q(a(ii),Qind) = Q(a(ii),Qind) + opt.alpha*(r - Q(a(ii),Qind));
    
    
    if rem(ii,opt.binsize)==0
        PESC(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp(l,1) = nanmean(PESC(l,M + 1:end)) - nanmean(PESC(l,1:M));
        PuffProb = nanmean(PESC(l,M + 1:end)); NoPuffProb = nanmean(PESC(l,1:M));
        [~,~,~,Pval(l),~] = gn_fa_meta_airpuff_sigtest_binom(PuffProb, ...
            NoPuffProb,[sum(S(1,:)),sum(S(2,:))],0.01);
        A = zeros(2,M); S = A;
        l=l+1;
    end
    
    if opt.verbose
        fprintf('\n .. Training %f percent done .. \n',ii*100/opt.train_ntrials);
        if ii > 50
            fprintf('\n .. Perceptron loss : %f .. \n',mean(error(ii-50)));
        end
    end
   
end
%smooth error
binsiz2 = 50;
Err = nan(1,floor(opt.train_ntrials/binsiz2));
l = 1;
for kk = 1:length(Err);
    Err(kk) = mean(error(l:l+binsiz2-1));
    l = l+binsiz2;
end

%% get criterion on training delpesc
PVAL_sig = Pval < 0.01; %significant p values
windoww = 8;
thresh = 0.9;
binsize = 200;
gn_get_bird_out_trialstocrit;

Tcrit_train(u) = trialstocrit;
ED_train(u) = mean(dpnp(binstocrit-2:binstocrit));
Dpnp_train(:,u) = dpnp;

%% generalization 
opt.gen_ntrials = 5000;

%rewards per trial
Rgen = nan(1,opt.gen_ntrials); 

%predicted class 
Chat = nan(1,opt.gen_ntrials);
S = zeros(2,Mgen);
A = zeros(2,Mgen); %count of escapes per stimulus
PESCgen = nan(opt.gen_ntrials/opt.binsize,2*Mgen);
dpnp_gen = nan(opt.gen_ntrials/opt.binsize,1); %Delta Pesc
a = nan(1,opt.gen_ntrials); %actions taken
error_gen = zeros(1,opt.gen_ntrials);
Pval_gen = nan(opt.gen_ntrials/opt.binsize,1);

l =1;
for ii = 1:opt.gen_ntrials
    
    h = randi(Mgen);
    %choose a class
    if rand > opt.Ppuff
        %this is No puff (1) class
        c = 1; label = -1;
        %choose a stimulus s in Rd
        %X(1,ii) = h;

        s = x1v(h,:);
        if ~opt.usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,opt.sigmaa);
            %normalize activation?
            if opt.rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if opt.WTA 
                [~,I] = max(phi);
                phi = zeros(K,1);
                phi(I) = 1;                
            end
            
        end
    else
        c = 2; label = 1;
        %X(2,ii) = h;
                
        s = x2v(h,:);
        if ~opt.usekernel
            %use stimulus as it is with a 1 attached for bias term
            phi = [s,1]';
        else
            %convert stimulus to RBF activation
            phi = get_RBF_activation(s,mu,opt.sigmaa);
            %normalize activation?
            if opt.rbf_norm
                phi = phi/sum(phi);
            end
            % winner - take - all operation?
            if opt.WTA 
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
    Rterm =  exp(opt.beta*Q(:,Qind));
    p = Rterm.*opt.Pescape;
    P = p/sum(p);
    
    %choose an action
    %deterministic policy
    %[~,a(ii)] = max(P(:,ii));
    %stochastic action
    a(ii) = 1 + double(P(2) > rand);
    
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
        theta = theta + opt.eta*[phi;1]*label;
    end
        
    
    if rem(ii,opt.binsize)==0
        PESCgen(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp_gen(l,1) = nanmean(PESCgen(l,Mgen + 1:end)) - nanmean(PESCgen(l,1:Mgen));
        PuffProb = nanmean(PESCgen(l,M + 1:end)); NoPuffProb = nanmean(PESCgen(l,1:M));
        [~,~,~,Pval_gen(l),~] = gn_fa_meta_airpuff_sigtest_binom(PuffProb, ...
            NoPuffProb,[sum(S(1,:)),sum(S(2,:))],0.01);
        A = zeros(2,Mgen); S = A;
        l=l+1;
    end
    
    if opt.verbose
        fprintf('\n .. Generalization %f percent done .. \n',ii*100/opt.gen_ntrials);
        if ii > 50
            fprintf('\n .. Perceptron loss : %f .. \n',mean(error(ii-50)));
        end
    end
end
%smooth error
binsiz2 = 50;
Errgen = nan(1,floor(opt.gen_ntrials/binsiz2));
l = 1;
for kk = 1:length(Errgen);
    Errgen(kk) = mean(error_gen(l:l+binsiz2-1));
    l = l+binsiz2;
end

%% get criterion on generalization  delpesc
PVAL_sig = Pval_gen < 0.01; %significant p values
windoww = 8;
thresh = 0.9;
binsize = 200;
gn_get_bird_out_trialstocrit;

Tcrit_gen(u) = trialstocrit;
ED_gen(u) = mean(dpnp_gen(binstocrit-2:binstocrit));

Dpnp_gen(:,u) = dpnp_gen;


fprintf('\n ... %d of %d experiments done ... \n',u,Nexp);
end

varargout{1} = Tcrit_gen;
varargout{2} = ED_train;
varargout{3} = ED_gen;
varargout{4} = Dpnp_train;
varargout{5} = Dpnp_gen;
varargout{6} = PHI;
varargout{7} = stim;
varargout{8} = {Err;Errgen};

