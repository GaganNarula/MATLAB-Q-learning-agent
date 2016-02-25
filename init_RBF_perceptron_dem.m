%RBF based perceptron for demonstrators
%This script performs several experiments at a given kernel density and
%returns the trials to criterion and end point delta pesc

clc
close all
clear

rng(123,'twister');
%% create input
artific = 0;
if artific
M = 40; D = 2;
m1 = 4; m2 = 6; shiftt = 5; s1 = 3; s2 = 4;
[x1,x2,V,dd,L] = create_class_input(M,D,m1,m2,shiftt,s1,s2);
else
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
    %load('/Users/GaganN/Dropbox/PCA - LDA analysis/Training_X1X2_dursorted_SAP_ptracecorrect.mat');
    
    Xtrain = [X1;X2]; Xtrain = zscore(Xtrain); 
    load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
    %load('/Users/GaganN/Dropbox/PCA - LDA analysis/Generalize_X1X2_dursorted_SAP_ptracecorrect.mat');
   
    Xgen = [X1;X2]; Xgen = zscore(Xgen);
end

%% options structure

%use RBF kernel?
opt.usekernel = 1;
%normalize kernel activations ?
opt.rbf_norm = 1;   
%make a winner-take-all type binarization ?
opt.WTA = 0;

%number of RBF kernels
K = [30 50 75 100 200];

%kernel scale parameter
sigmaa = [0.5 1 2 4 8 16];

%optimize scale parameter?
opt.opt_sigmaa = 0;
%TD learning learning rate alpha
opt.alpha = 0.1;
%Policy inverse temperature beta
opt.beta = 1;
%Perceptron learning rate
opt.eta = 0.1;
%total number of trials
opt.train_ntrials = 10000;
opt.gen_ntrials = 5000;
opt.binsize = 200;

opt.Pescape = [0.5;0.5]; % prior prob of escape, Pescape(1)= Prob of staying
%probability of choosing a puff stimulus
opt.Ppuff = 0.25;

opt.Nactions = 2;
opt.Nclasses = 2;

opt.verbose = 0;
%% call script that runs training experiment Nexp times for different kernels 

Nexp = 9;
Result = cell(length(K),1);
verbose  = 1;

%loop over number of kernels 
kk = 1;
for ii = 1:length(K)
    opt.nmb_kernels = K(ii);
    fprintf('\n\n .. OUTER LOOP , K : %d , %d left .. \n\n',K(ii),length(K)-ii);
    Result{ii,1} = cell(length(sigmaa),1);
    
    %loop over sigmaa
    for jj = 1:length(sigmaa)
        
        fprintf('\n\n .. INNER LOOP , sigma : %f , %d left .. \n\n',sigmaa(ii),length(sigmaa)-ii);
    
        opt.sigmaa = sigmaa(jj);
        
        [Tcrit_train,Tcrit_gen,ED_train,ED_gen, ...
            Dpnp_train,Dpnp_gen] = gn_do_RBF_percept_experiments(Xtrain,Xgen,opt,Nexp);
        
        
        Result{ii,1}{jj} = {Tcrit_train,Tcrit_gen,ED_train,ED_gen,Dpnp_train,Dpnp_gen};
        
        if 0
        figure;bar([mean(Tcrit_train);mean(Tcrit_gen)],0.2);
        hold on;scatter(ones(size(Tcrit_train)),Tcrit_train,20,'dk','fill');
        hold on;scatter(2*ones(size(Tcrit_gen)),Tcrit_gen,20,'dk','fill');
        ylabel 'Trails to crit' ; set(gca,'XTickLabel',{'TRAIN';'GFN'});
        title(['Sigma : ' num2str(opt.sigmaa) ' Nmb Kernels : ' num2str(K(ii))]);
        
        figure;bar([mean(ED_train);mean(ED_gen)],0.2);
        hold on;scatter(ones(size(ED_train)),ED_train,20,'dk','fill');
        hold on;scatter(2*ones(size(ED_gen)),ED_gen,20,'dk','fill');
        ylabel 'ED' ; set(gca,'XTickLabel',{'TRAIN';'GFN'});
        title(['Sigma : ' num2str(opt.sigmaa) ' Nmb Kernels : ' num2str(K(ii))]);
        end
        kk = kk+1;
        fprintf('\n\n ######## %f percent left ######## \n\n',kk*100/(length(K)*length(sigmaa)));
    end
    
end


%%