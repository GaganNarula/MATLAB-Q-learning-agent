%FISHER LINEAR DISCRIMINANT ANALYSIS ON dems and obs 

clc
clear 
close all

%load('/Users/GaganN/Dropbox/PCA - LDA analysis/Training_X1X2_dursorted_SAP_ptracecorrect.mat')
load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
 
Stimuli = [X1;X2];
Stimuli = zscore(Stimuli);

d = size(Stimuli,2);
N = size(Stimuli,1);

%Calculate within class means
Class_means = zeros(2,d);
Class_means(1,:) = mean(Stimuli(1:N/2,:));
Class_means(2,:) = mean(Stimuli(N/2 + 1:end,:));

%Calculate within class variance
%first subtract mean from stimulus in each class separately
Class1 = Stimuli(1:N/2,:) - repmat(Class_means(1,:),N/2,1);
Class2 = Stimuli(N/2 +1:end,:) - repmat(Class_means(2,:),N/2,1);
%within class total covariance
S_w = Class1'*Class1 + Class2'*Class2;

%calculate inverse of S_w
S_w_inv = inv(S_w);

%Fisher LD projection direction is 
lda_w = S_w_inv*(Class_means(1,:) - Class_means(2,:))';

%Project stimulus onto this discriminant
Proj_stimulus = lda_w'*Stimuli';

%% classify
y1 = Proj_stimulus(1:N/2);
y2 = Proj_stimulus(N/2 + 1:N);

figure(1);plot(y1,'db'); hold on; plot(y2,'dr')
[n1,c1] = hist(y1,20);
[n2,c2] = hist(y2,20);

figure(2);bar(c1,n1,0.5,'b');hold on;bar(c2,n2,0.5,'r');

y1pred = y1 >= 0;
y2pred = y2 < 0;

errorst = sum(y1pred==0) + sum(y2pred==0)

%% generalization
%load('/Users/GaganN/Dropbox/PCA - LDA analysis/Generalize_X1X2_dursorted_SAP_ptracecorrect.mat')
load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Generalize_X1X2_dursorted_SAP_ptracecorrect');
   
Stimgen = [X1;X2];
Stimgen = zscore(Stimgen);

ygen1 = lda_w'*Stimgen(1:N/2,:)';
ygen2 = lda_w'*Stimgen(N/2 + 1:N,:)';

figure(3);plot(ygen1,'db'); hold on; plot(ygen2,'dr')
[n1,c1] = hist(ygen1,20);
[n2,c2] = hist(ygen2,20);

figure(4);bar(c1,n1,0.5,'b');hold on;bar(c2,n2,0.5,'r');

y1gpred = ygen1 >= 0;
y2gpred = ygen2 < 0;

errorsg = sum(y1gpred==0) + sum(y2gpred==0)



%% for observers training set

%generate labels
Posslabl_prob = [0.5*ones(N/2,1);0.1*ones(N/2,1)];

labls = Posslabl_prob > rand(N,1);

Class1obs = Stimuli(labls==1,:);
Class2obs = Stimuli(labls==0,:);

%within class total covariance
S_wobs = Class1obs'*Class1obs + Class2obs'*Class2obs;

%calculate inverse of S_w
S_w_invobs = inv(S_wobs);

%Fisher LD projection direction is 
lda_wobs = S_w_invobs*(mean(Class1obs) - mean(Class2obs))';

%Project stimulus onto this discriminant
P_obs = lda_wobs'*[Class1obs;Class2obs]';

%classify
y1obs = P_obs(1:N/2); y2obs = P_obs(N/2 + 1:N);

figure(5);plot(y1obs,'db'); hold on; plot(y2obs,'dr')
[n1,c1] = hist(y1obs,20);
[n2,c2] = hist(y2obs,20);

figure(6);bar(c1,n1,0.5,'b');hold on;bar(c2,n2,0.5,'r');

y1pred_obs = y1obs >= 0;
y2pred_obs = y2obs < 0;

errorst_obs = sum(y1pred_obs==0) + sum(y2pred_obs==0)