%Kernel WTA for observers


clc
close all


WTA = 1;
load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
X = [X1;X2]; X = zscore(X);
%x1 = X(1:size(X,1)/2,:); x2 = X(size(X,1)/2 + 1:end,:); %from my data
M = size(X,1); D = size(X,2);

K = 100; sigmaa =20; ntrials = 4000;
[PHI,mu_ks] = get_kernel_rep(X,K,sigmaa); %here is stimulus
PHI = PHI./repmat(sum(PHI),K,1);
%WTA?
if WTA
[~,I] = max(PHI);
for ii = 1:M
    pp = zeros(K,1);
    pp(I(ii)) = 1;
    PHI(:,ii) = pp;
end
end
figure(98);imagesc(PHI);ylabel 'Kernels';xlabel 'Stimuli';

%dem policy 
%initialize demonstrators policy 
%For two action case : index 1 is stay, 2 is leave
Ns = K; Na = 2;
pid_as = nan(Na,Ns);
pid_as(1,:) = [0.9*ones(1,Ns/2),0.5*ones(1,Ns/2)];
pid_as(2,:) = 1 - pid_as(1,:);

P_as = nan(Na,K);
for ii = 1:ntrials
   
    h = randi(M);
    x = X(h,:);
    phi = get_kernel_dist(x,K,sigmaa,mu_ks);
    phi = phi/sum(phi);
    [~,I] = max(phi);
    
    %fill in table
    P_as(:,I) = pid_as(:,I);
    
end
figure(2);plot(pid_as(2,:),'r');hold on;plot(P_as(2,:),'b');
%% generalization
ntrials = 4000; binsize = 400;
load('C:\Users\songbird\Dropbox\PCA - LDA analysis\Training_X1X2_dursorted_SAP_ptracecorrect');
Xgen = [X1;X2]; Xgen = zscore(Xgen);
%x1 = X(1:size(X,1)/2,:); x2 = X(size(X,1)/2 + 1:end,:); %from my data
Mgen = size(Xgen,1); D = size(Xgen,2);

S = zeros(2,Mgen/2);
A = zeros(2,Mgen/2); %count of escapes per stimulus
PESCgen = nan(ntrials/binsize,Mgen);
dpnp_gen = nan(ntrials/binsize,1); %Delta Pesc
a = nan(1,ntrials); %actions taken

l=1;
for ii = 1:ntrials
    
   h = randi(Mgen); 
   if h > Mgen/2
       c = 2;
       sind = h-Mgen/2;
   else 
       c = 1;
       sind = h;
   end
   S(c,sind) = S(c,sind) + 1;
   
   x = Xgen(h,:);
   phi = get_kernel_dist(x,K,sigmaa,mu_ks);
   phi = phi/sum(phi);
   [~,I] = max(phi);
   pp = zeros(K,1);
    pp(I(ii)) = 1;
   PHIgen(:,ii) = pp;
   a(ii) = 1 + double(P_as(2,I) > rand);
   
   if a(ii)==2 
        A(c,sind) = A(c,sind)+1;
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
    
      
    if rem(ii,binsize)==0
        PESCgen(l,:) = [A(1,:)./S(1,:), A(2,:)./S(2,:)];  %probability of escape
        dpnp_gen(l,1) = nanmean(PESCgen(l,Mgen/2 + 1:end)) - nanmean(PESCgen(l,1:Mgen/2));
        A = zeros(2,Mgen/2); S = A;
        l=l+1;
    end
   
end
figure(133);imagesc(PESCgen');title 'Generalization Pesc(s)'; ylabel 'Stimulus ID'
figure(144);plot(dpnp_gen,'-dk','LineWidth',2.4);title 'Generalization \Delta Pesc';