%display script for RBF perceptron experiments

% plot histograms of trials to criterion for training vs generalization
% for one choice of sigma and all 
close all
nPlots = length(K)*length(sigmaa);

countmat = nan(length(sigmaa),length(K));
count = 1:nPlots;
countmat = reshape(count,size(countmat,1),size(countmat,2));
countmat = countmat';
Stats_T_train = cell(2,1);
Stats_T_train{1} = nan(length(K),length(sigmaa)); %mean 
Stats_T_train{2} = nan(length(K),length(sigmaa)); %std
Stats_T_gen = Stats_T_train;
Stats_ED_train = Stats_T_train;
Stats_ED_gen = Stats_T_train;

do_save = 1;
path2sav = 'C:\Users\songbird\Dropbox\Observational learning paper\Simulation results\RBF_perceptron_DEM\';
        
%load demonstrators data

% figure(201); clf;
% for jj = 1:length(K)
%     for kk = 1:length(sigmaa) 
% %         subplot(length(K),length(sigmaa),countmat(jj,kk));
% %         [n,x] = hist(Result{jj}{kk}{1},20); 
% %         bar(x,n);
% %         [n,x] = hist(Result{jj}{kk}{2},20);
% %         hold on; bar(x,n,'r');
% %         
% %         title(['T (train,gen) , K : ' num2str(K(jj)) ' S: ' num2str(sigmaa(kk))]);
% %         
%     end
%     
% end

    td = cell2mat(OUTdems.trialstocrit_overbirds);
    gd = cell2mat(OUTGendems.trialstocrit_overbirds);
    
    ted = nan(length(OUTdems.BIRDS),1);
    for jj = 1:length(ted)
        ted(jj) = mean(OUTdems.delpnp_overbirds{jj}(OUTdems.trialstocrit_overbirds{jj}/200 -2: ...
            OUTdems.trialstocrit_overbirds{jj}/200));
    end
    
    ged = nan(length(OUTGendems.BIRDS),1);
    
    for jj = 1:length(ged)
        ged(jj) = mean(OUTGendems.delpnp_overbirds{jj}(OUTGendems.trialstocrit_overbirds{jj}/200 -2: ...
            OUTGendems.trialstocrit_overbirds{jj}/200));
    end

    

for jj = 1:length(K)
    for kk = 1:length(sigmaa)
        Stats_T_train{1}(jj,kk) = mean(Result{jj}{kk}{1});
        Stats_T_train{2}(jj,kk) = std(Result{jj}{kk}{1});
        Stats_T_gen{1}(jj,kk) = mean(Result{jj}{kk}{2});
        Stats_T_gen{2}(jj,kk) = std(Result{jj}{kk}{2});
        
        t1 = Result{jj}{kk}{1}; %trials to crit training 
        t2 = Result{jj}{kk}{2}; %trials to crit generalization
        e1 = Result{jj}{kk}{3}; %ED train
        e2 = Result{jj}{kk}{4}; %ED gen
        
        Stats_ED_train{1}(jj,kk) = mean(Result{jj}{kk}{3});
        Stats_ED_train{2}(jj,kk) = std(Result{jj}{kk}{3});
        Stats_ED_gen{1}(jj,kk) = mean(Result{jj}{kk}{4});
        Stats_ED_gen{2}(jj,kk) = std(Result{jj}{kk}{4});
        
        if 0
        figure;set(gcf,'Position',[300 100 700 900]);
        subplot(311);bar([mean(t1);mean(td);mean(t2);mean(gd)],0.3,'w');ylim([0 15000]);
        hold on;scatter(ones(size(t1)),t1,50,'ok','fill');
        hold on;scatter(2*ones(size(td)),td,50,'dk');
        hold on;scatter(3*ones(size(t2)),t2,50,'ok','fill');
        hold on;scatter(4*ones(size(gd)),gd,50,'dk');
        ylabel 'Trails to crit' ; set(gca,'XTickLabel',{'TRN';'TDATA';'GN';'GDATA'});
        title(['Mean train: ' num2str(mean(t1)) ' Mean gen: ' num2str(mean(t2)) ])
        
        subplot(312);bar([mean(e1);mean(ted);mean(e2);mean(ged)],0.3,'w');ylim([0 1]);
        hold on;scatter(ones(size(e1)),e1,50,'ok','fill');
        hold on;scatter(2*ones(size(ted)),ted,50,'dk');
        hold on;scatter(3*ones(size(e2)),e2,50,'ok','fill');
        hold on;scatter(4*ones(size(ged)),ged,50,'dk');
        ylabel 'ED' ; set(gca,'XTickLabel',{'TRN';'TDATA';'GN';'GDATA'});
        title(['Mean train: ' num2str(mean(e1)) ' Mean gen: ' num2str(mean(e2))]);
        
        %scatter plot of ED vs trials to crit
        [r,p] = corrcoef(t1,e1);
        mdl = fitlm(t1,e1);
        xx = min(t1):10:max(t1);
        yy = feval(mdl,xx);
        subplot(313);
        scatter(t1,e1,50,'ok','fill'); 
        xlabel 'Trials to criterion Train'; ylabel 'Esc Diff at end pt Train';
        subplot(313);hold on;plot(xx,yy,'k','LineWidth',2.2);
        title(['Sigma: ' num2str(sigmaa(kk)) '   Nmb Kernels: ' num2str(K(jj)) ...
            '  r: ' num2str(r(1,2)) ' p: ' num2str(p(1,2))]);
        
        if do_save
            saveas(gcf,[path2sav 'RBF_percep_K-' num2str(K(jj)) '-sigmaa-' num2str(sigmaa(kk))],'jpg');
        end
        
        end
        
        
                
       
    end
    
end


%% surface of trials to crit and error 
figure;surf(Stats_T_train{1}); zlabel 'Trials to Criterion'; 
set(gca,'XTickLabel',{sigmaa}); xlabel 'Sigma';
set(gca,'YTickLabel',{K}); ylabel 'Numb Kernels (K)';
title 'Mean Trials to Criterion for Training set';

figure;surf(Stats_T_gen{1}); zlabel 'Trials to Criterion'; 
set(gca,'XTickLabel',{sigmaa}); xlabel 'Sigma';
set(gca,'YTickLabel',{K}); ylabel 'Numb Kernels (K)';
title 'Mean Trials to Criterion for Generalization set';

figure;surf(Stats_ED_train{1}); zlabel 'Escape Difference'; 
set(gca,'XTickLabel',{sigmaa}); xlabel 'Sigma';
set(gca,'YTickLabel',{K}); ylabel 'Numb Kernels (K)';
title 'Mean Escape Difference for Training set';

figure;surf(Stats_ED_gen{1}); zlabel 'Escape Difference'; 
set(gca,'XTickLabel',{sigmaa}); xlabel 'Sigma';
set(gca,'YTickLabel',{K}); ylabel 'Numb Kernels (K)';
title 'Mean Escape Difference for Generalization set';
%% min point to mean of trials to criterion
avg_td = repmat(mean(td),length(K),length(sigmaa));
avg_gd = repmat(mean(gd),length(K),length(sigmaa));

%distance L1 norm
temp1 = abs(avg_td -  Stats_T_train{1});
[~,I1] = min(temp1(:));
[Irow,Icol] = ind2sub(size(temp1),I1);
bestKtrain = K(Irow)
bestSigmatrain = sigmaa(Icol)

temp2 = abs(avg_gd - Stats_T_gen{1});
[~,I2] = min(temp2(:))
[Irow,Icol] = ind2sub(size(temp2),I2)
bestKgen = K(Irow)
bestSigmagen = sigmaa(Icol)

%% min point to Escape Difference
avg_ted = repmat(mean(ted),length(K),length(sigmaa));
avg_ged = repmat(mean(ged),length(K),length(sigmaa));

%distance L1 norm
temp1 = abs(avg_ted -  Stats_ED_train{1});
[~,I1] = min(temp1(:));
[Irow,Icol] = ind2sub(size(temp1),I1);
bestKtrain_ED = K(Irow)
bestSigmatrain_ED = sigmaa(Icol)

temp2 = abs(avg_ged - Stats_ED_gen{1});
[~,I2] = min(temp2(:))
[Irow,Icol] = ind2sub(size(temp2),I2)
bestKgen_ED = K(Irow)
bestSigmagen = sigmaa(Icol)
