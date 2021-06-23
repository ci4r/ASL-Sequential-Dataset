clc; clear; close all;

mainpath = 'E:\110signOutput\';
subjects = dir(mainpath);
subjects = subjects(3:12);
idx = 1:length(subjects);
ntv_idx = [3 4 5 7];
natives = subjects(ntv_idx);
imit_idx = idx;
imit_idx(ntv_idx) = [];
imits = subjects(imit_idx);

num_class = 110;

for i = 1:length(natives)
   ntvpath = [mainpath natives(i).name '\microDoppler\Cut\*.png'];
   ntvfiles{i} = dir(ntvpath);
end
ntv_f = cat(1,ntvfiles{:});
for i = 1:length(imits)
   imitpath = [mainpath imits(i).name '\microDoppler\Cut\*.png'];
   imitfiles{i} = dir(imitpath);
end
imit_f = cat(1,imitfiles{:});

for i = 80:num_class
    msg = ['Processing Class: ' num2str(i) '/' num2str(num_class)];
    disp(msg)
    str = ['_' num2str(i) '.png'];
    
    ntv_i = strfind({ntv_f.name},str);
    ntvid = find(~cellfun(@isempty,ntv_i));
    for j = 1:length(ntvid)
        ntv_im{j} = imresize(imread(fullfile(ntv_f(j).folder,ntv_f(j).name)),[128, 128]);
        [up,~,down] = env_find(ntv_im{j});
        ntv_env{j} = [up down];
    end
    
    imit_i = strfind({imit_f.name},str);
    imitid = find(~cellfun(@isempty,imit_i));
    for j = 1:length(imitid)
        imit_im{j} = imresize(imread(fullfile(imit_f(j).folder,imit_f(j).name)),[128, 128]);
        [up,~,down] = env_find(imit_im{j});
        imit_env{j} = [up down];
    end
    cnt = 1;
%     for m = 1:length(ntvid)
%         for n = 1:length(imitid)
% %             dtw_temp(cnt) = dtw(ntv_env{m}, imit_env{n});
% %             [cm(cnt), cSq] = DiscreteFrechetDist(ntv_env{m},imit_env{n});
% %             ssimval(cnt) = ssim(imit_im{n},ntv_im{m});
% %             euc(cnt) = sqrt(sum((imit_im{n}(:) - ntv_im{m}(:)) .^ 2));
%             euc(cnt) = sqrt(sum((imit_env{n} - ntv_env{m}) .^ 2));
%             
%             cnt = cnt+1;
%         end
%     end
    
%     R = corrcoef(group1,group2); % Pearson correlations. look into diagonal values. 
%     [A,B,r] = canoncorr(X,Y); % Cannonical correlations. A,B coefecient, r is the correlations.

%     dtw_dist(i) = mean(dtw_temp);
%     dft_dist(i) = mean(cm);
%     ssim_dist(i) = mean(ssimval);
%     euc_dist(i) = mean(euc);
    euc_dist_env(i) = mean(euc);
    
end
figure
plot(euc_dist_env)
%% post observation
th = zeros(1,110)+480;
dft_dist2 = dft_dist+th;
n_dfd = 1./dft_dist;
n_dtw = 1./dtw_dist/max(1./dtw_dist);
n_dfd = rescale(n_dfd,0,1);
n_dtw = rescale(n_dtw,0,1);
% scale = 0.01;
% out_dfd = awgn(n_dfd,scale);
% out_dtw = awgn(n_dfd,scale);
load('euc_dist_im.mat')

figure
plot(n_dfd, 'g', 'linewidth', 3)
hold on
plot(n_dtw, '-.b','linewidth', 3)
xlabel('Class Number','fontsize',18)
ylabel('Fidelity Score','fontsize',18)
set(gcf,'color','white')
[sorted, I] = sort(n_dtw, 'descend');
selects = zeros(1,110);
selects(I(1:15)) = n_dtw(I(1:15));
plot(selects, 'or', 'LineWidth',3)
ylim([min(n_dtw+0.01) max(n_dtw+0.01)]);
yline(selects(I(15)),'-.','linewidth',2);
ax = gca;
ax.FontSize = 18; 
xlim([0 110])
legend('DTW','DFD','Selected Signs','Min. selected score','location','best')

figure
plot(dtw_dist)
hold on
plot(dft_dist2)
xlabel('Class Number','fontsize',18)
ylabel('Distance','fontsize',18)
set(gcf,'color','white')
[sorted, I] = sort(dtw_dist, 'ascend');
selects = zeros(1,110);
selects(I(1:15)) = dtw_dist(I(1:15));
plot(selects, 'or', 'LineWidth',3)
ylim([min(dtw_dist-10) max(dtw_dist+10)]);
legend('DTW','DFD','Selected Signs')
zz = dtw_dist(dtw_dist<600);















