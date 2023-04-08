clear; clc; close all; 

main = '/media/rspl-admin/Seagate Backup Plus Drive/Sequential Fall/';
subfolds = dir(main);
subs = subfolds(4:24);
for s = 20:length(subs)
        subject = subs(s).name;
        
        data = ['/media/rspl-admin/Seagate Backup Plus Drive/Sequential Fall/' subject '/77ghz/Front/*.bin'];
        RDout = ['/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/' subject '/rangeDoppler/'];
        mDout = ['/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/' subject '/microDoppler/'];
        mDout_mti = ['/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/' subject '/microDoppler-mti/'];
        DOAout = ['/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/' subject '/rangeDOA/'];
        
        % if ~exist(RDout, 'dir')
        %        mkdir(RDout)
        % end
        if ~exist(mDout_mti, 'dir')
                mkdir(mDout_mti)
        end
        % if ~exist(DOAout, 'dir')
        %        mkdir(DOAout)
        % end
        
        files = dir(data);
        
        seqPerRecord = 5;
        
        filenames2 = {files.name};
        for z = 1:length(filenames2)
                temp{1,z} = filenames2{z}(1:end-10);
        end
        uniqs = unique(temp);
        if s==20
                strt = 6;
        else
                strt = 1;
        end
        tic
        for j = strt:length(uniqs)
                match = strfind(filenames2,uniqs{j}); % find matches
                idx = find(~cellfun(@isempty,match)); % find non-empty indices
                RDC = [];
                % concat RDCs with same names
                for r = 1:length(idx)
                        fname = fullfile(files(idx(r)).folder,files(idx(r)).name);
                        temp2 = RDC_extract(fname);
                        RDC = [RDC temp2];
                end
                % divide into sub RDCs
                numChirps = floor(size(RDC,2)/seqPerRecord);
                for r =1:seqPerRecord
                        msg = ['Processing: Subject ''' subject ''', File: ' int2str(j) ' of ' int2str(length(uniqs)) ', Part ' ...
                                num2str(r) '/' num2str(seqPerRecord)];   % loading message
                        disp(msg);
                        subRDC = RDC(:,(r-1)*numChirps+1:r*numChirps,:);
                        mD_Out = [mDout_mti uniqs{j} '_' num2str(r) '.png'];
                        RD_Out = [RDout uniqs{j} '_' num2str(r) '.avi'];
                        DOA_Out = [DOAout uniqs{j} '_' num2str(r) '.avi'];
                        %                 [cfar_bins] = RDC_to_rangeDopp(subRDC, RD_Out);
                        cfar_bins = [20 20; 33 33];
                        RDC_to_microDopp_revision(subRDC, mD_Out, cfar_bins)
                        %                 RDC_to_rangeDOA_AWR1642(subRDC, DOA_Out)
                end
        toc
                
        end
end
