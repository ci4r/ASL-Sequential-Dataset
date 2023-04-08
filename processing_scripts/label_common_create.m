clc; clear; close all;

md = dir('/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/*/labels/microDoppler/*.txt');
rd = dir('/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/*/labels/rangeDoppler/*.txt');
y_md_1_1 = [];
y_md_1_2to5 = [];
y_md_2_1 = [];
y_md_2_2to5 = [];
y_md_3_1 = [];
y_md_3_2to5 = [];
y_md_4_1 = [];
y_md_4_2to5 = [];
y_md_5_1 = [];
y_md_5_2to5 = [];

y_rd_1_1 = [];
y_rd_1_2to5 = [];
y_rd_2_1 = [];
y_rd_2_2to5 = [];
y_rd_3_1 = [];
y_rd_3_2to5 = [];
y_rd_4_1 = [];
y_rd_4_2to5 = [];
y_rd_5_1 = [];
y_rd_5_2to5 = [];

for i = 1:length(md)
      fname_md = [md(i).folder '/' md(i).name];  
      fname_rd = [rd(i).folder '/' rd(i).name];  
      y_md = textread(fname_md);
      y_rd = textread(fname_rd);
      
      
      if md(i).name(4) == '1' && md(i).name(end-4) == '1'
              y_md_1_1 = cat(1, y_md, y_md_1_1);
              y_rd_1_1 = cat(1, y_rd, y_rd_1_1);
      elseif md(i).name(4) == '1' && md(i).name(end-4) ~= '1'
              y_md_1_2to5 = cat(1, y_md, y_md_1_2to5);
              y_rd_1_2to5 = cat(1, y_rd, y_rd_1_2to5);
      elseif md(i).name(4) == '2' && md(i).name(end-4) == '1'
              y_md_2_1 = cat(1, y_md, y_md_2_1);
              y_rd_2_1 = cat(1, y_rd, y_rd_2_1);
      elseif md(i).name(4) == '2' && md(i).name(end-4) ~= '1'
              y_md_2_2to5 = cat(1, y_md, y_md_2_2to5);
              y_rd_2_2to5 = cat(1, y_rd, y_rd_2_2to5);
      elseif md(i).name(4) == '3' && md(i).name(end-4) == '1'
              y_md_3_1 = cat(1, y_md, y_md_3_1);
              y_rd_3_1 = cat(1, y_rd, y_rd_3_1);
      elseif md(i).name(4) == '3' && md(i).name(end-4) ~= '1'
              y_md_3_2to5 = cat(1, y_md, y_md_3_2to5);
              y_rd_3_2to5 = cat(1, y_rd, y_rd_3_2to5);
      elseif md(i).name(4) == '4' && md(i).name(end-4) == '1'
              y_md_4_1 = cat(1, y_md, y_md_4_1);
              y_rd_4_1 = cat(1, y_rd, y_rd_4_1);
      elseif md(i).name(4) == '4' && md(i).name(end-4) ~= '1'
              y_md_4_2to5 = cat(1, y_md, y_md_4_2to5);
              y_rd_4_2to5 = cat(1, y_rd, y_rd_4_2to5);
      elseif md(i).name(4) == '5' && md(i).name(end-4) == '1'
              y_md_5_1 = cat(1, y_md, y_md_5_1);
              y_rd_5_1 = cat(1, y_rd, y_rd_5_1);
      elseif md(i).name(4) == '5' && md(i).name(end-4) ~= '1'
              y_md_5_2to5 = cat(1, y_md, y_md_5_2to5);
              y_rd_5_2to5 = cat(1, y_rd, y_rd_5_2to5);
      end
end

y_md_1_1 = mode(y_md_1_1,1);
y_md_1_2to5 = mode(y_md_1_2to5,1);
y_md_2_1 = mode(y_md_2_1,1);
y_md_2_2to5 = mode(y_md_2_2to5,1);
y_md_3_1 = mode(y_md_3_1,1);
y_md_3_2to5 = mode(y_md_3_2to5,1);
y_md_4_1 = mode(y_md_4_1,1);
y_md_4_2to5 = mode(y_md_4_2to5,1);
y_md_5_1 = mode(y_md_5_1,1);
y_md_5_2to5 = mode(y_md_5_2to5,1);

y_rd_1_1 = mode(y_rd_1_1,1);
y_rd_1_2to5 = mode(y_rd_1_2to5,1);
y_rd_2_1 = mode(y_rd_2_1,1);
y_rd_2_2to5 = mode(y_rd_2_2to5,1);
y_rd_3_1 = mode(y_rd_3_1,1);
y_rd_3_2to5 = mode(y_rd_3_2to5,1);
y_rd_4_1 = mode(y_rd_4_1,1);
y_rd_4_2to5 = mode(y_rd_4_2to5,1);
y_rd_5_1 = mode(y_rd_5_1,1);
y_rd_5_2to5 = mode(y_rd_5_2to5,1);

common_md = '/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/common labels/microDoppler/';
common_rd = '/mnt/HDD01/rspl-admin/DATASETS/Fall Sequential/Outputs/common labels/rangeDoppler/';

out_md = [common_md '1_1.txt'];
dlmwrite(out_md, y_md_1_1, 'delimiter', ' ');
out_md = [common_md '2_1.txt'];
dlmwrite(out_md, y_md_2_1, 'delimiter', ' ');
out_md = [common_md '3_1.txt'];
dlmwrite(out_md, y_md_3_1, 'delimiter', ' ');
out_md = [common_md '4_1.txt'];
dlmwrite(out_md, y_md_4_1, 'delimiter', ' ');
out_md = [common_md '5_1.txt'];
dlmwrite(out_md, y_md_5_1, 'delimiter', ' ');

out_md = [common_md '1_2to5.txt'];
dlmwrite(out_md, y_md_1_2to5, 'delimiter', ' ');
out_md = [common_md '2_2to5.txt'];
dlmwrite(out_md, y_md_2_2to5, 'delimiter', ' ');
out_md = [common_md '3_2to5.txt'];
dlmwrite(out_md, y_md_3_2to5, 'delimiter', ' ');
out_md = [common_md '4_2to5.txt'];
dlmwrite(out_md, y_md_4_2to5, 'delimiter', ' ');
out_md = [common_md '5_2to5.txt'];
dlmwrite(out_md, y_md_5_2to5, 'delimiter', ' ');

out_md = [common_rd '1_1.txt'];
dlmwrite(out_md, y_rd_1_1, 'delimiter', ' ');
out_md = [common_rd '2_1.txt'];
dlmwrite(out_md, y_rd_2_1, 'delimiter', ' ');
out_md = [common_rd '3_1.txt'];
dlmwrite(out_md, y_rd_3_1, 'delimiter', ' ');
out_md = [common_rd '4_1.txt'];
dlmwrite(out_md, y_rd_4_1, 'delimiter', ' ');
out_md = [common_rd '5_1.txt'];
dlmwrite(out_md, y_rd_5_1, 'delimiter', ' ');

out_md = [common_rd '1_2to5.txt'];
dlmwrite(out_md, y_rd_1_2to5, 'delimiter', ' ');
out_md = [common_rd '2_2to5.txt'];
dlmwrite(out_md, y_rd_2_2to5, 'delimiter', ' ');
out_md = [common_rd '3_2to5.txt'];
dlmwrite(out_md, y_rd_3_2to5, 'delimiter', ' ');
out_md = [common_rd '4_2to5.txt'];
dlmwrite(out_md, y_rd_4_2to5, 'delimiter', ' ');
out_md = [common_rd '5_2to5.txt'];
dlmwrite(out_md, y_rd_5_2to5, 'delimiter', ' ');



