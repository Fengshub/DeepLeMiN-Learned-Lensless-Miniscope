%%
%% 
a=readtable('lenscoordinates.xls')
a=table2array(a);
a=a*1000;
%% 
p=zeros(2048,3072);
for idx=1:108
    ax=round((a(idx,2)-2000)/1.85+2048/2);ay=round((a(idx,1)-3000)/1.85+3072/2);
    if ay>=1 && ay<=3072 && ax>=1 && ax<=2048
        p(ax,ay)=idx;
%         ly=[ly,ay];
%         lx=[lx,ax];
    end
end
p2=flip(p,2);
figure
psf=double(logical(p2));
imagesc(imgaussfilt(psf,5));
[lx,ly]=find(psf~=0);
title('lens array coordinates')
