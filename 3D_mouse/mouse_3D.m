%% load reconstructed 3D video
load('gen_img_recd_video0003 24-04-04 18-31-11_abetterrecordlong_03560_1_290_v4.mat')
gen=squeeze(generated_images_fu);
%% assemble 6-FOV into single FOV 
lx=[512 1536 512 1536 512 1536];
ly=[512 1536 2560 512 1536 2560];
xof=268;

lxm=lx+xof;
lym=ly+xof;

ratio=0.18088;
Y=zeros(905,1339);
yscale=(6000/(600/682)*ratio)/3072;
gxof=(size(Y,1)-(4000/(600/682)*ratio))/2;
gyof=(size(Y,2)-(6000/(600/682)*ratio))/2;
lxg=round(lx*yscale+gxof);
lyg=round(ly*yscale+gyof);

lidn=6;
Xtssize=704;Ytssize=208;
% Xts=zeros(2,Xtssize*2,Xtssize*2,lidn);
Yc=zeros(905,1339);
base=0;
for nid=1:1
    disp(nid)
    for lidid=1:lidn
        lid=lidid+base;
%         Xts(nid,:,:,lidid)=imrotate(squeeze(X(nid,lxm(lid)-Xtssize:lxm(lid)+Xtssize-1,lym(lid)-Xtssize:lym(lid)+Xtssize-1)),180);
        Yc(lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1)=Yc(lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1)+1;
    end
end
%% generate 3D reconstructed video
vfn=290;
Yv=zeros([vfn,size(Yc,1),size(Yc,2),13],'uint8');
parfor fid=1:vfn
    Yvtemp=zeros([1,size(Yc,1),size(Yc,2),13]);
    disp(fid)
    for lid=1:6
        temp=double(squeeze(gen(fid,lid,:,:,:)));
        tempr=reshape(temp,[1 416 416 13]);
        Yvtemp(1,lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1,:)=Yvtemp(1,lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1,:)+tempr;%imgaussfilt3(squeeze(gen(lid-9*(gid-1),:,:,:)),1);
    end
    for idx=1:13
        Yvtemp(1,:,:,idx)=squeeze(Yvtemp(1,:,:,idx))./Yc;
    end
    Yv(fid,:,:,:)=uint8(Yvtemp(1,:,:,:));
end
%% crop signal area with fluorescent opsin injection (neural activity)
Yv2=Yv(:,101:600,260-249:260+250,:);
Yv2m=Yv2(:,90:428,37:434,:);

%% temporal correlation map
Yv2m2=Yv2m;
Yv2mte=zeros([size(Yv2,2),size(Yv2,3),size(Yv2,4)]);
for idxd=1:13
    for idx1=11:size(Yv2,2)-10 % reduced calculation area to increase speed
        for idx2=11:size(Yv2,3)-10
            corrtmax=[];
            traceref=squeeze(Yv2m2(:,idx1,idx2,idxd));
            
            tracetar=squeeze(Yv2m2(:,idx1-1,idx2,idxd));
            corrt=xcorr(traceref,tracetar);
            corrtmax=[corrtmax,max(corrt)];
            tracetar=squeeze(Yv2m2(:,idx1+1,idx2,idxd));
            corrt=xcorr(traceref,tracetar);
            corrtmax=[corrtmax,max(corrt)];
            tracetar=squeeze(Yv2m2(:,idx1,idx2-1,idxd));
            corrt=xcorr(traceref,tracetar);
            corrtmax=[corrtmax,max(corrt)];
            tracetar=squeeze(Yv2m2(:,idx1,idx2+1,idxd));
            corrt=xcorr(traceref,tracetar);
            corrtmax=[corrtmax,max(corrt)];
            
            if idxd>1
                tracetar=squeeze(Yv2m2(:,idx1,idx2,idxd-1));
                corrt=xcorr(traceref,tracetar);
                corrtmax=[corrtmax,max(corrt)];
            end
            if idxd<13
                tracetar=squeeze(Yv2m2(:,idx1,idx2,idxd+1));
                corrt=xcorr(traceref,tracetar);
                corrtmax=[corrtmax,max(corrt)];
            end
%             tracetar=squeeze(Yv2m2(:,idx1-ofs:idx1+ofs,idx2-ofs:idx2+ofs,max(idxd-1,0):min(idxd+1,13)));
%             tracetar=squeeze(Yv2m2(:,idx1-ofs:idx1+ofs,idx2-ofs:idx2+ofs,max(idxd-1,0):min(idxd+1,13)));
            
%             tracetar=squeeze(mean(tracetar,[2 3]));
%             corrt=xcorr(traceref,tracetar);
%             Yv2mte(idx1,idx2,idxd)=max(corrt);

            Yv2mte(idx1,idx2,idxd)=mean(corrtmax);
        end
    end
end


%% 3D iterative clustering
%% iterative thresholding
recon_rec3=Yv2mte;
th1=0.005;
th2=15;
recon_rec=recon_rec3./max(recon_rec3(:));
CC = bwconncomp(recon_rec,26);
recon_rec5=zeros(size(recon_rec));
for th=0.02:0.02:0.5
    recon_rec=recon_rec3./max(recon_rec3(:));
    for idx=1:CC.NumObjects
        temp=CC.PixelIdxList{idx};
        if length(temp)<th2
            recon_rec(temp)=0;
        else
            recon_rec(temp)=recon_rec(temp)/max(recon_rec(temp));
        end
    end
    recon_rec(recon_rec<th)=0;
    disp(th)
    CC = bwconncomp(recon_rec,26)
    recon_rec4=zeros(size(recon_rec));
    for idx=1:CC.NumObjects
        temp=CC.PixelIdxList{idx};
            recon_rec(temp)=recon_rec(temp)+rand(size(temp))*0.005;
            a=find(recon_rec==max(recon_rec(temp)));%a=a(1);
            recon_rec4(a)=recon_rec3(a);%/(length(temp)/50).^0.3;
            recon_rec5(a)=recon_rec3(a);
%         end
    end
end
%% remove redundant clusters that existed in multiple thresholding
recon_rec3=recon_rec5;
th1=0.005;
th2=0;
recon_rec=recon_rec3./max(recon_rec3(:));
CC = bwconncomp(recon_rec,26);
recon_rec5=zeros(size(recon_rec));
for th=0.5
    recon_rec=recon_rec3./max(recon_rec3(:));
    for idx=1:CC.NumObjects
        temp=CC.PixelIdxList{idx};
        if length(temp)<th2
            recon_rec(temp)=0;
        else
            recon_rec(temp)=recon_rec(temp)/max(recon_rec(temp));
        end
    end
    recon_rec(recon_rec<th)=0;
    disp(th)
    CC = bwconncomp(recon_rec,26)
    recon_rec4=zeros(size(recon_rec));
    for idx=1:CC.NumObjects
        temp=CC.PixelIdxList{idx};
            recon_rec(temp)=recon_rec(temp)+rand(size(temp))*0.005;
            a=find(recon_rec==max(recon_rec(temp)));%a=a(1);
            recon_rec4(a)=recon_rec3(a);%/(length(temp)/50).^0.3;
            recon_rec5(a)=recon_rec3(a);
    end
end
%% 
recon_rec5(recon_rec5<2e4)=0; %Yv2mte_42

%% calculate lateral and axial span of each clustered center (total 151 clusters)
%%
recon_rec5s=sum(recon_rec5,3);
Yv2m2=Yv2mte;
recon_rec5s(1:15,:)=0;

[cx,cy]=find(recon_rec5s~=0);
ofs=5;
hi=zeros(length(cx),1);
for idx=1:length(cx)
    tempc=squeeze(recon_rec5(cx(idx),cy(idx),:));
    cmax=find(tempc==max(tempc));
%     traceref=squeeze(Yv2m2(:,cx(idx),cy(idx),cmax));
    xhi=zeros(ofs*2+1,1);
    xhi(ofs+1)=Yv2m2(cx(idx),cy(idx),cmax);
    for xidx=cx(idx)-ofs:cx(idx)+ofs
        xhi(xidx-cx(idx)+ofs+1)=Yv2m2(xidx,cy(idx),cmax);
%         tracetar=squeeze(Yv2m2(:,xidx,cy(idx),cmax));
%         corrt=xcorr(traceref,tracetar);
%         xhi(xidx-cx(idx)+ofs+1)=max(corrt);
    end
    yhi=zeros(ofs*2+1,1);
    yhi(ofs+1)=Yv2m2(cx(idx),cy(idx),cmax);
    for yidx=cy(idx)-ofs:cy(idx)+ofs
        yhi(yidx-cy(idx)+ofs+1)=Yv2m2(cx(idx),yidx,cmax);
%         tracetar=squeeze(Yv2m2(:,cx(idx),yidx,cmax));
%         corrt=xcorr(traceref,tracetar);
%         yhi(yidx-cy(idx)+ofs+1)=max(corrt);
    end
%     hi(idx)=min(length(find(xhi>max(xhi)/2))-1,length(find(yhi>max(yhi)/2))-1);
    hi(idx)=min(length(find(xhi>max(xhi)/2)),length(find(yhi>max(yhi)/2)));
    if hi(idx)==1
        bb=1;
    end
end
%%
figure
histogram(hi*4.86)
% axis off
xlim([0 45])
grid on
ax=gca;
ax.FontSize=20;
title('lateral fwhm')
xlabel('\mum')
ylabel('number of clusters')
%%
recon_rec5s=sum(recon_rec5,3);
[cx,cy]=find(recon_rec5s~=0);
lfwhm=zeros(length(cx),1);
cz=zeros(length(cx),1);
for idx=1:length(cx)
    ctemp=find(recon_rec5(cx(idx),cy(idx),:)~=0);
    cz(idx)=ctemp(1);
    tar=Yv2mte(cx(idx),cy(idx),cz(idx));
    ridx=0;
    for zidx=cz(idx):-1:1
        if Yv2mte(cx(idx),cy(idx),zidx)>tar/2
            ridx=ridx+1;
        else
            break;
        end
    end
    for zidx=cz(idx):1:13
        if Yv2mte(cx(idx),cy(idx),zidx)>tar/2
            ridx=ridx+1;
        else
            break;
        end
    end
    lfwhm(idx)=ridx;
end
histogram(lfwhm*50,[100 150 200 250 300])
xlim([100 300])
title('axial fwhm')
xlabel('\mum')
ylabel('number of clusters')
