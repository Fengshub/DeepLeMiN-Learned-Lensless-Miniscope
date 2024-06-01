%% calibrated miniscope frame coordinates
cali_angle=0;
cali_FOV_x=1376;
cali_FOV_y=2064;
x_half=1370;y_half=1950;
rmin=cali_FOV_x-x_half;rmax=cali_FOV_x+x_half;
cmin=cali_FOV_y-y_half;cmax=cali_FOV_y+y_half;
xof=268;
mini_FOV_x=1323;mini_x_half=1024+xof;
mini_FOV_y=1988;mini_y_half=1536+xof;
mini_rmin=mini_FOV_x-mini_x_half+1;mini_rmax=mini_FOV_x+mini_x_half;
mini_cmin=mini_FOV_y-mini_y_half+1;mini_cmax=mini_FOV_y+mini_y_half;
disidx=1;
num=1;
%% 
ratio=0.226; % calibrated ground truth training image size & pixel scaling ratio
Y=zeros(1,619,882);
%% read lens unit coordinates
sup_psf;
%% 
lxm=lx+xof;
lym=ly+xof;
%%
yscale=(6000/(1200/682)*ratio)/3072;
gxof=(size(Y,2)-(4000/(1200/682)*ratio))/2
gyof=(size(Y,3)-(6000/(1200/682)*ratio))/2
lxg=round(lx*yscale+gxof);
lyg=round(ly*yscale+gyof);
%%
Xtssize=360;Ytssize=90; % half side length of raw measurement and reconstruction patches in pixel
Yc=zeros(size(Y,2),size(Y,3));
    for lid=1:108
        Yc(lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1)=Yc(lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1)+1;
    end
%% plot combined reconstruction FOV patches
Yv=zeros(size(Y,2),size(Y,3));
load(['gen_lenstissue.mat'])
gen=double(squeeze(generated_images));
for lid=1:108
    Yv(lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1)=Yv(lxg(lid)-Ytssize:lxg(lid)+Ytssize-1,lyg(lid)-Ytssize:lyg(lid)+Ytssize-1)+gen(:,:,lid);
end
%
% close all
figure
Yvv=(Yv./max(Yc,1));
imagesc(Yvv)
daspect([1 1 1])
