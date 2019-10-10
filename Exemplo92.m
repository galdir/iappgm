clear
close all
load Exemplo8
colormap(map)

image(X)
pause

n= 5; w= 'db1';
[C,S] = wavedec2(X,n,w);                                     

%Manual threshold
thr=5;
[Xd,Cd,Sd,perf0,perfl2] = wdencmp('gbl',C,S,w,n,thr,'h',1);
% wavelet coefficients thresholding and computation of  2-norm recovery.                           
[thr,sorh,keepapp] = ddencmp('den','wv',X);
[Xd,Cd1,Sd1,perf0,perfl2] = wdencmp('gbl',C,S,w,n,thr,'h',1);

Xclean = waverec2(Cd,Sd,w);
Xclean1 = waverec2(Cd1,Sd1,w);
image([X,Xclean,Xclean1])

% Direct denosing from image
thr_h = [100 150];        % horizontal thresholds.              
thr_d = [1 5];        % diagonal thresholds.                
thr_v = [1 5];        % vertical thresholds.                
thr = [thr_h ; thr_d ; thr_v];                                
[Xd,cxd,lxd,perf0,perfl2] = wdencmp('lvd',X,w,2,thr,'h');

image([X(100:200,100:200),Xd(100:200,100:200)])
