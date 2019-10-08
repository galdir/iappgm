
clear
close all
f= [20 70 120 200];
N=512;
Fs= 1024;                     % Sample frequency
t = (0:N-1)/Fs;              % Time vector

x = [];                       % Signal
seg = length(f);               % Number of signal's segments
nseg = N/seg;                  % Size of each signal's segment
w = 2*pi*f;                    % Angular frequency
for i = 1:seg
  tseg = (1:nseg) + (i-1)*nseg;
  x = [x , sin(w(i)*t(tseg))];
end
plot(1000*t,x);
xlabel('Time (ms)')
axis([0 500 -1 1])
%pause

scale= (2:2:128);
c= cwt(x,scale,'db2');
subplot(311)
mesh(t*1000,scale,c)
title('DAUBECHIES')

scale= (2:2:128);
c= cwt(x,scale,'haar');
subplot(312)
mesh(t*1000,scale,c)
title('HAAR')

scale= (2:2:128);
c= cwt(x,scale,'mexh');
subplot(313)
mesh(t*1000,scale,c)
title('MEXICAN HAT')


%alternativamente usar a funcao mais recente
figure;
%'morse' (default) | 'amor' | 'bump'
cwt(x,'morse', Fs);
figure;
%'morse' (default) | 'amor' | 'bump'
cwt(x,'amor', Fs);
figure;
%'morse' (default) | 'amor' | 'bump'
cwt(x,'bump', Fs);