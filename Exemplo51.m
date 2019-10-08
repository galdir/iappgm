clear
close all
N=512;
Fs= 1024;                     % Sample frequency
t = (0:N-1)/Fs;              % Time vector

% Create the signal
x = sin(2*pi*20*t) + sin(2*pi*70*t) + sin(2*pi*120*t) + sin(2*pi*200*t);
plot(1000*t,x);
xlabel('Time (ms)')
axis([0 500 -4 4])
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
cwt(x,'morse');
figure;
%'morse' (default) | 'amor' | 'bump'
cwt(x,'amor');
figure;
%'morse' (default) | 'amor' | 'bump'
cwt(x,'bump');




