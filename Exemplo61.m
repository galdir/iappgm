N=512;
Fs= 1024;                    % Sample frequency
t = (0:N-1)/Fs;              % Time vector

% Create the signal
x = sin(2*pi*20*t) + sin(2*pi*70*t) + sin(2*pi*120*t) + sin(2*pi*200*t);
plot(1000*t,x)
xlabel('Time (ms)')
axis tight
v= axis;
pause

[c,l]= wavedec(x,3,'db2');

% Extract aproximation and detail coefficients at levels 1, 2 and 3,
% from wavelet decomposition 
[cD1,cD2,cD3] = detcoef(c,l,[1 2 3]);
cA3= appcoef(c,l,'db2',3);
cA2= appcoef(c,l,'db2',2);
cA1= appcoef(c,l,'db2',1);

subplot(421), plot(cA1), title('cA1'), axis tight
subplot(422), plot(cD1), title('cD1'), axis tight
subplot(423), plot(cA2), title('cA2'), axis tight
subplot(424), plot(cD2), title('cD2'), axis tight
subplot(425), plot(cA3), title('cA3'), axis tight
subplot(426), plot(cD3), title('cD3'), axis tight
subplot(4,2,7:8), plot(1000*t,x), title('Signal'), axis tight
% pause
% 
% a= upcoef('a',cA1,'db2',1,l(length(l)));
% e= x-a; ne= norm(e)
% subplot(311), plot(e), axis tight
% d= upcoef('d',cD1,'db2',1,l(length(l)));
% e1= e-d; ne1= norm(e1)
% subplot(312), plot(e1), axis tight
% subplot(313), plot(1000*t,x), title('Signal'), axis tight
