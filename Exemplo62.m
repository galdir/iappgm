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
pause

a= upcoef('a',cA1,'db2',3,l(length(l)));
e= x-a; ne= norm(e)
subplot(121), plot(a), axis tight
d= upcoef('d',cD1,'db2',3,l(length(l)));
e1= e-d; ne1= norm(e1)
subplot(122), plot(d), axis tight
