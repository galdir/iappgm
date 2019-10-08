% Calculate STFT of a signal composed by segments with different frequencies
% TT - total time
% N - number of samples
% f - vector of frequencies
% a - window's length
clear 
close all

f= [20 70 120 200];
%f= [20 120 120 200];
N=512;
Fs= 1024;                     % Sample frequency
TT= N/Fs;
t = (0:N-1)/Fs;              % Time vector
faxis = 2*t/N;                % Frequency Scale

x = [];                       % Signal
seg = length(f);               % Number of signal's segments
nseg = N/seg;                  % Size of each signal's segment
w = 2*pi*f;                    % Angular frequency
% Create the signal
for i = 1:seg
  tseg = (1:nseg) + (i-1)*nseg;
  x = [x , sin(w(i)*t(tseg))];
end
plot(1000*t,x);
xlabel('Time (ms)')
axis([0 500 -1 1])


%usa o programa do matlab spectrograma 
figure;
tJanela=16;
sobrepos=1;
spectrogram(x,tJanela,sobrepos,0:0.001:512,Fs);


%figure;
%cwt(x,Fs);


%pause
%alternativamente faz a stft "manualmente" criando uma janela e executando
%a fft via psd (powerspectrum density) com janela deslizante

%Monta uma janela exponencial para a STFT
win= winexp(N,TT,5000);
figure;
plot(t,win(1:N))

% Compute STFT using shiftwin function for shift the window along the time
for i = 1:N
  X(:,i) = psd(x.*shiftwin(win,i,N));
end

fs=(0:N-1)*Fs/N;
figure;
mesh(1000*t,fs(1:N/2),X(1:N/2,:))
view(2)
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
axis([0 500 0 250])
