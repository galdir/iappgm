% Calculate STFT of a signal composed by segments with different frequencies
% TT - total time
% N - number of samples
% f - vector of frequencies
% a - fator de espalhamento

clear 
close all

N=512;
Fs= 1024;                     % Sample frequency
TT= N/Fs;                   %tempo total
t = (0:N-1)/Fs;              % Time vector
faxis = 2*t/N;                % Frequency Scale

% Angular frequency
% Create the signal
x = sin(2*pi*20*t) + sin(2*pi*70*t) + sin(2*pi*120*t) + sin(2*pi*200*t);
plot(1000*t,x);
xlabel('Time (ms)')
axis([0 500 -4 4])
%pause

%usa o programa do matlab spectrograma 
figure;
tJanela=64;
sobrepos=1;
spectrogram(x,tJanela,sobrepos,N,Fs);


pause
%alternativamente faz a stft "manualmente" criando uma janela e executando
%a fft via psd (powerspectrum density) com janela deslizante

%Monta uma janela exponencial para a STFT
a=5000;                     %fator de espalhamento
win= winexp(N,TT,a);
figure;
plot(t,win(1:N))
%pause

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


