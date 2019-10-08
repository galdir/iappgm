%Exemplo 1:

clear
close all

N=512; % amostras
Fs= 1024;       % taxa de amostragem / sample frequecy
t= (0:N-1)/Fs; %vetor de tempo

%gerando sinal com 4 componentes
amp=1;
x = amp*sin(2*pi*20*t+pi) + (amp*2)*sin(2*pi*70*t) + amp*sin(2*pi*120*t) + amp*sin(2*pi*200*t); 
plot(t,x)
axis tight
xlabel('time (ms)')
%pause


X = fft(x);     % DFT of x

LS = (abs(X)*2)/N;                %espectro linear
 
f = (0:N-1)*Fs/N;             % Frequency vector

figure();
stem(f(1:N/2),abs(X(1:N/2))*2/N,'.');
grid on
ylabel('Magnitude');
xlabel('frequency (Hz)')






pause

figure();
stem(f(1:N/2),LS(1:N/2),'.');
title("half spectrum");
grid on

Pxx = X.*conj(X)/N;            % Periodogram
PSD = Pxx/N;                   % Power Spectrum Density
Px = sum(PSD);                 % Total power
figure();
stem(f,PSD,'.');
stem(f(1:N/2),PSD(1:N/2),'.'); 
%title('Magnitude');
ylabel('Magnitude');
set(gca,'XTick',[20 70 120 200]);
xlabel('frequency (Hz)')
grid on
axis tight



%axis([0 N/2 0 max(PSD)])

%figure; plot(f,p*180/pi); title('Phase');
%set(gca,'XTick',[20 70 120 200]);



