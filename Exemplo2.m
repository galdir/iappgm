
clear
close all
N=512;
Fs= 1024;       % sample frequecy
t= (0:N-1)/Fs;

f=[20 70 120 200];
x = [];                       % Signal
seg = length(f);               % Number of signal's segments
nseg = N/seg;                  % Size of each signal's segment
wf = 2*pi*f;                    % Angular frequency
% Create the signal
for i = 1:seg
  tseg = (1:nseg) + (i-1)*nseg;
  x = [x , sin(wf(i)*t(tseg))];
end
plot(1000*t,x)
axis tight
xlabel('time (ms)')
%pause


X = fft(x);                    % DFT of x
%Pxx = X.*conj(X)/N;            % Periodogram
%PSD = Pxx/N;                   % Power Spectrum Density
%P= sum(PSD);                % Total power

f = (0:N-1)*Fs/N;             % Frequency vector
figure();
stem(f(1:N/2),abs(X(1:N/2))/N,'.'); %half linear spectrum 
%stem(f(1:N/2),PSD(1:N/2),'.');
%title('Magnitude');
ylabel('Magnitude');
%set(gca,'XTick',[20 70 120 200]);
xlabel('frequency (Hz)');
grid
%axis([0 N/2 0 max(PSD)])
