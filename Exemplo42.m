% Calculate STFT of a signal composed by segments with different frequencies
% TT - total time
% N - number of samples
% f - vector of frequencies
% a - window's length

%f= [20 70 120 200];%frequencias
f= [20 120 120 200];%frequencias
N=512;%quantidade de amostras
Fs= 1024;                     % Sample frequency
TT= N/Fs;%resolucao?
t = (0:N-1)/Fs;              % Time vector
faxis = 2*t/N;                % Frequency Scale %normalizando?

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
pause

%Análises STFT para diversos suportes de janela
a= [10 100 500 1000 10000];
for j=1:length(a)
    %Monta uma janela exponencial para a STFT
    win= winexp(N,TT,a(j));
    plot(t,win(1:N))
%    pause

    % Compute STFT using shiftwin function for shift the window along the time
    for i = 1:N
      X(:,i) = psd(x.*shiftwin(win,i,N));
    end

    fs=(0:N-1)*Fs/N;
    mesh(1000*t,fs(1:N/2),X(1:N/2,:))
    view(2)
    xlabel('Time (ms)')
    ylabel('Frequency (Hz)')
    title(['a= ' num2str(a(j))])
    axis([0 500 0 250])
    pause
end