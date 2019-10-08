% Calculate STFT of a signal composed by segments with different frequencies
% TT - total time
% N - number of samples
% f - vector of frequencies
% a - window's length

clear
N=512;
Fs= 1024;                     % Sample frequency
TT= N/Fs;
t = (0:N-1)/Fs;              % Time vector
faxis = 2*t/N;                % Frequency Scale

% Create the signal
x = sin(2*pi*20*t) + sin(2*pi*70*t) + sin(2*pi*120*t) + sin(2*pi*200*t);
plot(1000*t,x);
xlabel('Time (ms)')
axis([0 500 -4 4])
%pause

%An�lises STFT para diversos suportes de janela
a= [10 100 1000 10000];
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