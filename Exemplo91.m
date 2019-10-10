clear
close all

load Exemplo7
l= length(x);
plot(1:l,x);
axis tight
pause

% Decomposition
[C,L] = wavedec(x,3,'db1');
a3 = wrcoef('a',C,L,'db1',3);
d1 = wrcoef('d',C,L,'db1',1);
d2 = wrcoef('d',C,L,'db1',2);
d3 = wrcoef('d',C,L,'db1',3);

% Approximation and details
subplot(2,2,1); plot(a3); title('Approximation a3'); axis tight
subplot(2,2,2); plot(d1); title('Detail d1'); axis tight
subplot(2,2,3); plot(d2); title('Detail d2'); axis tight
subplot(2,2,4); plot(d3); title('Detail d3'); axis tight
pause

% Original versus Level 3 Approximation
subplot(2,1,1); plot(x);title('Original'); axis off
subplot(2,1,2); plot(a3);title('Level 3 Approximation'); axis off
pause

% Details
subplot(3,1,1); plot(d1); title('Detail Level 1'); axis off
subplot(3,1,2); plot(d2); title('Detail Level 2'); axis off
subplot(3,1,3); plot(d3); title('Detail Level 3'); axis off

% These values can be used for wdencmp with option 'gbl'.
% default for de-noising: soft thresholding and approximation coefficients kept 
% thr = sqrt(2*log(n))*s where s is an estimate of level noise and n is equal to prod(size(x)).
[thr,sorh,keepapp] = ddencmp('den','wv',x)
xclean = wdencmp('gbl',C,L,'db1',3,thr,sorh,keepapp);

subplot(3,1,1); plot(x); title('Original')
subplot(3,1,2); plot(a3); title('Level 3 Approximation')
subplot(3,1,3); plot(xclean); title('De-noised')