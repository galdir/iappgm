clear
close all

load Exemplo8
colormap(map)

nbcol = size(map,1);          
cod_X = wcodemat(X,nbcol);    
image(cod_X)

%pause
% Level 1                                 
%returns the approximation coefficients matrix cA and detail coefficients matrices 
%cH, cV, and cD (horizontal, vertical, and diagonal, respectively).
[ca1,ch1,cv1,cd1] = dwt2(X,'db1');       

cod_ca1 = wcodemat(ca1,nbcol);           
cod_ch1 = wcodemat(ch1,nbcol);           
cod_cv1 = wcodemat(cv1,nbcol);           
cod_cd1 = wcodemat(cd1,nbcol);           

figure;
colormap(map)
image([cod_ca1,cod_ch1;cod_cv1,cod_cd1]);

%pause

% Level 2        
[ca2,ch2,cv2,cd2] = dwt2(ca1,'db1');
cod_ca2 = wcodemat(ca2,nbcol);      
cod_ch2 = wcodemat(ch2,nbcol);      
cod_cv2 = wcodemat(cv2,nbcol);      
cod_cd2 = wcodemat(cd2,nbcol);      
figure;
colormap(map)
image([[cod_ca2,cod_ch2;cod_cv2,cod_cd2],cod_ch1;cod_cv1,cod_cd1]);
