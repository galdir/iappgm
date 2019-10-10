%ECG Classification Using Wavelet Features
%https://github.com/mathworks/physionet_ECG_data/

%The goal is to train a classifier to distinguish between arrhythmia (ARR), congestive heart failure (CHF), and normal sinus rhythm (NSR).

clear
close all 

tempdir='d:\';
load(fullfile(tempdir,'ECGData','ECGData.mat'))

percent_train = 70;
[trainData,testData,trainLabels,testLabels] = ...
    helperRandomSplit(percent_train,ECGData);


Ctrain = countcats(categorical(trainLabels))./numel(trainLabels).*100
Ctest = countcats(categorical(testLabels))./numel(testLabels).*100

helperPlotRandomRecords(ECGData,14);

%Feature Extraction
    %Autoregressive model (AR) coefficients of order 4 [8].
    %Shannon entropy (SE) values for the maximal overlap discrete wavelet packet transform (MODPWT) at level 4 [5].
    %Multifractal wavelet leader estimates of the second cumulant of the scaling exponents and the range of Holder exponents, or singularity spectrum [4].

timeWindow = 8192;
ARorder = 4;
MODWPTlevel = 4;
[trainFeatures,testFeatures,featureindices] = ...
    helperExtractFeatures(trainData,testData,timeWindow,ARorder,MODWPTlevel);


allFeatures = [trainFeatures;testFeatures];
allLabels = [trainLabels;testLabels];
figure
boxplot(allFeatures(:,featureindices.HRfeatures(1)),allLabels,'notch','on')
ylabel('Holder Exponent Range')
title('Range of Singularity Spectrum by Group (First Time Window)')
grid on



[p,anovatab,st] = anova1(allFeatures(:,featureindices.HRfeatures(1)),...
    allLabels);
c = multcompare(st,'display','off')



boxplot(allFeatures(:,featureindices.WVARfeatures(end-1)),allLabels,'notch','on')
ylabel('Wavelet Variance')
title('Wavelet Variance by Group')
grid on


%Signal Classification
features = [trainFeatures; testFeatures];
rng(1)
template = templateSVM(...
    'KernelFunction','polynomial',...
    'PolynomialOrder',2,...
    'KernelScale','auto',...
    'BoxConstraint',1,...
    'Standardize',true);
model = fitcecoc(...
    features,...
    [trainLabels;testLabels],...
    'Learners',template,...
    'Coding','onevsone',...
    'ClassNames',{'ARR','CHF','NSR'});
kfoldmodel = crossval(model,'KFold',5);
classLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)*100
[confmatCV,grouporder] = confusionmat([trainLabels;testLabels],classLabels);

%Precision, Recall, and F1 Score
CVTable = helperPrecisionRecall(confmatCV);
disp(CVTable)
model = fitcecoc(...
     trainFeatures,...
     trainLabels,...
     'Learners',template,...
     'Coding','onevsone',...
     'ClassNames',{'ARR','CHF','NSR'});
predLabels = predict(model,testFeatures);

correctPredictions = strcmp(predLabels,testLabels);
testAccuracy = sum(correctPredictions)/length(testLabels)*100
[confmatTest,grouporder] = confusionmat(testLabels,predLabels);

testTable = helperPrecisionRecall(confmatTest);
disp(testTable)

%Classification on Raw Data and Clustering
disp('Classification on Raw Data');
rawData = [trainData;testData];
Labels = [trainLabels;testLabels];
rng(1)
template = templateSVM(...
    'KernelFunction','polynomial', ...
    'PolynomialOrder',2, ...
    'KernelScale','auto', ...
    'BoxConstraint',1, ...
    'Standardize',true);
model = fitcecoc(...
    rawData,...
    [trainLabels;testLabels],...
    'Learners',template,...
    'Coding','onevsone',...
    'ClassNames',{'ARR','CHF','NSR'});
kfoldmodel = crossval(model,'KFold',5);
classLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)*100
[confmatCVraw,grouporder] = confusionmat([trainLabels;testLabels],classLabels);
rawTable = helperPrecisionRecall(confmatCVraw);
disp(rawTable)


disp('Classification on FFT data')
rawDataDFT = abs(fft(rawData,[],2));
rawDataDFT = rawDataDFT(:,1:2^16/2+1);
rng(1)
template = templateSVM(...
    'KernelFunction','polynomial',...
    'PolynomialOrder',2,...
    'KernelScale','auto',...
    'BoxConstraint',1,...
    'Standardize',true);
model = fitcecoc(...
    rawDataDFT,...
    [trainLabels;testLabels],...
    'Learners',template,...
    'Coding','onevsone',...
    'ClassNames',{'ARR','CHF','NSR'});
kfoldmodel = crossval(model,'KFold',5);
classLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)*100
[confmatCVDFT,grouporder] = confusionmat([trainLabels;testLabels],classLabels);
dftTable = helperPrecisionRecall(confmatCVDFT);
disp(dftTable)


rng default
eva = evalclusters(features,'kmeans','gap','KList',[1:6]);
eva


countcats(categorical(eva.OptimalY))









%funcoes auxiliares

function helperPlotRandomRecords(ECGData,randomSeed)
% This function is only intended to support the XpwWaveletMLExample. It may
% change or be removed in a future release.

if nargin==2
    rng(randomSeed)
end

M = size(ECGData.Data,1);
idxsel = randperm(M,4);
for numplot = 1:4
    subplot(2,2,numplot)
    plot(ECGData.Data(idxsel(numplot),1:3000))
    ylabel('Volts')
    if numplot > 2
        xlabel('Samples')
    end
    title(ECGData.Labels{idxsel(numplot)})
end

end



function [trainFeatures, testFeatures,featureindices] = helperExtractFeatures(trainData,testData,T,AR_order,level)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
trainFeatures = [];
testFeatures = [];

for idx =1:size(trainData,1)
    x = trainData(idx,:);
    x = detrend(x,0);
    arcoefs = blockAR(x,AR_order,T);
    se = shannonEntropy(x,T,level);
    [cp,rh] = leaders(x,T);
    wvar = modwtvar(modwt(x,'db2'),'db2');
    trainFeatures = [trainFeatures; arcoefs se cp rh wvar']; %#ok<AGROW>

end

for idx =1:size(testData,1)
    x1 = testData(idx,:);
    x1 = detrend(x1,0);
    arcoefs = blockAR(x1,AR_order,T);
    se = shannonEntropy(x1,T,level);
    [cp,rh] = leaders(x1,T);
    wvar = modwtvar(modwt(x1,'db2'),'db2');
    testFeatures = [testFeatures;arcoefs se cp rh wvar']; %#ok<AGROW>

end

featureindices = struct();
% 4*8
featureindices.ARfeatures = 1:32;
startidx = 33;
endidx = 33+(16*8)-1;
featureindices.SEfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+7;
featureindices.CP2features = startidx:endidx;
startidx = endidx+1;
endidx = startidx+7;
featureindices.HRfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+13;
featureindices.WVARfeatures = startidx:endidx;
end


function se = shannonEntropy(x,numbuffer,level)
numwindows = numel(x)/numbuffer;
y = buffer(x,numbuffer);
se = zeros(2^level,size(y,2));
for kk = 1:size(y,2)
    wpt = modwpt(y(:,kk),level);
    % Sum across time
    E = sum(wpt.^2,2);
    Pij = wpt.^2./E;
    % The following is eps(1)
    se(:,kk) = -sum(Pij.*log(Pij+eps),2);
end
se = reshape(se,2^level*numwindows,1);
se = se';
end


function arcfs = blockAR(x,order,numbuffer)
numwindows = numel(x)/numbuffer;
y = buffer(x,numbuffer);
arcfs = zeros(order,size(y,2));
for kk = 1:size(y,2)
    artmp =  arburg(y(:,kk),order);
    arcfs(:,kk) = artmp(2:end);
end
arcfs = reshape(arcfs,order*numwindows,1);
arcfs = arcfs';
end


function [cp,rh] = leaders(x,numbuffer)
y = buffer(x,numbuffer);
cp = zeros(1,size(y,2));
rh = zeros(1,size(y,2));
for kk = 1:size(y,2)
    [~,h,cptmp] = dwtleader(y(:,kk));
    cp(kk) = cptmp(2);
    rh(kk) = range(h);
end
end


function PRTable = helperPrecisionRecall(confmat)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
precisionARR = confmat(1,1)/sum(confmat(:,1))*100;
precisionCHF = confmat(2,2)/sum(confmat(:,2))*100 ;
precisionNSR = confmat(3,3)/sum(confmat(:,3))*100 ;
recallARR = confmat(1,1)/sum(confmat(1,:))*100;
recallCHF = confmat(2,2)/sum(confmat(2,:))*100;
recallNSR = confmat(3,3)/sum(confmat(3,:))*100;
F1ARR = 2*precisionARR*recallARR/(precisionARR+recallARR);
F1CHF = 2*precisionCHF*recallCHF/(precisionCHF+recallCHF);
F1NSR = 2*precisionNSR*recallNSR/(precisionNSR+recallNSR);
% Construct a MATLAB Table to display the results.
PRTable = array2table([precisionARR recallARR F1ARR;...
    precisionCHF recallCHF F1CHF; precisionNSR recallNSR...
    F1NSR],'VariableNames',{'Precision','Recall','F1_Score'},'RowNames',...
    {'ARR','CHF','NSR'});

end

function [trainData, testData, trainLabels, testLabels] = helperRandomSplit(percent_train_split,ECGData)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
    Labels = ECGData.Labels;
    Data = ECGData.Data;
    percent_train_split = percent_train_split/100;
    idxARRbegin = find(strcmpi(Labels,'ARR'),1,'first');
    idxARRend = find(strcmpi(Labels,'ARR'),1,'last');
    Narr = idxARRend-idxARRbegin+1;
    idxCHFbegin = find(strcmpi(Labels,'CHF'),1,'first');
    idxCHFend = find(strcmpi(Labels,'CHF'),1,'last');
    Nchf = idxCHFend-idxCHFbegin+1;
    idxNSRbegin = find(strcmpi(Labels,'NSR'),1,'first');
    idxNSRend = find(strcmpi(Labels,'NSR'),1,'last');
    Nnsr = idxNSRend-idxNSRbegin+1;
    % Obtain number needed for percentage split
    num_train_arr = round(percent_train_split*Narr);
    num_train_chf = round(percent_train_split*Nchf);
    num_train_nsr = round(percent_train_split*Nnsr);
    rng default;
    Parr = randperm(Narr,num_train_arr);
    Pchf = randperm(Nchf,num_train_chf);
    Pnsr = randperm(Nnsr,num_train_nsr);
    notParr = setdiff(1:Narr,Parr);
    notPchf = setdiff(1:Nchf,Pchf);
    notPnsr = setdiff(1:Nnsr,Pnsr);
    ARRdata = Data(idxARRbegin:idxARRend,:);
    ARRLabels = Labels(idxARRbegin:idxARRend);
    CHFdata = Data(idxCHFbegin:idxCHFend,:);
    CHFLabels = Labels(idxCHFbegin:idxCHFend);
    NSRdata = Data(idxNSRbegin:idxNSRend,:);
    NSRLabels = Labels(idxNSRbegin:idxNSRend);
    trainARR = ARRdata(Parr,:);
    trainARRLabels = ARRLabels(Parr);
    testARR = ARRdata(notParr,:);
    testARRLabels = ARRLabels(notParr);
    trainCHF = CHFdata(Pchf,:);
    trainCHFLabels = CHFLabels(Pchf);
    testCHF = CHFdata(notPchf,:);
    testCHFLabels = CHFLabels(notPchf);
    trainNSR = NSRdata(Pnsr,:);
    trainNSRLabels = NSRLabels(Pnsr);
    testNSR = NSRdata(notPnsr,:);
    testNSRLabels = NSRLabels(notPnsr);
    trainData = [trainARR ; trainCHF; trainNSR];
    trainLabels = [trainARRLabels ; trainCHFLabels; trainNSRLabels];
    testData = [testARR ; testCHF; testNSR];
    testLabels = [testARRLabels; testCHFLabels; testNSRLabels];
   
end