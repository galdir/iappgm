%examplo matlab Deep Learning Speech Recognition

clear
close all

%Load Speech Commands Data Set
%download from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz 

datafolder = fullfile('d:\','speech_commands_v0.01');

addpath(fullfile(matlabroot,'toolbox','audio','audiodemos'))
ads = audioexample.Datastore(datafolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames', ...
    'ReadMethod','File')
ads0 = copy(ads);


%Choose Words to Recognize
commands = ["yes","no","up","down","left","right","on","off","stop","go"];
%commands = ["yes","no","up","down"];

isCommand = ismember(ads.Labels,categorical(commands));
isUnknown = ~ismember(ads.Labels,categorical([commands,"_background_noise_"]));

probIncludeUnknown = 0.1; %probabilidade de um audio ser usado no conjunto de desconhecidos
mask = rand(numel(ads.Labels),1) < probIncludeUnknown;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

ads = getSubsetDatastore(ads,isCommand|isUnknown);
countEachLabel(ads)

%Split Data into Training, Validation, and Test Sets
[adsTrain,adsValidation,adsTest] = splitData(ads,datafolder);


%Compute Speech Spectrograms
%To prepare the data for efficient training of a convolutional neural network, convert the speech waveforms to log-bark auditory spectrograms.
%Define the parameters of the spectrogram calculation. segmentDuration is the duration of each speech clip (in seconds). frameDuration is the duration of each frame for spectrogram calculation. hopDuration is the time step between each column of the spectrogram. numBands is the number of log-bark filters and equals the height of each spectrogram.
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
numBands = 40;

%Compute the spectrograms for all the training, validation, and test sets by using the supporting function speechSpectrograms. The speechSpectrograms function uses auditorySpectrogram for the spectrogram calculations. To obtain data with a smoother distribution, take the logarithm of the spectrograms using a small offset epsil.
addpath(fullfile(matlabroot,'examples','audio','main'))
epsil = 1e-6;

XTrain = speechSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
XTrain = log10(XTrain + epsil);

XValidation = speechSpectrograms(adsValidation,segmentDuration,frameDuration,hopDuration,numBands);
XValidation = log10(XValidation + epsil);

XTest = speechSpectrograms(adsTest,segmentDuration,frameDuration,hopDuration,numBands);
XTest = log10(XTest + epsil);

YTrain = adsTrain.Labels;
YValidation = adsValidation.Labels;
YTest = adsTest.Labels;


%Visualize Data
%Plot the waveforms and spectrograms of a few training examples. Play the corresponding audio clips.
specMin = min(XTrain(:));
specMax = max(XTrain(:));
idx = randperm(size(XTrain,4),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))

    subplot(2,3,i+3)
    spect = XTrain(:,:,1,idx(i));
    pcolor(spect)
    caxis([specMin+2 specMax])
    shading flat

    sound(x,fs)
    pause(2)
end

%Neural networks train most easily when their inputs have a reasonably smooth distribution and are normalized. 
%To check that data distribution is smooth, plot a histogram of the pixel values of the training data.
figure
histogram(XTrain,'EdgeColor','none','Normalization','pdf')
axis tight
ax = gca;
ax.YScale = 'log';
xlabel("Input Pixel Value")
ylabel("Probability Density")

%Add Background Noise Data
%The network should not only be able to recognize different spoken words. It should also be able to detect if a word is spoken at all, or if the input only contains background noise.
adsBkg = getSubsetDatastore(ads0, ads0.Labels=="_background_noise_");
numBkgClips = 4000;
volumeRange = [1e-4,1];

XBkg = backgroundSpectrograms(adsBkg,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands);
XBkg = log10(XBkg + epsil);


%Split the spectrograms of background noise over the training, validation, and test sets. 
%Because the _background_noise_ folder only contains about five and a half minutes of background noise, the background samples in the different data sets are highly correlated. 
%To increase the variation in the background noise, you can create your own background files and add them to the folder. To increase the robustness to noise, you can also try mixing background noise into the speech files.
numTrainBkg = floor(0.8*numBkgClips);
numValidationBkg = floor(0.1*numBkgClips);
numTestBkg = floor(0.1*numBkgClips);

XTrain(:,:,:,end+1:end+numTrainBkg) = XBkg(:,:,:,1:numTrainBkg);
XBkg(:,:,:,1:numTrainBkg) = [];
YTrain(end+1:end+numTrainBkg) = "background";

XValidation(:,:,:,end+1:end+numValidationBkg) = XBkg(:,:,:,1:numValidationBkg);
XBkg(:,:,:,1:numValidationBkg) = [];
YValidation(end+1:end+numValidationBkg) = "background";

XTest(:,:,:,end+1:end+numTestBkg) = XBkg(:,:,:,1: numTestBkg);
clear XBkg;
YTest(end+1:end+numTestBkg) = "background";

YTrain = removecats(YTrain);
YValidation = removecats(YValidation);
YTest = removecats(YTest);

%Plot the distribution of the different class labels in the training and validation sets. The test set has a very similar distribution to the validation set.
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
subplot(2,1,1)
histogram(YTrain)
title("Training Label Distribution")
subplot(2,1,2)
histogram(YValidation)
title("Validation Label Distribution")


%Add Data Augmentation
%Create an augmented image datastore for automatic augmentation and resizing of the spectrograms. 
%Translate the spectrogram randomly up to 10 frames (100 ms) forwards or backwards in time, and scale the spectrograms along the time axis up or down by 20 percent. 
%Augmenting the data somewhat increases the effective size of the training data and helps prevent the network from overfitting. 
%The augmented image datastore creates augmented images in real time and inputs these to the network. No augmented spectrograms are saved in memory.
sz = size(XTrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter(...
    'RandXTranslation',[-10 10],...
    'RandXScale',[0.8 1.2],...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain,...
    'DataAugmentation',augmenter,...
    'OutputSizeMode','randcrop');


%Define Neural Network Architecture
classNames = categories(YTrain);
classWeights = 1./countcats(YTrain);
classWeights = classWeights/mean(classWeights);
numClasses = numel(classNames);

dropoutProb = 0.2;
layers = [
    imageInputLayer(imageSize)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2,'Padding',[0,1])

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2,'Padding',[0,1])

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(dropoutProb)
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer([1 13])

    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedCrossEntropyLayer(classNames,classWeights)];


%Train Network
%Specify the training options. Use the Adam optimizer with a mini-batch size of 128 and a learning rate of 5e-4. Train for 25 epochs and reduce the learning rate by a factor of 10 after 20 epochs.
miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',5e-4, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',Inf, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20);

%Train the network. If you do not have a GPU, then training the network can take some time. To load a pretrained network instead of training a network from scratch, set doTraining to false.
doTraining = true;
if doTraining
    trainedNet = trainNetwork(augimdsTrain,layers,options);
    save('commandNet.mat','trainedNet');
else
    s = load('commandNet.mat');
    trainedNet = s.trainedNet;
end

%Evaluate Trained Network
YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

figure
plotconfusion(YValidation,YValPred,'Validation Data')

%In applications with constrained hardware resources, such as mobile applications, it is important to respect limitations on available memory and computational resources. 
%Compute the total size of the network in kilobytes, and test its prediction speed when using the CPU. 
%The prediction time is the time for classifying a single input image. 
%If you input multiple images to the network, these can be classified simultaneously, leading to shorter prediction times per image. 
%For this application, however, the single-image prediction time is the most relevant.
info = whos('trainedNet');
disp("Network size: " + info.bytes/1024 + " kB")

for i=1:100
    x = randn(imageSize);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment",'cpu');
    time(i) = toc;
end
disp("Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms")

%Detect Commands Using Streaming Audio from Microphone
%Specify the audio sampling rate and classification rate in Hz and create an audio device reader to read audio from your microphone.
fs = 16e3;
classificationRate = 20;
audioIn = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',floor(fs/classificationRate));

%Specify parameters for the streaming spectrogram computations and initialize a buffer for the audio. 
%Extract the classification labels of the network and initialize buffers of half a second for the labels and classification probabilities of the streaming audio. 
%Use these buffers to build 'agreement' over when a command is detected using multiple frames over half a second.
frameLength = frameDuration*fs;
hopLength = hopDuration*fs;
waveBuffer = zeros([fs,1]);

labels = trainedNet.Layers(end).ClassNames;
YBuffer(1:classificationRate/2) = "background";
probBuffer = zeros([numel(labels),classificationRate/2]);

%Create a figure and detect commands as long as the created figure exists. To stop the live detection, simply close the figure. Add the path of the auditorySpectrogram function that calculates the spectrograms.
h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
addpath(fullfile(matlabroot,'examples','audio','main'))

while ishandle(h)

    % Extract audio samples from audio device and add to the buffer.
    x = audioIn();
    waveBuffer(1:end-numel(x)) = waveBuffer(numel(x)+1:end);
    waveBuffer(end-numel(x)+1:end) = x;

    % Compute the spectrogram of the latest audio samples.
    spec = auditorySpectrogram(waveBuffer,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength-hopLength, ...
        'NumBands',numBands, ...
        'Range',[50,7000], ...
        'WindowType','Hann', ...
        'WarpType','Bark', ...
        'SumExponent',2);
    spec = log10(spec + epsil);

    % Classify the current spectrogram, save the label to the label buffer,
    % and save the predicted probabilities to the probability buffer.
    [YPredicted,probs] = classify(trainedNet,spec,'ExecutionEnvironment','cpu');
    YBuffer(1:end-1)= YBuffer(2:end);
    YBuffer(end) = YPredicted;
    probBuffer(:,1:end-1) = probBuffer(:,2:end);
    probBuffer(:,end) = probs';

    % Plot the current waveform and spectrogram.
    subplot(2,1,1);
    plot(waveBuffer)
    axis tight
    ylim([-0.2,0.2])

    subplot(2,1,2)
    pcolor(spec)
    caxis([specMin+2 specMax])
    shading flat

    % Now do the actual command detection by performing a very simple
    % thresholding operation. Declare a detection and display it in the
    % figure title if all of the following hold:
    % 1) The most common label is not |background|.
    % 2) At least |countThreshold| of the latest frame labels agree.
    % 3) The maximum predicted probability of the predicted label is at least |probThreshold|.
    % Otherwise, do not declare a detection.
    [YMode,count] = mode(YBuffer);
    countThreshold = ceil(classificationRate*0.2);
    maxProb = max(probBuffer(labels == YMode,:));
    probThreshold = 0.7;
    subplot(2,1,1);
    if YMode == "background" || count<countThreshold || maxProb < probThreshold
        title(" ")
    else
        title(YMode,'FontSize',20)
    end

    drawnow

end







% getSubsetDatastore(ads,indices) creates a datastore using ads that only
% contains the files and labels indexed by indices.
function dsSubset = getSubsetDatastore(ads,indices)

dsSubset = copy(ads);
dsSubset.Files  = ads.Files(indices);
dsSubset.Labels = ads.Labels(indices);

end

% splitData(ads,datafolder) splits the data store ads for the Speech
% Commands Dataset into training, validation, and test datastores based on
% the list of validation and test files validation_list.txt and
% testing_list.txt in datafolder.

function [adsTrain,adsValidation,adsTest] = splitData(ads,datafolder)

% Read the list of validation files
c = fileread(fullfile(datafolder,'validation_list.txt'));
filesValidation = string(split(c));
filesValidation  = filesValidation(filesValidation ~= "");

% Read the list of test files
c = fileread(fullfile(datafolder,'testing_list.txt'));
filesTest = string(split(c));
filesTest  = filesTest(filesTest ~= "");

% Determine which files in the datastore should go to validation set and
% which should go to test set
files = ads.Files;
sf    = split(files,filesep);
isValidation = ismember(sf(:,end-1) + "/" + sf(:,end),filesValidation);
isTest       = ismember(sf(:,end-1) + "/" + sf(:,end),filesTest);

adsTest = getSubsetDatastore(ads,isTest);
adsValidation = getSubsetDatastore(ads,isValidation);
adsTrain = getSubsetDatastore(ads,~isValidation & ~isTest);

end

% speechSpectrograms(ads,segmentDuration,frameDuration,hopDuration,numBands)
% computes speech spectrograms for the files in the datastore ads.
% segmentDuration is the total duration of the speech clips (in seconds),
% frameDuration the duration of each spectrogram frame, hopDuration the
% time shift between each spectrogram frame, and numBands the number of
% frequency bands.

function X = speechSpectrograms(ads,segmentDuration,frameDuration,hopDuration,numBands)

disp("Computing speech spectrograms...");

numHops = ceil((segmentDuration - frameDuration)/hopDuration);
numFiles = length(ads.Files);
X = zeros([numBands,numHops,1,numFiles],'single');

for i = 1:numFiles
    
    [x,info] = read(ads);
    
    fs = info.SampleRate;
    frameLength = round(frameDuration*fs);
    hopLength = round(hopDuration*fs);
    
    spec = auditorySpectrogram(x,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength - hopLength, ...
        'NumBands',numBands, ...
        'Range',[50,7000], ...
        'WindowType','Hann', ...
        'WarpType','Bark', ...
        'SumExponent',2);
    
    % If the spectrogram is less wide than numHops, then put spectrogram in
    % the middle of X.
    w = size(spec,2);
    left = floor((numHops-w)/2)+1;
    ind = left:left+w-1;
    X(:,ind,1,i) = spec;
    
    if mod(i,1000) == 0
        disp("Processed " + i + " files out of " + numFiles)
    end
    
end

disp("...done");

end

% backgroundSpectrograms(ads,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands)
% calculates numBkgClips spectrograms of background clips taken from the
% audio files in the |ads| datastore. Approximately the same number of
% clips is taken from each audio file. Before calculating spectrograms, the
% function rescales each audio clip with a factor sampled from a
% log-uniform distribution in the range given by volumeRange.
% segmentDuration is the total duration of the speech clips (in seconds),
% frameDuration the duration of each spectrogram frame, hopDuration the
% time shift between each spectrogram frame, and numBands the number of
% frequency bands.

function Xbkg = backgroundSpectrograms(ads,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands)

disp("Computing background spectrograms...");

logVolumeRange = log10(volumeRange);

numBkgFiles = numel(ads.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));

numHops = segmentDuration/hopDuration - 2;
Xbkg = zeros(numBands,numHops,1,numBkgClips,'single');

ind = 1;
for count = 1:numBkgFiles
    [wave,info] = read(ads);
    
    fs          = info.SampleRate;
    frameLength = frameDuration*fs;
    hopLength   = hopDuration*fs;
    
    for j = 1:numClipsPerFile(count)
        indStart =  randi(numel(wave)-fs);
        logVolume = logVolumeRange(1) + diff(logVolumeRange)*rand;
        volume = 10^logVolume;
        x = wave(indStart:indStart+fs-1)*volume;
        x = max(min(x,1),-1);
        
        Xbkg(:,:,:,ind) = auditorySpectrogram(x,fs, ...
            'WindowLength',frameLength, ...
            'OverlapLength',frameLength - hopLength, ...
            'NumBands',numBands, ...
            'Range',[50,7000], ...
            'WindowType','Hann', ...
            'WarpType','Bark', ...
            'SumExponent',2);
        
        if mod(ind,1000)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end

disp("...done");

end


