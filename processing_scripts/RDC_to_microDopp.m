function [] = RDC_to_microDopp( RDC, fOut, cfar_bins )
        
        numADCBits = 16; % number of ADC bits per sample
        SweepTime = 40e-3; % Time for 1 frame
        NTS = size(RDC,1); %256 Number of time samples per sweep
        numADCSamples = NTS;
        numTX = 2; % '1' for 1 TX, '2' for BPM
        NoC = 128;%128; % Number of chirp loops
        NPpF = numTX*NoC; % Number of pulses per frame
        numRX = 4;
        
        numLanes = 2; % do not change. number of lanes is always 4 even if only 1 lane is used. unused lanes
        % NoF = fileSize/2/NPpF/numRX/NTS; % Number of frames
        numChirps = size(RDC,2);
        NoF = round(numChirps/NPpF); % Number of frames, 4 channels, I&Q channels (2)
        dT = SweepTime/NPpF; %
        prf = 1/dT; %
        
        rp = fft(RDC(:,:,1));
        
      %% MTI Filter (not working)
      
        [m,n]=size(rp);
        %     ns = size(rp,2)+4;
        h=[1 -2 3 -2 1]';
        ns = size(rp,2)+length(h)-1;
        rngpro=zeros(m,ns);
        for k=1:m
                rngpro(k,:)=conv(h,rp(k,:,1));
        end
        
       %% MTI v2
        %     [b,a]=butter(1, 0.01, 'high'); %  4th order is 24dB/octave slope, 6dB/octave per order of n
        % %                                      [B,A] = butter(N,Wn, 'high') where N filter order, b (numerator), a (denominator), ...
        % %                                      highpass, Wn is cutoff freq (half the sample rate)
        %     [m,n]=size(rp(:,:,1));
        %     rngpro=zeros(m,n);
        %     for k=1:size(rp,1)
        %         rngpro(k,:)=filter(b,a,rp(k,:,1));
        %     end
      %% STFT
%             rBin = min(cfar_bins(1)):1+max(cfar_bins(2)); %covid 18:30, front ignore= 7:nts/2, %lab 15:31 for front
        %     for i = 1:size(cfar_bins,2) % fill zeros
        %             if cfar_bins(1,i) == 0 || cfar_bins(2,i) == 0
        %                     cfar_bins(:,i) = cfar_bins(:,i-1);
        %             end
        %     end
        
       % rBin = min(cfar_bins(cfar_bins>0)):median(cfar_bins(2,:));
        rBin=40:130;
        nfft = 2^12;window = 256;noverlap = 200;shift = window - noverlap;
        sx = myspecgramnew(sum(rngpro(rBin,:)),window,nfft,shift); % mti filter and IQ correction
        
      %% cfar bins
        %
        %     numrep = floor(ns/size(cfar_bins,2));
        %     b = ones(1,numrep);
        %     extended_bins = kron(cfar_bins,b);
        %     extended_bins(:,end+1:ns) = repmat(extended_bins(:,end),1,ns-size(extended_bins,2))+1;
        %     mask = zeros(size(rngpro));
        %     for i = 1:ns
        %             mask(extended_bins(1,i):extended_bins(2,i),i) = 1;
        %     end
        %     rngpro2 = rngpro.*mask;
        %     num_used = sum(mask);
        %     sx = myspecgramnew(sum(rngpro2)./num_used,window,nfft,shift); % mti filter and IQ correction
        
        sx2 = abs(flipud(fftshift(sx,1)));
        %% Spectrogram
        timeAxis = [1:NPpF*NoF]*SweepTime/NPpF*numTX ; % Time
        freqAxis = linspace(-prf/2,prf/2,nfft); % Frequency Axis
        figure('visible','on');
        colormap(jet(256));
        imagesc(timeAxis,[-prf/2 prf/2],20*log10(sx2./max(sx2(:))));

        %title(fOut(end-28:end-10))
          title(rBin(1))
        caxis([-40 0]) % 40
        set(gca, 'YDir','normal')
      
        axis([0 timeAxis(end) -prf/6 prf/6])
   
        set(gca,'xtick',[],'ytick',[])
        frame = frame2im(getframe(gca));
        imwrite(frame,[fOut(1:end-4) '.png']);
        close all
        
end