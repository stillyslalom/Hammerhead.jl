function [X,Y,U,V,C,Dia,Corrplanes,uncertainty2D,SNRmetric]=PIVwindowed(im1,im2,tcorr,window,res,zpad,D,Zeromean,Peaklocator,Peakswitch,fracval,saveplane,X,Y,uncertainty,Uin,Vin)
% --- DPIV Correlation ---
imClass = 'double';

%     This file is part of prana, an open-source GUI-driven program for
%     calculating velocity fields using PIV or PTV.
%
%     Copyright (C) 2012  Virginia Polytechnic Institute and State
%     University
% 
%     Copyright 2014-2015.  Los Alamos National Security, LLC. This material was
%     produced under U.S. Government contract DE-AC52-06NA25396 for Los 
%     Alamos National Laboratory (LANL), which is operated by Los Alamos 
%     National Security, LLC for the U.S. Department of Energy. The U.S. 
%     Government has rights to use, reproduce, and distribute this software.
%     NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
%     WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF
%     THIS SOFTWARE.  If software is modified to produce derivative works,
%     such modified software should be clearly marked, so as not to confuse
%     it with the version available from LANL.
%
%     prana is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.


%convert input parameters
im1=cast(im1,imClass);
im2=cast(im2,imClass);
L=size(im1);

%convert to gridpoint list
X=X(:);
Y=Y(:);

%preallocate velocity fields and grid format
Nx = window(1);
Ny = window(2);

if nargin <=16
    Uin = zeros(length(X),1,imClass);
    Vin = zeros(length(X),1,imClass);
end

if Peakswitch
    Uin=repmat(Uin(:,1),[1 3]);
    Vin=repmat(Vin(:,1),[1 3]);
    U = zeros(length(X),3,imClass);
    V = zeros(length(X),3,imClass);
    C = zeros(length(X),3,imClass);
    Dia = zeros(length(X),3,imClass);
else
    U = zeros(length(X),1,imClass);
    V = zeros(length(X),1,imClass);
    C = zeros(length(X),1,imClass);
    Dia = zeros(length(X),1,imClass);
end

%sets up extended domain size
if zpad~=0
    Sy=2*Ny;
    Sx=2*Nx;
elseif strcmpi(tcorr,'DCC')
    Sy = res(1,2)+res(2,2)-1;
    Sx = res(1,1)+res(2,1)-1;
else
    Sy=Ny;
    Sx=Nx;
end

%fftshift indicies
fftindy = [ceil(Sy/2)+1:Sy 1:ceil(Sy/2)];
fftindx = [ceil(Sx/2)+1:Sx 1:ceil(Sx/2)];

%window masking filter
sfilt1 = windowmask([Sx Sy],[res(1, 1) res(1, 2)]);
sfilt2 = windowmask([Sx Sy],[res(2, 1) res(2, 2)]);

% sfilt12 = ifft2(fft2(sfilt2).*conj(fft2(sfilt1)));
% 
% keyboard

%correlation plane normalization function (always off)
cnorm = ones(Ny,Nx,imClass);
% s1   = fftn(sfilt1,[Sy Sx]);
% s2   = fftn(sfilt2,[Sy Sx]);
% S21  = s2.*conj(s1);
% 
% %Standard Fourier Based Cross-Correlation
% iS21 = ifftn(S21,'symmetric');
% iS21 = iS21(fftindy,fftindx);
% cnorm = 1./iS21;
% cnorm(isinf(cnorm)) = 0;


%RPC spectral energy filter
spectral = fftshift(energyfilt(Sx,Sy,D,0));

% This is a check for the fractionally weighted correlation.  We won't use
% the spectral filter with FWC or GCC
if strcmpi(tcorr,'FWC')
    frac = fracval;
    spectral = ones(size(spectral));
elseif strcmpi(tcorr,'GCC')
    frac = 1;
    spectral = ones(size(spectral));
elseif strcmpi(tcorr,'RPCG')
    %we're cheating - fracval is actually the grid angles in degrees
    grid_angle = fracval;
    frac  = 1;
    spectral = fftshift(energyfilt_grid(Sx,Sy,D,grid_angle,0));
else
    frac = 1;
end

% For dynamic rpc flip this switch which allows for dynamic calcuation of
% the spectral function using the diameter of the autocorrelation.
if strcmpi(tcorr,'DRPC')
    dyn_rpc = 1;
else
    dyn_rpc = 0;
end

if saveplane
    Corrplanes=zeros(Sy,Sx,length(X),imClass);
else
    Corrplanes = 0;
end

%to save space, initialize only the variables we will be using

if uncertainty.ppruncertainty==1
    SNRmetric.PPR         = zeros(length(X),1);
    uncertainty2D.Upprx   = zeros(length(X),1);
    uncertainty2D.Uppry   = zeros(length(X),1);
    uncertainty2D.UpprxLB = zeros(length(X),1);
    uncertainty2D.UppryLB = zeros(length(X),1);
    uncertainty2D.UpprxUB = zeros(length(X),1);
    uncertainty2D.UppryUB = zeros(length(X),1);
end
if uncertainty.miuncertainty==1
    SNRmetric.MI         = zeros(length(X),1);
    uncertainty2D.UmixLB = zeros(length(X),1);
    uncertainty2D.UmiyLB = zeros(length(X),1);
    uncertainty2D.UmixUB = zeros(length(X),1);
    uncertainty2D.UmiyUB = zeros(length(X),1);
    uncertainty2D.Autod  = zeros(length(X),1);
end
if uncertainty.mcuncertainty==1
    uncertainty2D.Ixx   = zeros(length(X),1);
    uncertainty2D.Iyy   = zeros(length(X),1);
    uncertainty2D.biasx = zeros(length(X),1);
    uncertainty2D.biasy = zeros(length(X),1);
    uncertainty2D.Neff  = zeros(length(X),1);
end
if uncertainty.imuncertainty==1
    uncertainty2D.Uimx  = zeros(length(X),1);
    uncertainty2D.Uimy  = zeros(length(X),1);
    uncertainty2D.Nump  = zeros(length(X),1);
end
            
switch upper(tcorr)
    
    %Standard Cross Correlation
    case {'SCC' 'HSSCC'}
        
        if size(im1,3) == 3
            Gens=zeros(Sy,Sx,3,imClass);
            for n=1:length(X)
                %for each new ROI, reset HSDCC option
                HSSCC = strcmpi(tcorr,'HSSCC');
                
                %apply the second order discrete window offset
                x1 = X(n) - floor(round(Uin(n))/2);
                x2 = X(n) +  ceil(round(Uin(n))/2);
                
                y1 = Y(n) - floor(round(Vin(n))/2);
                y2 = Y(n) +  ceil(round(Vin(n))/2);
                
                xmin1 = x1- ceil(Nx/2)+1;
                xmax1 = x1+floor(Nx/2);
                xmin2 = x2- ceil(Nx/2)+1;
                xmax2 = x2+floor(Nx/2);
                ymin1 = y1- ceil(Ny/2)+1;
                ymax1 = y1+floor(Ny/2);
                ymin2 = y2- ceil(Ny/2)+1;
                ymax2 = y2+floor(Ny/2);
                
                r=1;

                while r<=size(im1,3)
                    %find the image windows
                    zone1 = im1( max([1 ymin1]):min([L(1) ymax1]),max([1 xmin1]):min([L(2) xmax1]),r);
                    zone2 = im2( max([1 ymin2]):min([L(1) ymax2]),max([1 xmin2]):min([L(2) xmax2]),r);
                    
                    if size(zone1,1)~=Ny || size(zone1,2)~=Nx
                        w1 = zeros(Ny,Nx);
                        w1( 1+max([0 1-ymin1]):Ny-max([0 ymax1-L(1)]),1+max([0 1-xmin1]):Nx-max([0 xmax1-L(2)]) ) = zone1;
                        zone1 = w1;
                    end
                    if size(zone2,1)~=Ny || size(zone2,2)~=Nx
                        w2 = zeros(Ny,Nx);
                        w2( 1+max([0 1-ymin2]):Ny-max([0 ymax2-L(1)]),1+max([0 1-xmin2]):Nx-max([0 xmax2-L(2)]) ) = zone2;
                        zone2 = w2;
                    end
                    
                    if Zeromean==1
                        zone1=zone1-mean(zone1(:));
                        zone2=zone2-mean(zone2(:));
                    end
                    
                    %apply the image spatial filter
                    region1 = (zone1).*sfilt1;
                    region2 = (zone2).*sfilt2;
                    
                    %if we are still good for only 3 middle points, just do
                    %the minimum work, else use FFT to find the whole plane
                    if HSSCC
                        
                        %abuse the Gens matrix:
                        %G(1) is x=-1, y= 0
                        %G(2) is x= 0, y=-1
                        %G(3) is x= 0, y= 0
                        %G(4) is x= 0, y=+1
                        %G(5) is x=+1, y= 0
                        
                        Gens(3) = abs( sum(region1(:).*region2(:)) );
                        Gens(1) = abs( sum(sum(region1( :, 1:end-1).*region2( :, 2:end  )))+sum(region1(:,end).*region2(:,1  )) );
                        Gens(5) = abs( sum(sum(region1( :, 2:end  ).*region2( :, 1:end-1)))+sum(region1(:,1  ).*region2(:,end)) );
                        Gens(2) = abs( sum(sum(region1(1:end-1, : ).*region2(2:end  , : )))+sum(region1(end,:).*region2(1,  :)) );
                        Gens(4) = abs( sum(sum(region1(2:end  , : ).*region2(1:end-1, : )))+sum(region1(1  ,:).*region2(end,:)) );
                        
                        if max(Gens(1:5))~=Gens(3)
                            %dump the subset, and start over at first color index 
                            %(incr. at end of while loop back to 1)
                            HSSCC = 0;
                            r=0;  
                        end
                        
                    else
                        %FFTs and Cross-Correlation
                        f1   = fftn(region1,[Sy Sx]);
                        f2   = fftn(region2,[Sy Sx]);
                        P21  = f2.*conj(f1);

                        %Standard Fourier Based Cross-Correlation
                        G = ifftn(P21,'symmetric');
                        G = G(fftindy,fftindx);
                        Gens(:,:,r) = abs(G);
                    end
                    
                    r=r+1;
                end
                G = mean(Gens,3);
                
                %if we're still doing HSDCC, cheat and use cheap subpixel
                %functions - only 3pt fit works, else do full work
                if HSSCC
                    %a velocity will only be saved in the first peak
                    %location, regardless of other options selected
                    %the *(1) stubs are place holders for treating the
                    %cnorm weighting function properly.  Right now cnorm=1
                    %so it doesn't matter
                    %X:
                    lCm1 = log(G(1))*(1);
                    lC00 = log(G(3))*(1);
                    lCp1 = log(G(5))*(1);
                    U(n,1) = (lCm1-lCp1)/(2*(lCm1+lCp1-2*lC00));
                    %Y:
                    lCm1 = log(G(2))*(1);
                    %lC00 = log(G(3))*(1);
                    lCp1 = log(G(4))*(1);
                    V(n,1) = (lCm1-lCp1)/(2*(lCm1+lCp1-2*lC00));
                    
                    %save peak height at 0
                    C(n,1) = G(3);
                    %don't calculate diameters
                    Dia(n,1) = 0;
                    
                else
                    %subpixel estimation
                    [U(n,:),V(n,:),Ctemp,Dtemp]=subpixel(G,Sx,Sy,cnorm,Peaklocator,Peakswitch,D);
                    if Peakswitch
                        C(n,:)=Ctemp;
                        Dia(n,:)=Dtemp;
                    end
                    if saveplane
                        Corrplanes(:,:,n) = G;
                    end
                    
                    % Evaluate uncertainty options for SCC
                    if uncertainty.ppruncertainty==1
                        %SNR calculation the other output arguments of Cal_SNR
                        %are Maximum peak value,PRMSR,PCE,ENTROPY
                        metric='PPR';
                        PPRval = Cal_SNR(G,metric);
                        % Save the SNR metrics
                        SNRmetric.PPR(n)=PPRval;
                        % Evaluate PPR Uncertainty
                        % John J Charonko Model
                        [Ux,Uy,~,~,~,~]=calibration_based_uncertainty('PPR_Charonkomodel',PPRval,upper(tcorr));
                        uncertainty2D.Upprx(n)=Ux;
                        uncertainty2D.Uppry(n)=Uy;

                        % Xue Model
                        [~,~,UxLB,UxUB,UyLB,UyUB]=calibration_based_uncertainty('PPR_Xuemodel',PPRval,upper(tcorr));
                        uncertainty2D.UpprxLB(n)=UxLB;
                        uncertainty2D.UppryLB(n)=UyLB;
                        uncertainty2D.UpprxUB(n)=UxUB;
                        uncertainty2D.UppryUB(n)=UyUB;
                    end
                    
                end

            end
            
        else
            for n=1:length(X)
                %for each new ROI, reset HSDCC option
                HSSCC = strcmpi(tcorr,'HSSCC');
                
                %apply the second order discrete window offset
                x1 = X(n) - floor(round(Uin(n))/2);
                x2 = X(n) +  ceil(round(Uin(n))/2);
                
                y1 = Y(n) - floor(round(Vin(n))/2);
                y2 = Y(n) +  ceil(round(Vin(n))/2);
                
                xmin1 = x1- ceil(Nx/2)+1;
                xmax1 = x1+floor(Nx/2);
                xmin2 = x2- ceil(Nx/2)+1;
                xmax2 = x2+floor(Nx/2);
                ymin1 = y1- ceil(Ny/2)+1;
                ymax1 = y1+floor(Ny/2);
                ymin2 = y2- ceil(Ny/2)+1;
                ymax2 = y2+floor(Ny/2);
                
                %find the image windows
                zone1 = im1( max([1 ymin1]):min([L(1) ymax1]),max([1 xmin1]):min([L(2) xmax1]));
                zone2 = im2( max([1 ymin2]):min([L(1) ymax2]),max([1 xmin2]):min([L(2) xmax2]));
                
                if size(zone1,1)~=Ny || size(zone1,2)~=Nx
                    w1 = zeros(Ny,Nx);
                    w1( 1+max([0 1-ymin1]):Ny-max([0 ymax1-L(1)]),1+max([0 1-xmin1]):Nx-max([0 xmax1-L(2)]) ) = zone1;
                    zone1 = w1;
                end
                if size(zone2,1)~=Ny || size(zone2,2)~=Nx
                    w2 = zeros(Ny,Nx);
                    w2( 1+max([0 1-ymin2]):Ny-max([0 ymax2-L(1)]),1+max([0 1-xmin2]):Nx-max([0 xmax2-L(2)]) ) = zone2;
                    zone2 = w2;
                end
                
                if Zeromean==1
                    zone1=zone1-mean(zone1(:));
                    zone2=zone2-mean(zone2(:));
                end
                
                %apply the image spatial filter
                region1 = (zone1).*sfilt1;
                region2 = (zone2).*sfilt2;
                
                %if we are still good for only 3 middle points, just do
                %the minimum work, else use FFT to find the whole plane
                if HSSCC
                    
                    %abuse the G matrix:
                    %G(1) is x=-1, y= 0
                    %G(2) is x= 0, y=-1
                    %G(3) is x= 0, y= 0
                    %G(4) is x= 0, y=+1
                    %G(5) is x=+1, y= 0
                    
                    G = zeros(5,1);
                    G(3) = abs( sum(region1(:).*region2(:)) );
                    G(1) = abs( sum(sum(region1( :, 1:end-1).*region2( :, 2:end  )))+sum(region1(:,end).*region2(:,1  )) );
                    G(5) = abs( sum(sum(region1( :, 2:end  ).*region2( :, 1:end-1)))+sum(region1(:,1  ).*region2(:,end)) );
                    G(2) = abs( sum(sum(region1(1:end-1, : ).*region2(2:end  , : )))+sum(region1(end,:).*region2(1,  :)) );
                    G(4) = abs( sum(sum(region1(2:end  , : ).*region2(1:end-1, : )))+sum(region1(1  ,:).*region2(end,:)) );
                    
                    if max(G)~=G(3)
                        %dump the subset, and start over at first color index
                        %(incr. at end of while loop back to 1)
                        HSSCC = 0;
                    else
                        %a velocity will only be saved in the first peak
                        %location, regardless of other options selected
                        %the *(1) stubs are place holders for treating the
                        %cnorm weighting function properly.  Right now cnorm=1
                        %so it doesn't matter
                        %X:
                        lCm1 = log(G(1))*(1);
                        lC00 = log(G(3))*(1);
                        lCp1 = log(G(5))*(1);
                        U(n,1) = (lCm1-lCp1)/(2*(lCm1+lCp1-2*lC00));
                        %Y:
                        lCm1 = log(G(2))*(1);
                        %lC00 = log(G(3))*(1);
                        lCp1 = log(G(4))*(1);
                        V(n,1) = (lCm1-lCp1)/(2*(lCm1+lCp1-2*lC00));

                        %save peak height at 0
                        C(n,1) = G(3);
                        %don't calculate diameters
                        Dia(n,1) = 0;
                    end
                end
                if ~HSSCC
                    %this seems redundant, but we might have changed the
                    %value of HSDCC if the peak isn't centered
                
                    %FFTs and Cross-Correlation
                    f1   = fftn(region1,[Sy Sx]);
                    f2   = fftn(region2,[Sy Sx]);
                    P21  = f2.*conj(f1);

                    %Standard Fourier Based Cross-Correlation
                    G = ifftn(P21,'symmetric');
                    G = G(fftindy,fftindx);
                    G = abs(G);
                    
                    % Minimum value of the correlation plane is subtracted
                    G=G-min(G(:));

                    %subpixel estimation
                    [U(n,:),V(n,:),Ctemp,Dtemp,DXtemp,DYtemp]=subpixel(G,Sx,Sy,cnorm,Peaklocator,Peakswitch,D);
                    if Peakswitch
                        C(n,:)=Ctemp;
                        Dia(n,:)=Dtemp;
                    end
                    if saveplane
                        Corrplanes(:,:,n) = G;
                    end
                
                    % Evaluate uncertainty options for SCC
                    if uncertainty.ppruncertainty==1
                            %SNR calculation the other output arguments of Cal_SNR
                            %are Maximum peak value,PRMSR,PCE,ENTROPY
                        metric='PPR';
                        PPRval = Cal_SNR(G,metric);
                        % Save the SNR metrics
                        SNRmetric.PPR(n)=PPRval;
                        % Evaluate PPR Uncertainty
                        % John J Charonko Model
                        [Ux,Uy,~,~,~,~]=calibration_based_uncertainty('PPR_Charonkomodel',PPRval,upper(tcorr));
                        uncertainty2D.Upprx(n)=Ux;
                        uncertainty2D.Uppry(n)=Uy;

                        % Xue Model
                        [~,~,UxLB,UxUB,UyLB,UyUB]=calibration_based_uncertainty('PPR_Xuemodel',PPRval,upper(tcorr));
                        uncertainty2D.UpprxLB(n)=UxLB;
                        uncertainty2D.UppryLB(n)=UyLB;
                        uncertainty2D.UpprxUB(n)=UxUB;
                        uncertainty2D.UppryUB(n)=UyUB;

                    end

                    if uncertainty.miuncertainty==1
                        %Autocorrelations
                        P11 = f1.*conj(f1);
                        P22 = f2.*conj(f2);
                        Auto1 = ifftn(P11,'symmetric');
                        Auto2 = ifftn(P22,'symmetric');
                        Auto1 = Auto1(fftindy,fftindx);
                        Auto2 = Auto2(fftindy,fftindx);
                        Auto1=abs(Auto1);
                        Auto2=abs(Auto2);
                        nAuto1 = Auto1-min(Auto1(:)); % Autocorrelation plane of image 1
                        nAuto2 = Auto2-min(Auto2(:)); % Autocorrelation plane of image 2

                        % 3 pt Gaussian fit to Autocorrelation Diameter
                        [~,~,~,~,Dauto1x3,Dauto1y3,~]=subpixel(nAuto1,Sx,Sy,cnorm,1,0,D);
                        [~,~,~,~,Dauto2x3,Dauto2y3,~]=subpixel(nAuto2,Sx,Sy,cnorm,1,0,D);
                        Diap1=sqrt(Dauto1x3*Dauto1y3/2);
                        Diap2=sqrt(Dauto2x3*Dauto2y3/2);

                        %Average Autocorrelation Diameter
                        Autod=mean([Diap1 Diap2]);
                        uncertainty2D.Autod(n)=Autod;

                        %MI Calculation
                        INTS1 = max(region1(:));
                        INTS2 = max(region2(:));
                        [MIval,~,~,~,~,~] = MI_Cal_SCC(G,nAuto1,nAuto2,INTS1,INTS2,Dauto1x3,Dauto1y3,Dauto2x3,Dauto2y3,Sx,Sy,fftindx,fftindy);
                        SNRmetric.MI(n)=MIval;
                        % Estimate MI uncertainty
                        [~,~,UxLB,UxUB,UyLB,UyUB]=calibration_based_uncertainty('MI_Xuemodel',MIval,upper(tcorr));
                        uncertainty2D.UmixLB(n)=UxLB;
                        uncertainty2D.UmiyLB(n)=UyLB;
                        uncertainty2D.UmixUB(n)=UxUB;
                        uncertainty2D.UmiyUB(n)=UyUB;

                    end
                    if uncertainty.mcuncertainty==1
                        if uncertainty.miuncertainty==1
                            MIest=SNRmetric.MI(n);
                            [Ixx,Iyy,biasx,biasy,Neff,~]=Moment_of_correlation(P21,f1,f2,Sx,Sy,cnorm,D,fftindx,fftindy,G,DXtemp(1),DYtemp(1),region1,region2,MIest);

                        else %this should never happen - miuncertainty forced to 1 if mcuncertainty==1
                            MIest=-1;
                            [Ixx,Iyy,biasx,biasy,Neff,Autod]=Moment_of_correlation(P21,f1,f2,Sx,Sy,cnorm,D,fftindx,fftindy,G,DXtemp(1),DYtemp(1),region1,region2,MIest);
                            uncertainty2D.Autod(n)=Autod;
                        end
                        uncertainty2D.Ixx(n)=Ixx;
                        uncertainty2D.Iyy(n)=Iyy;
                        uncertainty2D.biasx(n)=biasx;
                        uncertainty2D.biasy(n)=biasy;
                        uncertainty2D.Neff(n)=Neff;

                    end
                    
                
                end
            end
        end
        
    %Direct Cross Correlation
    case 'DCC'
        
        % %initialize correlation tensor
        % CC = zeros(Sy,Sx,length(X),imClass);
        
        if size(im1,3) == 3
            Gens=zeros(res(1,2)+res(2,2)-1,res(1,1)+res(2,1)-1,3,imClass);
            for n=1:length(X)
                
                %apply the second order discrete window offset
                x1 = X(n) - floor(round(Uin(n))/2);
                x2 = X(n) +  ceil(round(Uin(n))/2);
                
                y1 = Y(n) - floor(round(Vin(n))/2);
                y2 = Y(n) +  ceil(round(Vin(n))/2);
                
                xmin1 = x1- ceil(res(1,1)/2)+1;
                xmax1 = x1+floor(res(1,1)/2);
                xmin2 = x2- ceil(res(2,1)/2)+1;
                xmax2 = x2+floor(res(2,1)/2);
                ymin1 = y1- ceil(res(1,2)/2)+1;
                ymax1 = y1+floor(res(1,2)/2);
                ymin2 = y2- ceil(res(2,2)/2)+1;
                ymax2 = y2+floor(res(2,2)/2);
                
                for r=1:size(im1,3)
                    %find the image windows
                    zone1 = im1( max([1 ymin1]):min([L(1) ymax1]),max([1 xmin1]):min([L(2) xmax1]),r );
                    zone2 = im2( max([1 ymin2]):min([L(1) ymax2]),max([1 xmin2]):min([L(2) xmax2]),r );
                    if size(zone1,1)~=Ny || size(zone1,2)~=Nx
                        w1 = zeros(res(1,2),res(1,1));
                        w1( 1+max([0 1-ymin1]):res(1,2)-max([0 ymax1-L(1)]),1+max([0 1-xmin1]):res(1,1)-max([0 xmax1-L(2)]) ) = zone1;
                        zone1 = w1;
                    end
                    if size(zone2,1)~=Ny || size(zone2,2)~=Nx
                        w2 = zeros(res(2,2),res(2,1));
                        w2( 1+max([0 1-ymin2]):res(2,2)-max([0 ymax2-L(1)]),1+max([0 1-xmin2]):res(2,1)-max([0 xmax2-L(2)]) ) = zone2;
                        zone2 = w2;
                    end
                    
                    if Zeromean==1
                        zone1=zone1-mean(zone1(:));
                        zone2=zone2-mean(zone2(:));
                    end
                    
                    %apply the image spatial filter
                    region1 = (zone1);%.*sfilt1;
                    region2 = (zone2);%.*sfilt2;
                                    
                    %correlation done using xcorr2 which is faster then fft's
                    %for strongly uneven windows.
                    %G = xcorr2(region2,region1);
                    % This is a stripped out version of xcorr2
                    G = conv2(region2, rot90(conj(region1),2));
                    
                    region1_std = std(region1(:));
                    region2_std = std(region2(:));
                    if region1_std == 0 || region2_std == 0
                        G = zeros(Sy,Sx);
                    else
                        G = G/std(region1(:))/std(region2(:))/length(region1(:));
                    end
                    Gens(:,:,r) = G;
                    
                    %store correlation matrix
                end
                G = mean(Gens,3);
                
                %subpixel estimation
                [U(n,:),V(n,:),Ctemp,Dtemp]=subpixel(G,Sx,Sy,cnorm,Peaklocator,Peakswitch,D);
                if Peakswitch
                    C(n,:)=Ctemp;
                    Dia(n,:)=Dtemp;
                end
                if saveplane
                    Corrplanes(:,:,n) = G;
                end
            end
        else
            for n=1:length(X)
                
                %apply the second order discrete window offset
                x1 = X(n) - floor(round(Uin(n))/2);
                x2 = X(n) +  ceil(round(Uin(n))/2);
                
                y1 = Y(n) - floor(round(Vin(n))/2);
                y2 = Y(n) +  ceil(round(Vin(n))/2);
                
                xmin1 = x1- ceil(res(1,1)/2)+1;
                xmax1 = x1+floor(res(1,1)/2);
                xmin2 = x2- ceil(res(2,1)/2)+1;
                xmax2 = x2+floor(res(2,1)/2);
                ymin1 = y1- ceil(res(1,2)/2)+1;
                ymax1 = y1+floor(res(1,2)/2);
                ymin2 = y2- ceil(res(2,2)/2)+1;
                ymax2 = y2+floor(res(2,2)/2);
                
                %find the image windows
                zone1 = im1( max([1 ymin1]):min([L(1) ymax1]),max([1 xmin1]):min([L(2) xmax1]) );
                zone2 = im2( max([1 ymin2]):min([L(1) ymax2]),max([1 xmin2]):min([L(2) xmax2]) );
                if size(zone1,1)~=res(1,2) || size(zone1,2)~=res(1,1)
                    w1 = zeros(res(1,2),res(1,1));
                    w1( 1+max([0 1-ymin1]):res(1,2)-max([0 ymax1-L(1)]),1+max([0 1-xmin1]):res(1,1)-max([0 xmax1-L(2)]) ) = zone1;
                    zone1 = w1;
                end
                if size(zone2,1)~=res(2,2) || size(zone2,2)~=res(2,1)
                    w2 = zeros(res(2,2),res(2,1));
                    w2( 1+max([0 1-ymin2]):res(2,2)-max([0 ymax2-L(1)]),1+max([0 1-xmin2]):res(2,1)-max([0 xmax2-L(2)]) ) = zone2;
                    zone2 = w2;
                end

                if Zeromean==1
                    zone1=zone1-mean(zone1(:));
                    zone2=zone2-mean(zone2(:));
                end
                
                %apply the image spatial filter
                region1 = (zone1);%.*sfilt1;
                region2 = (zone2);%.*sfilt2;
                
                %correlation done using xcorr2 which is faster then fft's
                %for strongly uneven windows.
                %G = xcorr2(region2,region1);
                % This is a stripped out version of xcorr2
                G = conv2(region2, rot90(conj(region1),2));
                
                region1_std = std(region1(:));
                region2_std = std(region2(:));
                if region1_std == 0 || region2_std == 0
                    G = zeros(Sy,Sx);
                else
                    G = G/std(region1(:))/std(region2(:))/length(region1(:));
                end

                %subpixel estimation
                [U(n,:),V(n,:),Ctemp,Dtemp]=subpixel(G,Sx,Sy,cnorm,Peaklocator,Peakswitch,D);
                if Peakswitch
                    C(n,:)=Ctemp;
                    Dia(n,:)=Dtemp;
                end
                if saveplane
                    Corrplanes(:,:,n) = G;
                end

                
            end
        end


    %Robust Phase Correlation
    case {'RPC','DRPC','GCC','FWC','RPCG'}
        
        if size(im1,3) == 3
            Gens=zeros(Sy,Sx,3,imClass);
            for n=1:length(X)
                
                %apply the second order discrete window offset
                x1 = X(n) - floor(round(Uin(n))/2);
                x2 = X(n) +  ceil(round(Uin(n))/2);
                
                y1 = Y(n) - floor(round(Vin(n))/2);
                y2 = Y(n) +  ceil(round(Vin(n))/2);
                
                xmin1 = x1- ceil(Nx/2)+1;
                xmax1 = x1+floor(Nx/2);
                xmin2 = x2- ceil(Nx/2)+1;
                xmax2 = x2+floor(Nx/2);
                ymin1 = y1- ceil(Ny/2)+1;
                ymax1 = y1+floor(Ny/2);
                ymin2 = y2- ceil(Ny/2)+1;
                ymax2 = y2+floor(Ny/2);
                
                for r=1:size(im1,3)
                    %find the image windows
                    zone1 = im1( max([1 ymin1]):min([L(1) ymax1]),max([1 xmin1]):min([L(2) xmax1]),r);
                    zone2 = im2( max([1 ymin2]):min([L(1) ymax2]),max([1 xmin2]):min([L(2) xmax2]),r);
                    if size(zone1,1)~=Ny || size(zone1,2)~=Nx
                        w1 = zeros(Ny,Nx);
                        w1( 1+max([0 1-ymin1]):Ny-max([0 ymax1-L(1)]),1+max([0 1-xmin1]):Nx-max([0 xmax1-L(2)]) ) = zone1;
                        zone1 = w1;
                    end
                    if size(zone2,1)~=Ny || size(zone2,2)~=Nx
                        w2 = zeros(Ny,Nx);
                        w2( 1+max([0 1-ymin2]):Ny-max([0 ymax2-L(1)]),1+max([0 1-xmin2]):Nx-max([0 xmax2-L(2)]) ) = zone2;
                        zone2 = w2;
                    end
                    
                    if Zeromean==1
                        zone1=zone1-mean(zone1(:));
                        zone2=zone2-mean(zone2(:));
                    end
                    
                    %apply the image spatial filter
                    region1 = (zone1).*sfilt1;
                    region2 = (zone2).*sfilt2;
                    
                    %FFTs and Cross-Correlation
                    f1   = fftn(region1,[Sy Sx]);
                    f2   = fftn(region2,[Sy Sx]);
                    P21  = f2.*conj(f1);
                    
                    %Phase Correlation
                    W = ones(Sy,Sx);
                    Wden = sqrt(P21.*conj(P21));
                    W(Wden~=0) = Wden(Wden~=0);
                    if frac ~= 1
                        R = P21./(W.^frac); %Apply fractional weighting to the normalization
                    else
                        R = P21./W;
                    end
                    
                    % If DRPC, the calculate the spectral function
                    % dynamically based on the autocorrelation
                    if dyn_rpc
                        CPS = ifftn(Wden,'symmetric');
                        [~,~,~,Drpc]=subpixel(CPS(fftindy,fftindx),Sx,Sy,cnorm,Peaklocator,0,D);
                        spectral = fftshift(energyfilt(Sx,Sy,Drpc/sqrt(2),0));
                    end
                    
                    %Robust Phase Correlation with spectral energy filter
                    G = ifftn(R.*spectral,'symmetric');
                    G = G(fftindy,fftindx);
                    Gens(:,:,r) = abs(G);
                end
                G=mean(Gens,3);
                
                %subpixel estimation
                [U(n,:),V(n,:),Ctemp,Dtemp]=subpixel(G,Sx,Sy,cnorm,Peaklocator,Peakswitch,D);
                if Peakswitch
                    C(n,:)=Ctemp;
                    Dia(n,:)=Dtemp;
                end
                if saveplane
                    Corrplanes(:,:,n) = G;
                end                
            end
            
        else
            for n=1:length(X)
                
                %apply the second order discrete window offset
                x1 = X(n) - floor(round(Uin(n))/2);
                x2 = X(n) +  ceil(round(Uin(n))/2);
                
                y1 = Y(n) - floor(round(Vin(n))/2);
                y2 = Y(n) +  ceil(round(Vin(n))/2);
                
                xmin1 = x1- ceil(Nx/2)+1;
                xmax1 = x1+floor(Nx/2);
                xmin2 = x2- ceil(Nx/2)+1;
                xmax2 = x2+floor(Nx/2);
                ymin1 = y1- ceil(Ny/2)+1;
                ymax1 = y1+floor(Ny/2);
                ymin2 = y2- ceil(Ny/2)+1;
                ymax2 = y2+floor(Ny/2);
                
                %find the image windows
                zone1 = im1( max([1 ymin1]):min([L(1) ymax1]),max([1 xmin1]):min([L(2) xmax1]));
                zone2 = im2( max([1 ymin2]):min([L(1) ymax2]),max([1 xmin2]):min([L(2) xmax2]));
                if size(zone1,1)~=Ny || size(zone1,2)~=Nx
                    w1 = zeros(Ny,Nx);
                    w1( 1+max([0 1-ymin1]):Ny-max([0 ymax1-L(1)]),1+max([0 1-xmin1]):Nx-max([0 xmax1-L(2)]) ) = zone1;
                    zone1 = w1;
                end
                if size(zone2,1)~=Ny || size(zone2,2)~=Nx
                    w2 = zeros(Ny,Nx);
                    w2( 1+max([0 1-ymin2]):Ny-max([0 ymax2-L(1)]),1+max([0 1-xmin2]):Nx-max([0 xmax2-L(2)]) ) = zone2;
                    zone2 = w2;
                end
                
                if Zeromean==1
                    zone1=zone1-mean(zone1(:));
                    zone2=zone2-mean(zone2(:));
                end
                
                %apply the image spatial filter
                region1 = (zone1).*sfilt1;
                region2 = (zone2).*sfilt2;
                
                %FFTs and Cross-Correlation
                f1   = fftn(region1,[Sy Sx]);
                f2   = fftn(region2,[Sy Sx]);
                P21  = f2.*conj(f1);
                
                %Phase Correlation
                W = ones(Sy,Sx);
                Wden = sqrt(P21.*conj(P21));
                W(Wden~=0) = Wden(Wden~=0);
                if frac ~= 1
                    R = P21./(W.^frac);%apply factional weighting to the normalization
                else
                    R = P21./W;
                end
                
                % If DRPC, the calculate the spectral function
                % dynamically based on the autocorrelation
                if dyn_rpc
                    CPS = ifftn(Wden,'symmetric');
                    [~,~,~,Drpc]=subpixel(CPS(fftindy,fftindx),Sx,Sy,cnorm,Peaklocator,0,D);
                    spectral = fftshift(energyfilt(Sx,Sy,Drpc/sqrt(2),0));
                end

                %Robust Phase Correlation with spectral energy filter
                G = ifftn(R.*spectral,'symmetric');
                G = G(fftindy,fftindx);
                G = abs(G);
                
                %subpixel estimation
                [U(n,:),V(n,:),Ctemp,Dtemp,DXtemp,DYtemp]=subpixel(G,Sx,Sy,cnorm,Peaklocator,Peakswitch,D);
                if Peakswitch
                    C(n,:)=Ctemp;
                    Dia(n,:)=Dtemp;
                end
                if saveplane
                    Corrplanes(:,:,n) = G;
                end
                
                    % Evaluate uncertainty options for RPC
                if uncertainty.ppruncertainty==1
                        %SNR calculation the other output arguments of Cal_SNR
                        %are Maximum peak value,PRMSR,PCE,ENTROPY
                    metric='PPR';
                    if Peakswitch % Can reuse secondary peak calc'd in `subpixel`
                        PPRval = Cal_SNR(G,metric,Ctemp);
                    else
                        PPRval = Cal_SNR(G,metric);
                    end
                    % Save the SNR metrics
                    SNRmetric.PPR(n)=PPRval;
                    % Evaluate PPR Uncertainty
                    % John J Charonko Model
                    [Ux,Uy,~,~,~,~]=calibration_based_uncertainty('PPR_Charonkomodel',PPRval,upper(tcorr));
                    uncertainty2D.Upprx(n)=Ux;
                    uncertainty2D.Uppry(n)=Uy;
                    
                    % Xue Model
                    [~,~,UxLB,UxUB,UyLB,UyUB]=calibration_based_uncertainty('PPR_Xuemodel',PPRval,upper(tcorr));
                    uncertainty2D.UpprxLB(n)=UxLB;
                    uncertainty2D.UppryLB(n)=UyLB;
                    uncertainty2D.UpprxUB(n)=UxUB;
                    uncertainty2D.UppryUB(n)=UyUB;
                    
                end
                
                if uncertainty.miuncertainty==1
                    %Autocorrelations
                    P11 = f1.*conj(f1);
                    P22 = f2.*conj(f2);
                    Auto1 = ifftn(P11,'symmetric');
                    Auto2 = ifftn(P22,'symmetric');
                    Auto1 = Auto1(fftindy,fftindx);
                    Auto2 = Auto2(fftindy,fftindx);
                    Auto1=abs(Auto1);
                    Auto2=abs(Auto2);
                    nAuto1 = Auto1-min(Auto1(:)); % Autocorrelation plane of image 1
                    nAuto2 = Auto2-min(Auto2(:)); % Autocorrelation plane of image 2
                    
                    % 3 pt Gaussian fit to Autocorrelation Diameter
                    [~,~,~,~,Dauto1x3,Dauto1y3,~]=subpixel(nAuto1,Sx,Sy,cnorm,1,0,D);
                    [~,~,~,~,Dauto2x3,Dauto2y3,~]=subpixel(nAuto2,Sx,Sy,cnorm,1,0,D);
                    Diap1=sqrt(Dauto1x3*Dauto1y3/2);
                    Diap2=sqrt(Dauto2x3*Dauto2y3/2);
%                     
%                     %Average Autocorrelation Diameter
                    Autod=mean([Diap1 Diap2]);
                    uncertainty2D.Autod(n)=Autod;
                    
                    %MI Calculation
                    INTS1 = max(region1(:));
                    INTS2 = max(region2(:));
                    
                    %This is redundant - already defined from above:
                    %Calculate the magnitude part of the correlation plane
                    W = ones(Sy,Sx);
                    Wden = sqrt(P21.*conj(P21));
                    W(Wden~=0) = Wden(Wden~=0);
                    
                    % Replaced in favor of above section (which is a copy)
                    % % This part should be checked original function was a
                    % % bit confusing but from the paper this should work
                    % W = ones(Sy,Sx);
                    % Wden = sqrt(P21.*conj(P21));
                    % % Wden1 = ifftn(Wden,'symmetric');
                    % W(P21~=0) = Wden(P21~=0);

                    magG = ifftn(W,'symmetric');
                    magG = magG(fftindy,fftindx);
                    
                    [MIval,~,~,~,~,~] = MI_Cal_RPC(magG,nAuto1,nAuto2,INTS1,INTS2,Dauto1x3,Dauto1y3,Dauto2x3,Dauto2y3,Sx,Sy,fftindx,fftindy);
                    SNRmetric.MI(n)=MIval;
                    % Estimate MI uncertainty
                    [~,~,UxLB,UxUB,UyLB,UyUB]=calibration_based_uncertainty('MI_Xuemodel',MIval,upper(tcorr));
                    uncertainty2D.UmixLB(n)=UxLB;
                    uncertainty2D.UmiyLB(n)=UyLB;
                    uncertainty2D.UmixUB(n)=UxUB;
                    uncertainty2D.UmiyUB(n)=UyUB;
                    
                end


                % The uncertainty estimation using Moment of Correlation
                % method for RPC needs to be figured out.
                % JJC: For now just use the naive approach:
                % P21 is stripped to R, the phase correlation for MC
                % f1,f2 are only used for autocorr if MI is missing
                % Sx,Sy are window sizes; D is RPC diameter
                % Differences in RPC MI: use magG instead of G
                % fftindx, fftindy are for fftshift
                % G is used for diameter of xcorr peak, and for MI
                % DXtemp,DYtemp are diameters of RPC xcorr peak from subpix
                % but they may have multiple peaks so just send the first
                % In MI calc, RPC uses magG, not G 
                % Problem: G needs to be magG for MI, but just G for MC.
                % Solution: switch to always computing MI outside of MC
                % function - we do the work internally if missing anyway!
                if uncertainty.mcuncertainty==1
                    if uncertainty.miuncertainty==1
                        MIest=SNRmetric.MI(n);
                        [Ixx,Iyy,biasx,biasy,Neff,~]=Moment_of_correlation(P21,f1,f2,Sx,Sy,cnorm,D,fftindx,fftindy,G,DXtemp(1),DYtemp(1),region1,region2,MIest);
                        
                    else %this should never happen - miuncertainty forced to 1 if mcuncertainty==1
                        MIest=-1;
                        [Ixx,Iyy,biasx,biasy,Neff,Autod]=Moment_of_correlation(P21,f1,f2,Sx,Sy,cnorm,D,fftindx,fftindy,G,DXtemp(1),DYtemp(1),region1,region2,MIest);
                        uncertainty2D.Autod(n)=Autod;
                    end
                    uncertainty2D.Ixx(n)=Ixx;
                    uncertainty2D.Iyy(n)=Iyy;
                    uncertainty2D.biasx(n)=biasx;
                    uncertainty2D.biasy(n)=biasy;
                    uncertainty2D.Neff(n)=Neff;
                    
                end


            end
        end
        
    otherwise
        %throw an error, we shouldn't be here
        error('invalid correlation type')

end

%add DWO to estimation
U = round(Uin)+U;
V = round(Vin)+V;

% Intializing SBRmetric and uncertainty 2D structure
%initialize an empty state in case no uncertainty analysis is done
if ~exist('SNRmetric','var')
    SNRmetric     = 0;
end
if ~exist('uncertainty2D','var')
    uncertainty2D = 0;
end

end
    