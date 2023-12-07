i1 = iread('img1_rectified.png');
i2 = iread('img2_rectified.png');
stdisp(i1, i2)
%%
% L = iread('left-fullcar.jpg', 'reduce', 2);
L = iread('left-fullcar.jpg');
% , 'mono', 'double',  
% sL = isurf(L);
% idisp(L);
% sL.plot_scale()
% R = iread('right-fullcar.jpg', 'reduce', 2);
R = iread('right-fullcar.jpg');
% sR = isurf(R);
% % idisp(R);
% % sR.plot_scale()
% m = sL.match(sR, 'top', 1000);
% F = m.ransac(@fmatrix,1e-4, 'verbose');
% % F
% [Lr,Rr] = irectify(F, m, L, R);
% 
% stdisp(Lr, Rr)
% stdisp(L, R)
% cvexRectifyStereoImages('left-fullcar.jpg', 'right-fullcar.jpg')
%%
figure 
imshow(stereoAnaglyph(L,R))
title("Composite Image (Red - Left Image, Cyan - Right Image)")
%%
L_g = rgb2gray(L);
R_g = rgb2gray(R);
blobs1 = detectSURFFeatures(L_g,MetricThreshold=2000);
blobs2 = detectSURFFeatures(R_g,MetricThreshold=2000);
%%
% figure 
% imshow(L)
% hold on
% plot(selectStrongest(blobs1,30))
% title("Thirty Strongest SURF Features")
%%
% figure
% imshow(R)
% hold on
% plot(selectStrongest(blobs2,30))
% title("Thirty Strongest SURF Features")

%%
[features1,validBlobs1] = extractFeatures(L_g,blobs1);
[features2,validBlobs2] = extractFeatures(R_g,blobs2);
indexPairs = matchFeatures(features1,features2,Metric="SAD", ...
  MatchThreshold=5);
matchedPoints1 = validBlobs1(indexPairs(:,1),:);
matchedPoints2 = validBlobs2(indexPairs(:,2),:);
% figure 
% showMatchedFeatures(L, R, matchedPoints1, matchedPoints2)
% legend("Putatively Matched Points In L","Putatively Matched Points In R")
%%
[fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
  matchedPoints1,matchedPoints2,Method="RANSAC", ...
  NumTrials=10000,DistanceThreshold=0.1,Confidence=99.99);
  
if status ~= 0 || isEpipoleInImage(fMatrix,size(L)) ...
  || isEpipoleInImage(fMatrix',size(R))
  error(["Not enough matching points were found or "...
         "the epipoles are inside the images. Inspect "...
         "and improve the quality of detected features ",...
         "and images."]);
end

inlierPoints1 = matchedPoints1(epipolarInliers, :);
inlierPoints2 = matchedPoints2(epipolarInliers, :);

% figure
% showMatchedFeatures(L, R, inlierPoints1, inlierPoints2)
% legend("Inlier Points In L","Inlier Points In R")
%%
[tform1, tform2] = estimateStereoRectification(fMatrix, ...
  inlierPoints1.Location,inlierPoints2.Location,size(R));
[I1Rect, I2Rect] = rectifyStereoImages(L,R,tform1,tform2);
figure
imshow(stereoAnaglyph(I1Rect,I2Rect))
title("Rectified Stereo Images (Red - Left Image, Cyan - Right Image)")
%%
stdisp(I1Rect, I2Rect)
%%
d = istereo(I1Rect, I2Rect, [-30, 80], 5, 'interp');
% d = istereo(Lr, Rr, [110, 200], 3, 'interp');
% max(d(:))
% min(d(:))
idisp(d, 'bar')
colormap(hot)
%%
% left params
reduction_factor = 1;
b = 0.508;
f = (.017/.0000042)/reduction_factor; % reminder: f is in pixels
car_disparity = 30;

Z_car = f * b ./ car_disparity;
Z_car
%%
% 
% left_focal_length=17 #mm
% left_pixel_size=0.0042 #mm, 4.2um
% right_focal_length=18 #mm
% right_pixel_size=0.00389 #mm, 3.89um

% % left params
% b = 0.508;
% f = (.017/.0000042)/2;
% 
% Z = f * b ./ d;
% surf(Z)
% shading interp; view(-180, 90)
% set(gca,'ZDir', 'reverse'); set(gca,'XDir', 'reverse')
% colormap(hot)
% cb = colorbar();