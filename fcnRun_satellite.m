function info = fcnRun_satellite(varargin)

run matconvnet/matlab/vl_setupnn;
addpath matconvnet/examples;

% experiment and data paths
opts.expDir = 'data/modelzoo/satellite_UgandaSST_test';
opts.dataDir = 'data/satellite/UgandaSST_jpgs';
opts.modelPath = 'data/fcn8s-satellite/net-epoch-3.mat';
opts.modelFamily = 'matconvnet';
[opts, varargin] = vl_argparse(opts, varargin);

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = true ;
opts.vocAdditionalSegmentationsMergeMode = 2 ;
opts.gpus = [1] ;
opts = vl_argparse(opts, varargin) ;

resPath = fullfile(opts.expDir, 'results.mat') ;
if exist(resPath)
  info = load(resPath) ;
  return ;
end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1))
end

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  % run specific variables
  imdb.paths.imagesToRun = 'data/satellite/UgandaSST_jpgs/%s.jpg';
  imdb.paths.imageDir = 'data/satellite/UgandaSST_jpgs';
  imdb.images.run = dir(imdb.paths.imageDir);
  imdb.images.run = imdb.images.run(3:end);
else
  disp('IMDB.MAT NOT IN CORRECT MODELZOO DIRECTORY')
end

% Get validation subset
val = find(imdb.images.set == 2 & imdb.images.segmentation) ;
disp('loggins')

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

switch opts.modelFamily
  case 'matconvnet'
    net = load(opts.modelPath) ;
    net = dagnn.DagNN.loadobj(net.net) ;
    net.mode = 'test' ;
    for name = {'objective', 'accuracy'}
      net.removeLayer(name) ;
    end
    net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
    predVar = net.getVarIndex('prediction') ;
    inputVar = 'input' ;
    imageNeedsToBeMultiple = true ;

  case 'ModelZoo'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('upscore') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;

  case 'TVG'
    net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
    net.mode = 'test' ;
    predVar = net.getVarIndex('coarse') ;
    inputVar = 'data' ;
    imageNeedsToBeMultiple = false ;
end

if ~isempty(opts.gpus)
  gpuDevice(opts.gpus(1)) ;
  net.move('gpu') ;
end
net.mode = 'test' ;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

disp(size(imdb.images.run,1))

fileExt = '.jpg';

for i = 1:size(imdb.images.run,1)
  [~, resultname, ~] = fileparts(imdb.images.run(i).name);
  resultpath = fullfile(opts.expDir, [resultname '.png']);
  
  if ~exist(resultpath,'file')
      name = regexprep(imdb.images.run(i).name,fileExt,'') ;
      rgbPath = sprintf(imdb.paths.imagesToRun, name) ;

      % Load an image and gt segmentation
      rgb = vl_imreadjpeg({rgbPath}) ;
      rgb = rgb{1} ;

      % Subtract the mean (color)
      im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;

      % Soome networks requires the image to be a multiple of 32 pixels
      if imageNeedsToBeMultiple
        sz = [size(im,1), size(im,2)] ;
        sz_ = round(sz / 32)*32 ;
        im_ = imresize(im, sz_) ;
      else
        im_ = im ;
      end

      if ~isempty(opts.gpus)
        im_ = gpuArray(im_) ;
      end

      net.eval({inputVar, im_}) ;
      scores_ = gather(net.vars(predVar).value) ;
      [~,pred_] = max(scores_,[],3) ;

      if imageNeedsToBeMultiple
        pred = imresize(pred_, sz, 'method', 'nearest') ;
      else
        pred = pred_ ;
      end

      % Plots
      if mod(i - 1,1) == 0 || i == numel(val)

        % Print segmentation
    %     figure(100) ;clf ;
    %     displayImage(rgb/255, '', pred) ;
    %     drawnow ;
        disp(name);

        % Save segmentation
        imPath = fullfile(opts.expDir, [name '.png']) ;
        imwrite(pred,labelColors(),imPath,'png');
      end
  end  
end
disp('done!')

% -------------------------------------------------------------------------
function cmap = labelColors()
% -------------------------------------------------------------------------
N=21;
cmap = zeros(N,3);
for i=1:N
  id = i-1; r=0;g=0;b=0;
  for j=0:7
    r = bitor(r, bitshift(bitget(id,1),7 - j));
    g = bitor(g, bitshift(bitget(id,2),7 - j));
    b = bitor(b, bitshift(bitget(id,3),7 - j));
    id = bitshift(id,-3);
  end
  cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;

