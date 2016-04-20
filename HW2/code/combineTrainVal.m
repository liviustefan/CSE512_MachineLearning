run ../../vlfeat/toolbox/vl_setup.m
mkdir('../hw2data/trainvalIms/');
imFiles=dir('../hw2data/trainIms/*.jpg');
nTrain=length(imFiles);
for i=1:nTrain
    system(sprintf('cp ../hw2data/trainIms/%04d.jpg ../hw2data/trainvalIms/%04d.jpg',i,i));
end
imFiles=dir('../hw2data/valIms/*.jpg');
nVal=length(imFiles);
for i=1:nVal
    system(sprintf('cp ../hw2data/valIms/%04d.jpg ../hw2data/trainvalIms/%04d.jpg',i,i+nTrain));
end

T=load('../hw2data/trainAnno.mat');
V=load('../hw2data/valAnno.mat');
ubAnno=[T.ubAnno,V.ubAnno];
save('../hw2data/trainvalAnno.mat','ubAnno');