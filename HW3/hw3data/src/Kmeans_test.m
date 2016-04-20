%% load data
X = load('../digits/digit.txt');
X = X';
Y = load('../digits/labels.txt');
[d,n]=size(X);

%% 3.5.1~2
for k=[2,4,6]
    %% kmeans
    opts=[];
    opts.maxIter=20;
    opts.verbose=0;
    opts.centerID=1:k;
    [C, A, iter]=Kmeans(X, k, opts);
    
    %% within-group sum-of-squares
    SS=nan(1,k);
    for i=1:k
        inx=find(A==i);
        SS(i)=sum(pdist2(X(:,inx)',C(:,i)','euclidean').^2);
    end
    totalSS=sum(SS);
    
    %% pair-counting meansure
    sameClass=(pdist2(Y,Y,@(x,y) x-y))==0;
    sameCluster=(pdist2(A,A,@(x,y) x-y))==0;
    sameClassSameCluster = sum(sum(sameClass & sameCluster))-n;
    sameClassDiffCluster = sum(sum(sameClass & ~sameCluster));
    diffClassSameCluster = sum(sum(~sameClass & sameCluster))-n;
    diffClassDiffCluster = sum(sum(~sameClass & ~sameCluster));
    p1=sameClassSameCluster/(sameClassSameCluster+sameClassDiffCluster);
    p2=diffClassDiffCluster/(diffClassDiffCluster+diffClassSameCluster);
    p3=(p1+p2)/2;
    
    %% report results
    fprintf('\nWhen k = %d:\n',k);
    fprintf('\tKmeans converges at iter = %d;\n',iter);
    fprintf('\tThe total within sum of squares is %.2f;\n',totalSS);
    fprintf('\tp1 = %.2f%%, p2 = %.2f%%, p3 = %.2f%%.\n\n',p1*100, p2*100, p3*100);    
end

%% 3.5.3~4
totalSS=[];
p1=[];p2=[];
for k=1:10
    for time=1:10
        %% kmeans
        opts=[];
        opts.maxIter=20;
        opts.verbose=0;
        %opts.centerID=1:k;
        [C, A, iter]=Kmeans(X, k, opts);

        %% within-group sum-of-squares
        SS=nan(1,k);
        for i=1:k
            inx=find(A==i);
            SS(i)=sum(pdist2(X(:,inx)',C(:,i)','euclidean').^2);
        end
        totalSS(k,time)=sum(SS);

        %% pair-counting meansure
        sameClass=(pdist2(Y,Y,@(x,y) x-y))==0;
        sameCluster=(pdist2(A,A,@(x,y) x-y))==0;
        sameClassSameCluster = sum(sum(sameClass & sameCluster))-n;
        sameClassDiffCluster = sum(sum(sameClass & ~sameCluster));
        diffClassSameCluster = sum(sum(~sameClass & sameCluster))-n;
        diffClassDiffCluster = sum(sum(~sameClass & ~sameCluster));
        p1(k,time)=sameClassSameCluster/(sameClassSameCluster+sameClassDiffCluster);
        p2(k,time)=diffClassDiffCluster/(diffClassDiffCluster+diffClassSameCluster);
    end
end
p3=(p1+p2)/2;

figure(1);
plot(mean(totalSS,2),'*-');
xlabel('K'); legend('SS_{total}');

figure(2);
plot(squeeze(mean(cat(3,p1,p2,p3),2)),'*-')
xlabel('K'); legend('p_1','p_2','p_3');
