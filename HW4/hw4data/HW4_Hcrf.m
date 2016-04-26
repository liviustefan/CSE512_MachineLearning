classdef HW4_Hcrf
% Homework for Linear Chain Hidden CRF for ML class Spring 2016
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 08-Feb-2016
% Last modified: 08-Feb-2016
    
    methods (Static)
        
        function main()
            [trD, trLb, valD, valLb, tstD] = MHW_ChaLearnData.load3ClassData(inf);
            nState = 10; % tunable
            nClass = max(trLb(:));
            nDim = size(trD{1},1);
            featObj = HW4_HcrfFeat(nState, nClass, nDim);
            featFunc = @featObj.cmpFeatVecs;

            % Call your trainning code
            % set maxIter=100 in HW4_Hcrf.train
            % w is what we need to learn
            w = HW4_Hcrf.train(trD, trLb, featFunc, nState, nClass);
            save(sprintf('w_C%d_S%d.mat',nClass,nState),'w');
            %load(sprintf('w_C%d_S%d.mat',nClass,nState),'w');
            
            % compute objective value for training and validation data
            trObj=HW4_Hcrf.cmpObjAndGrad(trD, trLb, featFunc, nClass, nState, w, 0.001);
            valObj=HW4_Hcrf.cmpObjAndGrad(valD, valLb, featFunc, nClass, nState, w, 0.001);
            
            % Code to evaluate the prediction and compute confusion matrix
            predLb = HW4_Hcrf.predict(trD, featFunc, w, nState, nClass);
            trC = confusionmat(trLb,predLb);
            top1acc=sum(diag(trC))/sum(trC(:)) % training accuracy
            predLb = HW4_Hcrf.predict(valD, featFunc, w, nState, nClass);
            valC = confusionmat(valLb,predLb);
            top1acc=sum(diag(valC))/sum(valC(:))
            
            % Your evaluation code on test data
            predLb = HW4_Hcrf.predict(tstD, featFunc, w, nState, nClass);
            % write predicition into a csv file
            csvwrite('predTestLabels_3classes.csv',[(1:length(tstD))',predLb]);            
        end;
        
        % You need to implement this function
        function w = train(Ds, lb, featFunc, nState, nClass)
            % initialize with gaussian
            wDim = size(featFunc(Ds{1}, 1, 1),1);
            w0 = rand(wDim,1);
            
            % optimization
            lambda=0.001;
            options = optimoptions('fminunc','GradObj','on', ...
                                   'Algorithm','quasi-newton', ...
                                   'MaxIter', 100, ...
                                   'Display','iter-detailed');
            fEvalGrad = @(w)HW4_Hcrf.cmpObjAndGrad(Ds, lb, featFunc, nClass, nState, w, lambda);
            w=fminunc(fEvalGrad,w0,options);
        end;

        % You need to implement this function
        function predLb = predict(tstD, featFunc, w, nState, nClass)
            nTest=length(tstD);
            for i=1:nTest
                X=tstD{i};
                %% forward backward algorithm -> PyXs
                logUnnormPyXs=[];
                for y_=1:nClass
                    logAlphas = HW4_Hcrf.forwardBackward(X, y_, w, featFunc, nState);
                    logUnnormPyXs(y_,1)=HW4_Utils.logSumExp(logAlphas(:,end));
                end
                PyXs=HW4_Utils.logUnnormProb2NormProb(logUnnormPyXs);
                [~,predLb(i,1)]=max(PyXs);
            end
        end
        
        % Implement this function for training loss,
        % it should return the function value and gradient wrt w
        function [fVal, grad] = cmpObjAndGrad(Ds, lb, featFunc, nClass, nState, w, lambda)
            %% compute the objective value and gradient
            fVal = 0.5*lambda*w'*w; % initialize the objective value
            grad = lambda*w; % intialize the gradient wrt w
            n=length(Ds);
            for i=1:n % go over each training data
                X = Ds{i};
                y = lb(i); % true label
                
                %% forward backward algorithm -> logAlphas, logBetas, PyXs
                logAlphas={};
                logBetas={};
                logUnnormPyXs=[];
                inx=1; % shouldn't matter, verified
                for y_=1:nClass
                    [logAlphas{y_}, logBetas{y_}] = HW4_Hcrf.forwardBackward(X, y_, w, featFunc, nState);
                    logUnnormPyXs(y_,1)=HW4_Utils.logSumExp(logAlphas{y_}(:,inx)+logBetas{y_}(:,inx));
                end
                PyXs=HW4_Utils.logUnnormProb2NormProb(logUnnormPyXs);

                %% update objective value
                fVal=fVal-log(PyXs(y))/n;
                
                %% update gradient when necessary
                if nargout > 1
                    % Compute d(log Z(y,X))/dw
                    derVec={};
                    for y_=1:nClass
                        derVec{y_} = HW4_Hcrf.cmpDerOfLogZ(X, y_, w, featFunc, nState, logAlphas{y_}, logBetas{y_});
                    end
                    dLogZyXdw=derVec{y};
                    % Compute d(log Z(X))/dw
                    dLogZXdw=cat(2,derVec{:})*PyXs;
                    % update grad
                    grad = grad - (dLogZyXdw - dLogZXdw)/n;
                end
            end
        end;
        
        % Compute d(log Z(y,X))/dw
        % X: d*seqLen matrix for a time series
        % y: a scalar for class label
        % w: current weight vector
        % featFunc: a handler to a feature function
        % nState: number of hidden states        
        % logAlphas, logBetas: nState*seqLen matrixes
        % Outputs:
        %   derVec: derivative vector = d(log Z(y,X))/dw
        function derVec = cmpDerOfLogZ(X, y, w, featFunc, nState, logAlphas, logBetas)
            seqLen  = size(X,2);
            
            featVecs = featFunc(X, y, 1);
            logUnnormP = logAlphas(:,end);
            normP = HW4_Utils.logUnnormProb2NormProb(logUnnormP(:));
            derVec = featVecs*normP;
            
            for t=2:seqLen
                featVecs = featFunc(X, y, t);
                wFeats = w'*featVecs;
                wFeats = reshape(wFeats, nState, nState);
                logUnnormP = repmat(logAlphas(:,t-1), 1, nState) + ...
                    wFeats + repmat(logBetas(:,t)', nState, 1);
                                
                normP = HW4_Utils.logUnnormProb2NormProb(logUnnormP(:));
                derVec =  derVec + featVecs*normP;
            end
        end;

        
        % Forward-backward algorithm
        % X: d*seqLen matrix for a time series
        % y: a scalar for class label
        % w: current weight vector
        % featFunc: a handler to a feature function
        % nState: number of hidden states
        % Outputs:
        %   logAlphas, logBetas: nState*seqLen matrixes
        %   logAlphas(i,t) = log(alpha_t(X_t = i));
        function [logAlphas, logBetas] = forwardBackward(X, y, w, featFunc, nState)
            seqLen = size(X,2);
            [logAlphas, logBetas] = deal(zeros(nState, seqLen));
            
            %forward pass
            featVecs = featFunc(X, y, 1);
            logAlphas(:,1) = w'*featVecs;
            for t=2:seqLen                
                featVecs = featFunc(X, y, t);                
                wFeats = reshape(w'*featVecs, nState, nState);            
                A = wFeats + repmat(logAlphas(:,t-1), 1, nState);
                logAlphas(:,t) = HW4_Utils.logSumExp(A);
            end;
            
            if nargout > 1 %backward pass if necessary              
                logBetas(:, seqLen) = 0;
                for t=seqLen-1:-1:1 
                    featVecs = featFunc(X, y, t+1);
                    wFeats = reshape(w'*featVecs, nState, nState);
                    A = wFeats' + repmat(logBetas(:,t+1), 1, nState);
                    logBetas(:,t) = HW4_Utils.logSumExp(A);
                end;
            end
        end;
        
        % A demo function that show:
        % 1. How to construct the feature function
        % 2. How forward-back algorithm is called
        % 3. Test that computation of derivative is correct
        function test_cmpDerOfLogZ()
            X = rand(60, 70); % random data, nClass, nState, nDim
            nDim = size(X,1);
            nState = 12;
            nClass = 5;
            y = 2;
                        
            featObj = HW4_HcrfFeat(nState, nClass, nDim);
            featFunc = @featObj.cmpFeatVecs;            
            
            % let's figure out the dim of the feature vector
            % by calling the function and check the dimension
            featVecs = featFunc(X, 1, 1);
            wDim = size(featVecs,1);
            
            w1 = rand(wDim,1);
            [logAlphas, logBetas] = HW4_Hcrf.forwardBackward(X, y, w1, featFunc, nState);
            f1 = HW4_Utils.logSumExp(logAlphas(:,end)); % function value at w1
            
            % The derivative computed analytically
            dw1 = HW4_Hcrf.cmpDerOfLogZ(X, y, w1, featFunc, nState, logAlphas, logBetas);

            % Small deviation vector
            epsVec = 1e-2*rand(size(w1));
            
            w2 = w1 + epsVec;            
            logAlphas2 = HW4_Hcrf.forwardBackward(X, y, w2, featFunc, nState);
            f2 = HW4_Utils.logSumExp(logAlphas2(:,end)); % function value at w2
            
            f2b = f1 + dw1'*epsVec;
            fprintf('If the derivative function is correct f2 and f2b should be similar\n');
            fprintf('f1: %f, f2b: %f, f2: %f\n', f1, f2b, f2);           
        end
    end    
end
