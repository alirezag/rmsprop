
require 'optim'
require 'dpnn'
require  'sys'
dl = require 'dataload'
dofile('../rmsprop.lua')



trainData , testData = dl.loadMNIST();

-- set parameters
local lr = .001;
local inputsize = 784;
local numnode = 50;
local batchsize = 100;
local epochsize = trainData.inputs:size(1);
local maxepoch = 100;



-- setup the whole network 
netbp = nn.Sequential()
    :add(nn.Convert())
    :add(nn.View(784))
    :add(nn.Linear(784,numnode))
    :add(nn.ReLU())
    :add(nn.Linear(numnode,numnode))
    :add(nn.ReLU())
    :add(nn.Linear(numnode,numnode))
    :add(nn.ReLU())
    :add(nn.Linear(numnode,numnode))
    :add(nn.ReLU())
    :add(nn.Linear(numnode,10))
    :add(nn.LogSoftMax())

netrms = netbp:clone();
netcrms = netbp:clone();

paramsbp, gparamsbp = netbp:getParameters();
paramsrms, gparamsrms = netrms:getParameters();
paramscrms, gparamscrms = netcrms:getParameters();

rmsst = {}; crmsst = {};

-- criterion
CF = nn.ClassNLLCriterion()


-- setu p log
logger = optim.Logger('testdata.log')

logger:setNames{'loss bp','loss rmsprop','loss centered rmsprop'}



for epoch=1,maxepoch do 
    sys.tic();
    for j, input, target in trainData:sampleiter(batchsize,epochsize) do 
        function fevalbp(params)
            netbp:zeroGradParameters();
            netbp:forward(input)
            lossbp = CF:forward(netbp.output,target);
            dlossbp = CF:backward(netbp.output,target);
            netbp:backward(input,dlossbp); 
            return lossbp, gparamsbp;
        end

        function fevalrms(params)
            netrms:zeroGradParameters();
            netrms:forward(input)
            lossrms = CF:forward(netrms.output,target);
            dlossrms = CF:backward(netrms.output,target);
            netrms:backward(input,dlossrms); 
            return lossrms, gparamsrms;
        end
        function fevalcrms(params)
            netcrms:zeroGradParameters();
            netcrms:forward(input)
            losscrms = CF:forward(netcrms.output,target);
            dlosscrms = CF:backward(netcrms.output,target);
            netcrms:backward(input,dlosscrms); 
            return losscrms, gparamscrms;
        end


        optim.sgd(fevalbp,paramsbp,{learningRate = lr})
        optim.rmsprop(fevalrms,paramsrms,{learningRate = lr,alpha=0.95,epsilon=0.0001,center=false},rmsst)
        optim.rmsprop(fevalcrms,paramscrms,{learningRate = lr,alpha=0.95,epsilon=0.0001,center=true},crmsst)

    end
    

  
    
   
    time = sys.toc();
    logger:add{lossbp,lossrms,losscrms}
    print(epoch,lossbp,lossrms,losscrms)
end


