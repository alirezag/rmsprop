# RMSPROP for Torch
Torch implementation of RMSPROP. The optim.rmsprop doesn't offer centered version of the RMSPROP optimization. 
I'm extending the original optim.rmsprop here as a function. The center config by default is true. You can switch it to false.

<br>
Author: Alireza Goudarzi <br>
Email: alireza.goudarzi@riken.jp <br>
<br>


## Example: 

'''

      function eval(params)
             allgrads:zero();
             out = net:forward(input);
             loss = criterion:forward(out,target);
             dloss = criterion:backward(out,target);
             net:backward(input,dloss_g2)
          return loss_g2,allgrads
      end
    rmsprop(eval,params,{learningRate=0.9, alpha  = 0.95, epsilon=0.001, center=true})
'''
