# classtorch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

import numpy as np
from scipy import signal

class CrossCorrelation:
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding_mode='valid'):
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding_mode = padding_mode
        self.kernel = np.random.rand(self.in_channels,*self.kernel_size)
        self.bias = None

    def forward(self, x):
        self.x = x
        if len(x.shape) > 3:
            batch_size = x.shape[0]
        else:
            batch_size = 1
        H_out = int(np.floor((x.shape[-2] - self.kernel_size[0]+1)))
        W_out = int(np.floor((x.shape[-1] - self.kernel_size[1]+1)))
        self.H_out = H_out
        self.W_out = W_out
        self.bias = np.random.rand(H_out, W_out)
        self.output = np.zeros(shape=(batch_size,self.out_channels,H_out, W_out))
        self.output[:,:,:,:] = self.bias

        for batch in range(batch_size):
            for i in range((self.out_channels)):
                for j in range((self.in_channels)):
                    self.output[batch][i] += signal.correlate2d(x[batch][j], self.kernel[j],mode=self.padding_mode)
        
        return self.output
    
    def backward(self, output_gradient, lr):
        kernel_gradient = np.zeros(shape=(self.kernel_size.shape))
        input_gradient =  np.zeros(shape=(self.H_out, self.W_out))

        for i in range(self.out_channels):
            for j in range(self.in_channels):
                kernel_gradient[i][j] = signal.correlate2d(self.x[j], output_gradient[i], padding_mode = 'valid')
                input_gradient[j]  += signal.convolve2d(output_gradient[i], self.kernel[i][j], padding_mode = 'full')

        self.kernel -= lr*kernel_gradient
        self.bias -= lr*output_gradient
        return input_gradient
    

        