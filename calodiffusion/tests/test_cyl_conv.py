from models import *
import torch
import torch.nn as nn


#x = torch.arange(12.0).reshape(1,1,1,4,3)
x =  torch.FloatTensor( [[[[[ 1, 2, 3],
                       [ 1, 2, 3],
                       [ 1, 2, 3],
                       [ 1, 2, 3], ]]]])

print(x)

cyl_model = CylindricalConv(1,1, kernel_size =(1,3,3), stride =1, padding = (0,1,1), bias = False )
conv_model = nn.Conv3d(1,1, kernel_size =(1,3,3), stride =1, padding = (0,1,1), bias = False )

print(cyl_model)
print(conv_model)

with torch.no_grad():
    #set all weights to 1
    cyl_model.conv.weight = nn.Parameter(torch.ones_like(cyl_model.conv.weight))
    conv_model.weight = nn.Parameter(torch.ones_like(conv_model.weight))
    cyl_out = cyl_model(x)
    conv_out = conv_model(x)

print("CYLINDRICAL")
print(cyl_out)
print("REGULAR")
print(conv_out)



