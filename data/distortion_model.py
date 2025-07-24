import math
import numpy as np
import torch

def distortionParameter(cls):
    parameters = []
    
    if (cls == 'barrel'):
        lower_bound = 0.1 # Originally 1 for whole img distortion
        upper_bound = 0.5 # Originally 5 for whole img distortion
        Lambda = (lower_bound + (upper_bound - lower_bound) * np.random.random_sample())* -5e-6/4 # Original: -5e-5/4 (-0.0000125)
        # Lambda= upper_bound * -5e-6/4
        x0 = 256
        y0 = 256
        parameters.append(Lambda)
        parameters.append(x0)
        parameters.append(y0)
        return parameters
    
    elif (cls == 'pincushion'):
        lower_bound = 1
        upper_bound = 3
        Lambda = (lower_bound + (upper_bound - lower_bound) * np.random.random_sample()) * 5e-6/4 # Original: -5e-5/4 (-0.0000125)
        # Lambda= upper_bound * 5e-6/4
        x0 = 256
        y0 = 256
        parameters.append(Lambda)
        parameters.append(x0)
        parameters.append(y0)
        return parameters
    
    elif (cls == 'rotation'):
        # theta = np.random.random_sample() * 30 - 15  # Original: theta = [-15,15]
        lower_bound = 1
        upper_bound = 15
        theta = (lower_bound + (upper_bound - lower_bound) * np.random.random_sample())
        # theta = upper_bound
        theta *= -1 if np.random.random_sample() > 0.5 else 1 # New theta range: [-15,-1] and [1,15]
        radian = math.pi*theta/180
        sina = math.sin(radian)
        cosa = math.cos(radian)
        parameters.append(sina)
        parameters.append(cosa)
        return parameters
    
    elif (cls == 'shear'):
        # shear = np.random.random_sample() * 0.8 - 0.4 # Original: [-0.4,0.4]
        lower_bound = 0.05
        upper_bound = 0.4 
        shear = (lower_bound + (upper_bound - lower_bound) * np.random.random_sample())
        # shear = upper_bound
        # shear *= -1 if np.random.random_sample() > 0.5 else 1 # New range: [-0.4,-0.05] and [0.05,0.4]
        parameters.append(shear)
        return parameters

    elif (cls == 'projective'):

        x1 = 0
        lower_bound = 0.1
        upper_bound = 0.2
        x4 = (lower_bound + (upper_bound - lower_bound) * np.random.random_sample())
        x4 = np.random.random_sample()* 0.1 + 0.1 # Original
        
        x2 = 1 - x1
        x3 = 1 - x4

        y1 = 0.005
        y4 = 1 - y1
        y2 = y1
        y3 = y4

        a31 = ((x1-x2+x3-x4)*(y4-y3) - (y1-y2+y3-y4)*(x4-x3))/((x2-x3)*(y4-y3)-(x4-x3)*(y2-y3))
        a32 = ((y1-y2+y3-y4)*(x2-x3) - (x1-x2+x3-x4)*(y2-y3))/((x2-x3)*(y4-y3)-(x4-x3)*(y2-y3))

        a11 = x2 - x1 + a31*x2
        a12 = x4 - x1 + a32*x4
        a13 = x1

        a21 = y2 - y1 + a31*y2
        a22 = y4 - y1 + a32*y4
        a23 = y1
       
        parameters.append(a11)
        parameters.append(a12)
        parameters.append(a13)
        parameters.append(a21)
        parameters.append(a22)
        parameters.append(a23)
        parameters.append(a31)
        parameters.append(a32)
        return parameters
    
    elif (cls == 'wave'):
        # mag = np.random.random_sample() * 32 # Original
        lower_bound = 3.2
        upper_bound = 8
        mag = (lower_bound + (upper_bound - lower_bound) * np.random.random_sample())
        parameters.append(mag)
        return parameters
    
    elif (cls=='none'):
        return parameters

def distortion_model(cls, H, W, parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_coords, y_coords = torch.meshgrid(
    torch.arange(W, device=device),
    torch.arange(H, device=device),
    indexing='xy'
    )

    if cls == 'none':
        pass
    elif cls == 'barrel' or cls == 'pincushion':
        Lambda, x0, y0 = parameters
        coeff = 1 + Lambda * ((x_coords - x0)**2 + (y_coords - y0)**2)
        x_coords = (x_coords.float() - x0) / coeff + x0
        y_coords = (y_coords.float() - y0) / coeff + y0
    elif cls == 'rotation':
        sina, cosa  = parameters
        x_coords =  cosa*x_coords + sina*y_coords + (1 - sina - cosa)*W/2
        y_coords = -sina*x_coords + cosa*y_coords + (1 + sina - cosa)*H/2
    elif cls == 'shear':
        shear = parameters[0]
        x_coords =  x_coords + shear*y_coords - shear*W/2
        y_coords =  y_coords + shear*x_coords - shear*H/2
    elif cls == 'projective':
        a11, a12, a13, a21, a22, a23, a31, a32 = parameters
        
        im = x_coords/(W - 1.0)
        jm = y_coords/(H - 1.0)
        x_coords = (W - 1.0) *(a11*im + a12*jm +a13)/(a31*im + a32*jm + 1)
        y_coords = (H - 1.0)*(a21*im + a22*jm +a23)/(a31*im + a32*jm + 1)
    elif cls == 'wave':
        mag = parameters[0]
        y_coords, x_coords = y_coords.float(), x_coords.float()
        # wave_fn = torch.sin if np.random.random_sample() > 0.5 else torch.cos
        # sign = -1  if np.random.random_sample() > 0.5 else 1
        # if np.random.random_sample() > 0.5:
        #     y_coords = y_coords.float() + mag * wave_fn(sign * 4 * math.pi * x_coords / H)
        # else:
        #     x_coords = x_coords.float() + mag * wave_fn(sign * 4 * math.pi * y_coords / W)
        x_coords += mag*torch.sin(np.pi*4*y_coords/W)
    return x_coords, y_coords
