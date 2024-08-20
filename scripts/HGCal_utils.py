import sys
import io
import pickle
from einops import rearrange
from torch.masked import masked_tensor
import torch.sparse

sys.path.append("..")
sys.path.append("../HGCalShowers/")
from utils import *
from HGCalShowers.HGCalGeo import *
from consts import *


def logit(x, alpha = 1e-6):
    o = alpha + (1 - 2*alpha)*x
    o = np.ma.log(o/(1-o)).filled(0)    
    return o

def reverse_logit(x, alpha = 1e-6):
    exp = np.exp(x)    
    o = exp/(1+exp)
    o = (o-alpha)/(1 - 2*alpha)
    return o


def preprocess_hgcal_shower(shower, e, shape, showerMap = 'log-norm', dataset_num = 2, orig_shape = False, ecut = 0, max_deposit = 2):

    print('dset', dataset_num)

    c = dataset_params[dataset_num]

    if('quantile' in showerMap and ecut > 0):
        np.random.seed(123)
        noise = (ecut/3) * np.random.rand(*shower.shape)
        shower +=  noise


    alpha = 1e-6
    per_layer_norm = False

    layerE = None
    prefix = ""
    print(shower.shape)
    if('layer' in showerMap):
        eshape = (-1, *(1,)*(len(shower.shape) -1))
        shower = np.ma.divide(shower, (max_deposit*e.reshape(eshape)))
        #regress total deposited energy and fraction in each layer

        layers = np.sum(shower,(2),keepdims=True)
        totalE = np.sum(shower, (1,2), keepdims = True)
        if(per_layer_norm): shower = np.ma.divide(shower,layers)

        #only logit transform for layers
        layer_alpha = 1e-6
        layers = np.ma.divide(layers,totalE)
        layers = logit(layers)


        layers = (layers - c['layers_mean']) / c['layers_std']
        totalE = (totalE - c['totalE_mean']) / c['totalE_std']
        #append totalE to layerE array
        totalE = np.reshape(totalE, (totalE.shape[0], 1))
        layers = np.squeeze(layers)
        layerE = np.concatenate((totalE,layers), axis = 1)

        if(per_layer_norm): prefix = "layerN_"
    else:
        eshape = (-1, *(1,)*(len(shower.shape) -1))
        shower = shower/(max_deposit*e.reshape(eshape))

    if('logit' in showerMap):
        shower = logit(shower)

        if('norm' in showerMap): shower = (shower - c[prefix +'logit_mean']) / c[prefix+'logit_std']
        elif('scaled' in showerMap): shower = 2.0 * (shower - c['logit_min']) / (c['logit_max'] - c['logit_min']) - 1.0

    elif('log' in showerMap):
        eps = 1e-8
        shower = np.ma.log(shower).filled(c['log_min'])
        if('norm' in showerMap): shower = (shower - c[prefix+'log_mean']) / c[prefix+'log_std']
        elif('scaled' in showerMap):  shower = 2.0 * (shower - c[prefix+'log_min']) / (c[prefix+'log_max'] - c[prefix+'log_min']) - 1.0


    return shower,layerE



def DataLoaderHGCal(file_name,shape,gen_max,gen_min, nevts=-1,  max_deposit = 2, ecut = 0, logE=True, showerMap = 'log-norm', nholdout = 0, from_end = False, dataset_num = 2, orig_shape = False,
        evt_start = 0, max_cells = None):

    with h5.File(file_name,"r") as h5f:
        #holdout events for testing
        if(nevts == -1 and nholdout > 0): nevts = -(nholdout)
        end = evt_start + int(nevts)
        if(from_end):
            evt_start = -int(nevts)
            end = None
        if(end == -1): end = None 
        print("Event start, stop: ", evt_start, end)
        gen_info = h5f['gen_info'][evt_start:end].astype(np.float32)
        shower = h5f['showers'][evt_start:end][:,:,:max_cells].astype(np.float32) *100 # shower is units of charge, multiply by 100 to get rough match to energy scale

    e = gen_info[:,0]
    gen_min = np.array(gen_min)
    gen_max = np.array(gen_max)

    print("Data loaded", shower.shape)


        
    shower_preprocessed, layerE_preprocessed = preprocess_hgcal_shower(shower, e, shape, showerMap, dataset_num = dataset_num, orig_shape = orig_shape, ecut = ecut, max_deposit=max_deposit)
    print("preprocessed")

    gen_preprocessed = (gen_info-gen_min)/(gen_max-gen_min)
    print('gen' , gen_preprocessed.shape)

    return shower_preprocessed, gen_preprocessed , layerE_preprocessed


def ReverseNormHGCal(voxels,e,shape,gen_max,gen_min, max_deposit=2,logE=True, layerE = None, showerMap ='log', dataset_num = 2, orig_shape = False, ecut = 0.):
    '''Revert the transformations applied to the training set'''

    print('vox', voxels.shape)


    c = dataset_params[dataset_num]

    alpha = 1e-6

    gen_min = np.array(gen_min)
    gen_max = np.array(gen_max)

    gen_out = gen_min + (gen_max-gen_min)*e
    energy = gen_out[:,0]

    prefix = ""
    #if('layer' in showerMap): prefix = "layerN_"

    if('quantile' in showerMap and c['qt'] is not None):
        print("Loading quantile transform from %s" % c['qt'])
        qt = joblib.load(c['qt'])
        shape = voxels.shape
        voxels = qt.inverse_transform(voxels.reshape(-1,1)).reshape(shape)

        
    if('logit' in showerMap):
        if('norm' in showerMap): voxels = (voxels * c[prefix+'logit_std']) + c[prefix+'logit_mean']
        elif('scaled' in showerMap): voxels = (voxels + 1.0) * 0.5 * (c[prefix+'logit_max'] - c[prefix+'logit_min']) + c[prefix+'logit_min']

        #avoid overflows
        #voxels = np.minimum(voxels, np.log(max_deposit/(1-max_deposit)))

        data = reverse_logit(voxels)

    elif('log' in showerMap):
        if('norm' in showerMap): voxels = (voxels * c[prefix+'log_std']) + c[prefix+'log_mean']
        elif('scaled' in showerMap): voxels = (voxels + 1.0) * 0.5 * (c[prefix+'log_max'] - c[prefix+'log_min']) + c[prefix+'log_min']

        voxels = np.minimum(voxels, np.log(max_deposit))


        data = np.exp(voxels)

    #Per layer energy normalization
    if('layer' in showerMap):
        assert(layerE is not None)
        totalE, layers = layerE[:,:1], layerE[:,1:]
        totalE = (totalE * c['totalE_std']) + c['totalE_mean']
        layers = (layers * c['layers_std']) + c['layers_mean']

        layers = reverse_logit(layers)

        #scale layer energies to total deposited energy
        layers /= np.sum(layers, axis = 1, keepdims = True)
        layers *= totalE


        data = np.squeeze(data)

        #remove voxels with negative energies so they don't mess up sums
        eps = 1e-6
        data[data < 0] = 0 
        #layers[layers < 0] = eps
        print('data', data.shape)


        #Renormalize layer energies
        prev_layers = np.sum(data,(2),keepdims=True)
        layers = layers.reshape((-1,data.shape[1],1))
        rescale_facs =  layers / (prev_layers + 1e-10)
        #If layer is essential zero from base network or layer network, don't rescale
        rescale_facs[layers < eps] = 1.0
        rescale_facs[prev_layers < eps] = 1.0
        data *= rescale_facs
                    
                
        data = data*max_deposit*energy.reshape(-1,1,1)

    if(ecut > 0): data[data < ecut ] = 0 #min from samples
    
    return data,gen_out



class Embeder(nn.Module):
    def __init__(self, dim1, dim2, mat, mask, trainable = False):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        if(trainable): 
            self.mat = torch.nn.Parameter(mat, requires_grad = True)
            print("Embed trainable")
        else: self.mat = mat
        self.mask = mask

        #Masked tensor doesn't suppoert einsum so can't use for now ... (unless hacky way to get properly indexed matmul to work?)
        #masked_mat = masked_tensor(mat, mask, requires_grad = trainable)
        #masked_mat = masked_mat.reshape(shape[1], shape[0], shape[2])
        #mat = masked_mat.to_sparse_coo()


    #reshape?
    def forward(self, x):
        masked_mat = self.mat * self.mask
        out = torch.einsum("l e n, b c l n -> b c l e", masked_mat, x)
        out = rearrange(out, " b c l (a r) -> b c l a r",a = self.dim1, r = self.dim2)
        return out

    def set(self, mat, mask):
        self.mat.values = mat
        self.mask = mask

class Decoder(nn.Module):
    def __init__(self, dim1, dim2, mat, mask, trainable = False):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

        if(trainable): self.mat = torch.nn.Parameter(mat, requires_grad = True)
        else: self.mat = mat
        self.mask = mask


    def forward(self, x):
        masked_mat = self.mat * self.mask
        out = rearrange(x, " b c l a r -> b c l (a r)", a = self.dim1, r = self.dim2)
        out = torch.einsum("l n e, b c l e -> b c l n", masked_mat, out)
        return out

    def set(self, mat, mask):
        self.mat.values = mat
        self.mask = mask

#initialize GLaM map
def init_map(num_alpha_bins, num_r_bins, geom, ilay, trainable = False):
    dim_in = geom.max_ncell
    ncells = int(round(geom.ncells[ilay]))
    dim_out = num_alpha_bins * num_r_bins

    #Weight matrix and sparsity mask
    weight_mat = torch.zeros((num_alpha_bins, num_r_bins, dim_in))
    mask = torch.zeros((num_alpha_bins, num_r_bins, dim_in))

    step_size = 2.*np.pi/num_alpha_bins
    ang_bins = torch.arange(0, 2.*np.pi + step_size, step_size)
    ang_bins += np.pi / num_alpha_bins # shift by half a bin width


    eps = 1e-2
    cell_alphas = torch.tensor(geom.theta_map[ilay][:dim_in])
    cell_ang_bins = torch.bucketize(cell_alphas + eps, ang_bins, right = True)
    #last bin = first bin b/c phi periodic
    cell_ang_bins[cell_ang_bins == num_alpha_bins] = 0 
    diffs = torch.abs(cell_alphas - ang_bins[cell_ang_bins-1])

    close_boundaries = (diffs < eps) | (torch.abs(diffs - 2. * np.pi) < eps)

    #special init for first bin
    #split among all ang bins in the central radial bin
    weight_mat[:, 0, 0] = (1./ num_alpha_bins)
    mask[:, 0, 0] = 1.0

    #dumb slow for loop to initialize for now
    for i in range(1, ncells): 
    #matrix is size of largest layer, but only process up to size of this layer
        a_bin = cell_ang_bins[i]
        r_bin = int(round(geom.ring_map[ilay, i]))
        if(close_boundaries[i]):
            weight_mat[a_bin, r_bin, i] = 0.5
            weight_mat[a_bin-1, r_bin, i] = 0.5

            #Local neighborhood of values trainable
            mask[a_bin, r_bin, i] = 1.0
            if(r_bin > 0): mask[a_bin, r_bin-1, i] = 1.0
            if(r_bin < num_r_bins-1): mask[a_bin, r_bin+1, i] = 1.0
            mask[a_bin-1, r_bin, i] = 1.0
            if(r_bin > 0): mask[a_bin-1, r_bin-1, i] = 1.0
            if(r_bin < num_r_bins-1): mask[a_bin-1, r_bin-1, i] = 1.0
        else:
            weight_mat[a_bin, r_bin, i] = 1.0

            #Local neighborhood of values trainable
            mask[a_bin, r_bin, i] = 1.0
            mask[ (a_bin-1) % num_alpha_bins, r_bin, i] = 1.0
            mask[ (a_bin+1) % num_alpha_bins, r_bin, i] = 1.0
            if(r_bin > 0): mask[a_bin, r_bin-1, i] = 1.0
            if(r_bin < num_r_bins-1): mask[a_bin, r_bin+1, i] = 1.0

    weight_mat = weight_mat.reshape((num_alpha_bins * num_r_bins, dim_in))
    mask = mask.reshape((num_alpha_bins * num_r_bins, dim_in))
    return weight_mat, mask


#Work around for a dumb pickle behavior...
# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "tools":
            renamed_module = "whyteboard.tools"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def pickle_load(file_obj):
    return RenameUnpickler(file_obj).load()

class HGCalConverter(nn.Module):
    "Convert irregular hgcal geometry to regular one, initialized with regular geometric conversion, but uses trainable linear map"
    def __init__(self, bins = None, geom_file = None, hidden_size = 32, device = None, trainable = False):
        super().__init__()
        print("Loading geometry from %s " % geom_file)

        self.geom_file = open(geom_file, 'rb')
        self.device = device

        self.trainable = trainable

        self.geom = pickle_load(self.geom_file)
        #angle from 0 to 2pi, arctan2 has (y,x) convention for some reason
        self.geom.theta_map = np.arctan2(self.geom.xmap, self.geom.ymap) % (2. *np.pi)
        self.geom.max_ncell = int(round(np.amax(self.geom.ncells)))
        print("Ncell max %i" % self.geom.max_ncell)


        self.bins = bins
        self.num_r_bins = bins[-1]
        self.num_alpha_bins = bins[-2]
        self.num_layers = bins[-3]


        self.enc_mat = torch.zeros((self.num_layers, self.num_alpha_bins * self.num_r_bins, self.geom.max_ncell), device = device)
        self.dec_mat = torch.zeros((self.num_layers, self.geom.max_ncell, self.num_alpha_bins * self.num_r_bins), device = device)

        self.enc_mask = torch.zeros((self.num_layers, self.num_alpha_bins * self.num_r_bins, self.geom.max_ncell), device = device, dtype=torch.bool)
        self.dec_mask = torch.zeros((self.num_layers, self.geom.max_ncell, self.num_alpha_bins * self.num_r_bins), device = device, dtype=torch.bool)

        #Need to init with dummy values
        self.embeder = Embeder(self.num_alpha_bins, self.num_r_bins, self.enc_mat, self.enc_mask, trainable = self.trainable) 
        self.decoder = Decoder(self.num_alpha_bins, self.num_r_bins, self.dec_mat, self.dec_mask, trainable = self.trainable) 

        self.nets = nn.ModuleList([self.embeder, self.decoder])


    #proper initialization of embedding
    def init(self, noise_scale = 0.):

        for i in range(self.geom.nlayers):
            lay_size = self.geom.ncells[i]

            conv_map, mask = init_map(self.num_alpha_bins, self.num_r_bins, self.geom, i)


            #How to define sparse mask of inverse ? 
            inv_init = torch.linalg.pinv(conv_map)
            inv_mask_init = torch.linalg.pinv(mask)
            #inv_init = torch.zeros((conv_map.shape[1], conv_map.shape[0]))

            if(noise_scale > 0.):
                noise = torch.randn_like(conv_map)
                conv_map += noise * noise_scale
                noise2 = torch.randn_like(inv_init)
                inv_init += eps*noise2

            self.enc_mat[i] = conv_map
            self.enc_mask[i] = mask > 0.
            self.dec_mat[i] = inv_init
            self.dec_mask[i] = inv_mask_init > 0.

            #print(i)
            #print('sum', torch.sum(conv_map))
            
        self.embeder.set(self.enc_mat, self.enc_mask)
        self.decoder.set(self.dec_mat, self.dec_mask)



    def enc(self, x):
        n_shower = x.shape[0]
        out = self.embeder(x)
        return out

    def dec(self, x):
        out = self.decoder(x)
        return out


    def forward(x):
        x = self.embeder(x)
        x = self.decoder(x)
        return x

