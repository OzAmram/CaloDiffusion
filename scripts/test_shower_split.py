from utils import *
sys.path.append("..")
from CaloChallenge.code.XMLHandler import *


def compute_weight_mats(all_r_edges, lay_r_edges):
    all_r_areas = (all_r_edges[1:]**2 - all_r_edges[:-1]**2)
    dim_r_out = len(all_r_edges) - 1
    weight_mats = []
    for ilay in range(len(lay_r_edges)):
        dim_in = len(lay_r_edges[ilay]) - 1
        weight_mat = torch.zeros((dim_r_out, dim_in))
        for ir in range(dim_in):
            o_idx_start = torch.nonzero(all_r_edges == lay_r_edges[ilay][ir])[0][0]
            o_idx_stop = torch.nonzero(all_r_edges == lay_r_edges[ilay][ir + 1])[0][0]

            split_idxs = list(range(o_idx_start, o_idx_stop))
            orig_area = (lay_r_edges[ilay][ir+1]**2 - lay_r_edges[ilay][ir]**2)

            weight_mat[split_idxs, ir] = all_r_areas[split_idxs]/orig_area

        weight_mats.append(weight_mat)
    return weight_mats

def convert(d, weight_mats, num_alphas = None, alpha_out = 1):
    out = []
    for i in range(len(d)):
        o = torch.einsum( '...ij,...j->...i', weight_mats[i], torch.FloatTensor(d[i]))
        if(num_alphas is not None):
            if(num_alphas[idx]  == 1):
                #distribute evenly in phi
                o = torch.repeat_interleave(o, alpha_out, dim = -2)/alpha_out
            elif(num_alphas[idx]  != alpha_out):
                print("Num alpha bins for layer %i is %i. Don't know how to handle" % (idx, num_alpha))
                exit(1)
        out.append(o)
    return out


def unconvert(d, weight_mats, num_alphas = None, alpha_out = 1):
    out = []
    for i in range(len(d)):
        o = torch.einsum( '...ij,...j->...i', torch.linalg.pinv(weight_mats[i]), torch.FloatTensor(d[i]))
        if(num_alphas is not None):
            if(num_alphas[idx]  == 1):
                #Only works for converting 1 alpha bin into multiple, ok for dataset1 but maybe should generalize
                o = torch.sum(o, dim = -2, keepdim = True)
            elif(num_alphas[idx]  != alpha_out):
                print("Num alpha bins for layer %i is %i. Don't know how to handle" % (idx, num_alpha))
                exit(1)
        out.append(o)
    return out





print("\n SIMPLE TEST : \n")
all_r_edges = torch.FloatTensor([0,1,2,4])
lay_r_edges = [[0,4],
                [0,1,2,4],
                [0,1,4],
                [0,1,2]]

weight_mat_test = compute_weight_mats(all_r_edges, lay_r_edges)

orig_data1 = [[1.],
              [2., 2., 2.],
              [3., 3.],
              [4., 4.]]

orig_data2 = [[1.],
              [1., 2., 3.], 
              [1., 2.],
              [5., 6.]]



odata1 = convert(orig_data1, weight_mat_test)
odata2 = convert(orig_data2, weight_mat_test)

print("o1")
print(odata1)
print("Deconvert")
og1 = unconvert(odata1, weight_mat_test)
print(og1)

print("o2")
print(odata2)
og2 = unconvert(odata2, weight_mat_test)
print(og2)





print("\n DATASET TEST : \n")

dataset = "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_1_photons_1.hdf5"

binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"

bins = XMLHandler("photon", binning_file)

layer_boundaries = np.unique(bins.GetBinEdges())
rel_layers = bins.GetRelevantLayers()
num_alpha = [len(bins.alphaListPerLayer[idx][0]) for idx, redge in enumerate(bins.r_edges) if len(redge) > 1]
alpha_out = np.amax(num_alpha)


all_r_edges = []

r_edges = [bins.r_edges[l] for l in rel_layers]
for ilay in range(len(r_edges)):
    for r_edge in r_edges[ilay]:
        all_r_edges.append(r_edge)
all_r_edges = torch.unique(torch.FloatTensor(all_r_edges))


weight_mat = compute_weight_mats(all_r_edges, r_edges)
for m in weight_mat: print(m.shape)

nevts = 10
with h5.File(dataset,"r") as h5f:
    raw_e = h5f['incident_energies'][0:int(nevts)].astype(np.float32)/1000.0
    raw_shower = h5f['showers'][0:int(nevts)].astype(np.float32)/1000.0


print(raw_shower.shape)
shower_reshape = []


for idx in range(len(layer_boundaries)-1):
    data_reshaped = raw_shower[:,layer_boundaries[idx]:layer_boundaries[idx+1]].reshape(raw_shower.shape[0], int(num_alpha[idx]), -1)
    shower_reshape.append(data_reshaped)



print("orig:\n")
print(shower_reshape[0])
for d in shower_reshape: print(d.shape)


conv = convert(shower_reshape, weight_mat, num_alpha, alpha_out)

print("CONV")
for c in conv: print(c.shape)
print(conv[0])

print("UNDO")
conv = unconvert(conv, weight_mat, num_alpha, alpha_out)
print(conv[0])







