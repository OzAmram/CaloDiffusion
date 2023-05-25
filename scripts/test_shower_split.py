from utils import *
sys.path.append("..")
from CaloChallenge.code.XMLHandler import *


print("\n SIMPLE TEST : \n")
all_r_edges = torch.FloatTensor([0,1,2,4])
lay_r_edges = [[0,4],
                [0,1,2,4],
                [0,1,4],
                [0,1,2]]

g1 = GeomConverter(all_r_edges = all_r_edges, lay_r_edges = lay_r_edges)

orig_data1 = [[[1.]],
              [[2., 2., 2.]],
              [[3., 3.]],
              [[4., 4.]]]

orig_data2 = [[[1.]],
              [[1., 2., 3.]], 
              [[1., 2.]],
              [[5., 6.]]]



odata1 = g1.convert(orig_data1)
odata2 = g1.convert(orig_data2)
og1 = g1.unconvert(odata1)
og2 = g1.unconvert(odata2)

print("o1")
print(odata1)
print("Deconvert")
print(og1)

print("o2")
print(odata2)
print("Deconvert")
print(og2)



print("\n DATASET PHOTONS TEST : \n")

dataset = "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_1_photons_1.hdf5"

binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"

bins = XMLHandler("photon", binning_file)


g = GeomConverter(bins=bins)
print("all_R", g.all_r_edges)

nevts = 5000
with h5.File(dataset,"r") as h5f:
    raw_e = h5f['incident_energies'][0:int(nevts)].astype(np.float32)/1000.0
    raw_shower = h5f['showers'][0:int(nevts)].astype(np.float32)/1000.0


print(raw_shower.shape)

shower_reshape = g.reshape(raw_shower)



print("orig:\n")
for d in shower_reshape: print(d.shape)

conv = g.convert(shower_reshape)
print("GLAM SHAPE", conv.shape)
print('layer avgs', torch.mean(conv, axis = (0,2,3)))
print('phi avgs', torch.mean(conv, axis = (0,1,3)))
for i in range(5):
    print("Lay %i phi avg " % i, torch.mean(conv[:,i], axis = (0,2)))

print("CONV")

print("UNDO")
unconv = g.unconvert(conv)
og = g.unreshape(unconv).detach().numpy()

n1 = NNConverter(g)
n_enc = n1.enc(torch.FloatTensor(raw_shower))
print('NN layer avgs', torch.mean(n_enc, axis = (0,2,3)))
n_dec = n1.dec(n_enc).detach().numpy()

assert( np.allclose(raw_shower, og))

print("CLOSE : ", np.allclose(raw_shower, og))
print("NN CLOSE : ", np.allclose(raw_shower, n_dec))




print("\n DATASET PIONS TEST : \n")

dataset2 = "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_1_pions_1.hdf5"

binning_file2 = "../CaloChallenge/code/binning_dataset_1_pions.xml"


bins2 = XMLHandler("pion", binning_file2)

g2 = GeomConverter(bins = bins2)

with h5.File(dataset2,"r") as h5f:
    raw_e2 = h5f['incident_energies'][0:int(nevts)].astype(np.float32)/1000.0
    raw_shower2 = h5f['showers'][0:int(nevts)].astype(np.float32)/1000.0


print('raw', raw_shower2.shape)


shower_reshape2 = g2.reshape(raw_shower2)

for d in shower_reshape2: print(d.shape)



conv2 = g2.convert(shower_reshape2)
print('GLAM shape: ', conv2.shape)
unconv2 = g2.unconvert(conv2)
og2 = g2.unreshape(unconv2).detach().numpy()
print("all_R:")
for r in g2.all_r_edges.numpy():
    print("%.2f, " % r, end ="")
print("")

assert( np.allclose(raw_shower2, og2))
print("CLOSE : ", np.allclose(raw_shower2, og2))



