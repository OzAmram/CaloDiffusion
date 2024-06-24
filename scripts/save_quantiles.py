from utils import *

dataset = "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_3_1.hdf5"
fout = 'dset3_quantile_transform.gz'
dataset_num = 3
shape_pad = [-1,1,45,50,18,1]
max_dep = 2.
#dataset = "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_2_1.hdf5"
#fout = 'dset2_quantile_transform.gz'
#dataset_num = 2
#shape_pad = [-1,1,45,16,9]
#max_dep = 2.
#ecut = 0.0000151
ecut = 0.0000151
emax = 100.
emin = 1.

#dataset = "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_1_photons_1.hdf5"
#fout = 'dset1_photons_quantile_transform.gz'
#dataset_num = 1
#shape_pad = [-1,1,5,10,30]
#dataset = "/wclustre/cms_mlsim/denoise/CaloChallenge/dataset_1_pions_1.hdf5"
#fout = 'dset1_pions_quantile_transform.gz'
#dataset_num = 0
#shape_pad = [-1,1,7,10,23]
#max_dep = 3.1
#emax = 4194.304
#emin = 0.256
#ecut = 0.0000001



nevts =-1
logE = True
showerMap = 'logit-norm-quantile'




with h5.File(dataset,"r") as h5f:
    raw_e = h5f['incident_energies'][0:int(nevts)].astype(np.float32)/1000.0
    raw_shower = h5f['showers'][0:int(nevts)].astype(np.float32)/1000.0

data_,e_ = DataLoader( dataset, shape_pad, emax = emax,emin = emin, nevts = nevts,
    max_deposit=max_dep, #noise can generate more deposited energy than generated
    logE=logE,
    showerMap = showerMap,
    dataset_num = dataset_num,
    ecut = ecut,
)

data_shaped = data_.reshape(-1,1)
print(data_shaped.shape)
n_samples =  data_shaped.shape[0]
qt = QuantileTransformer(n_quantiles = 100000, subsample = n_samples, random_state = 0, output_distribution = 'normal')
qt.fit(data_shaped)
data_qt = qt.transform(data_shaped)

#make a plot
make_histogram([data_shaped.reshape(-1), data_qt.reshape(-1)], ['logit-normed', 'logit-quantile-normed'], colors = ['black', 'blue'], xaxis_label = 'Normalized Voxel Energy', num_bins = 100,
        fname = fout.replace(".gz",".png"), logy= True)
#save transform
joblib.dump(qt, fout)

data_ = qt.inverse_transform(data_qt).reshape(shape_pad)



data_rev, e_rev = ReverseNorm( data_, e_, shape_pad, emax = emax,emin = emin,
    max_deposit=max_dep, #noise can generate more deposited energy than generated
    logE=logE,
    showerMap = showerMap,
    dataset_num = dataset_num,
    ecut = ecut,
)
data_rev[data_rev < ecut] = 0.


data = np.reshape(data_,shape_pad)

print("ShowerMap %s" % showerMap)
print("RAW: \n")
#print(raw_shower[0,0,10])
print("PREPROCESSED: \n")
#print(data_[0,0,10])
mean = np.mean(data)
std = np.std(data)
maxE = np.amax(data)
minE = np.amin(data)

maxEn = np.amax(e_)
minEn = np.amin(e_)
print("MEAN: %.4f STD : %.5f"  % (mean, std))
print("MAX: %.4f MIN : %.5f"  % (maxE, minE))
print("maxE %.2f minE %.2f" % (maxEn, minEn))
print("REVERSED: \n")
#print(data_rev[0,0,10])
print("AVG DIFF: ", torch.mean(torch.tensor(data_rev[0]) - raw_shower[0]))


