#precomputed values for datasets

#dataset1 pions
dataset0_params ={
'logit_mean' : -12.4783,
'logit_std' : 2.21267,
'logit_min': -13.81551,
'logit_max' :  0.9448,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,


'sqrt_mean' : 0.0,
'sqrt_std' : 1.0,
'sqrt_min' : 0.,
'sqrt_max' : 1.0,

'qt' : 'qts/dset1_pions_quantile_transform.gz',
}


#dataset1 pions, no geom reshaping (fully connected)
dataset0_fcn_params ={
'logit_mean' : -11.7610,
'logit_std' : 2.84317,
'logit_min': -13.81551,
'logit_max' :  0.2554,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,


'sqrt_mean' : 0.0,
'sqrt_std' : 1.0,
'sqrt_min' : 0.,
'sqrt_max' : 1.0,


'qt' : None,
}




#dataset1 photons
dataset1_params ={
'logit_mean' : -12.1444,
'logit_std' : 2.45056,
'logit_min': -13.81551,
'logit_max' :  -1.6615,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,


'sqrt_mean' : 0.0,
'sqrt_std' : 1.0,
'sqrt_min' : 0.,
'sqrt_max' : 1.0,

'qt' : 'qts/dset1_photons_quantile_transform.gz',
}


#dataset1 photons, no geom reshaping (fully connected)
dataset1_fcn_params ={
'logit_mean' : -9.9807,
'logit_std' : 3.14168,
'logit_min': -13.81551,
'logit_max' :  0.2554,

'log_mean' : 0.0,
'log_std' : 1.0,
'log_min' : 0.0,
'log_max' : 2.0,


'sqrt_mean' : 0.0,
'sqrt_std' : 1.0,
'sqrt_min' : 0.,
'sqrt_max' : 1.0,

'qt' : None,
}

dataset2_params = {
'logit_mean' : -12.8564,
'logit_std' : 1.9123,
'logit_min': -13.8155,
'logit_max' :  0.1153,

'log_mean' : -17.5451,
'log_std' : 4.4086,
'log_min' : -20.0,
'log_max' :  -0.6372,


'sqrt_mean' : 0.0026,
'sqrt_std' : 0.0073,
'sqrt_min' : 0.,
'sqrt_max' : 1.0,

'qt' : 'qts/dset2_quantile_transform.gz',
}


dataset3_params = {
'logit_mean' : -13.4753,
'logit_std' : 1.1070,
'logit_min': -13.81551,
'logit_max' :  0.2909,

'log_mean' : -1.1245,
'log_std' : 3.3451,
'log_min' : -18.6905,
'log_max' : 0.0,


'sqrt_mean' : 0.0,
'sqrt_std' : 1.0,
'sqrt_min' : 0.,
'sqrt_max' : 1.0,
'qt' : 'qts/dset3_quantile_transform.gz',
}
dataset_params = {
        0: dataset0_params, 
        1: dataset1_params, 
        2:dataset2_params, 
        3:dataset3_params,
        10: dataset0_fcn_params,
        11: dataset1_fcn_params,
        }
