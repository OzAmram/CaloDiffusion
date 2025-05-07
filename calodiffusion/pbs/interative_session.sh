# Submit a job to the debug queue for an hour 
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug -A CaloDiffusion -N interactive_debug

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128