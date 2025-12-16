import sys, os, math

def print_and_do(s):
    print(s)
    return os.system(s)

label = "Pions_nov14"
files_per_job = 1
#file_list = "/global/u1/o/ozamram/CaloDiffusion/CMSHGCaloChallenge/datasets/photon_files_test_local.txt"
#odir = "/global/cfs/cdirs/m2612/calodiffusion/HGCal_generated_samples/HGCal_photon_nov14/"
file_list = "/global/u1/o/ozamram/CaloDiffusion/CMSHGCaloChallenge/datasets/pion_files_test_local.txt"
odir = "/global/cfs/cdirs/m2612/calodiffusion/HGCal_generated_samples/HGCal_pion_nov14/"


job_dir = "inference_jobs/" + label + "/"

if(not os.path.exists(job_dir)):
    os.system("mkdir " + job_dir)


fin = open(file_list)
f_list = fin.readlines()

num_lines = len(f_list)
nJobs = int(math.ceil(num_lines/files_per_job))

print("%i jobs" % nJobs)

for i in range(nJobs):

    start = i*files_per_job
    end = (i+1)*files_per_job
    files_batch = f_list[start:end]

    #file list of job
    job_file_name = os.path.abspath(job_dir + "files_job%i.txt"%i)
    out_file = open(job_file_name, "w")
    for line in files_batch:
        out_file.write(line)
    out_file.close()

    #prepare slurm script
    script_name = job_dir + "inf_job%i.sh"%i
    os.system("cp inference_template.sh %s" % (script_name))

    print_and_do("sed -i 's:JOBNUM:%i:' %s" % (i, script_name))
    print_and_do("sed -i 's:OUTDIR:%s:' %s" % (odir, script_name))
    print_and_do("sed -i 's:SAMPLE_FILE:%s:' %s" % (job_file_name, script_name))

    print_and_do("cd %s; sbatch inf_job%i.sh"% (job_dir, i))
    print_and_do("cd -")

