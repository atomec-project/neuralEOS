import neuralEOS
import sys
import pickle as pkl
import time

params_serialized = sys.argv[1]
params = pkl.loads(bytes.fromhex(params_serialized))

savedir = sys.argv[2]

training_file = sys.argv[3]

params.cv_pkl_file = savedir + "/run_" + sys.argv[5] + ".pkl"

if sys.argv[4].lower() == "true":
    use_aa = True
else:
    use_aa = False
    

t0 = time.time()
trainer = neuralEOS.Training(params)
trainer.train_inner_loop(training_file, use_aa, save_dir=savedir)
t1 = time.time()

print("Time taken = ", t1 - t0)

