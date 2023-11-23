import neuralEOS
import sys
import time
import pickle as pkl

params_serialized = sys.argv[1]
params = pkl.loads(bytes.fromhex(params_serialized))

savedir = sys.argv[2]
training_file = sys.argv[3]
model_file = sys.argv[4]
score_index = int(sys.argv[5])

if sys.argv[6].lower() == "true":
    use_aa = True
else:
    use_aa = False

trainer = neuralEOS.Training(params)
trainer.train_outer_loop(
    training_file,
    savedir + "cv_summary.txt",
    save_dir=savedir,
    model_file=model_file,
    use_aa = use_aa,
    score_index = score_index,
)


