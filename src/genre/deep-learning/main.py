import os
import sys
import argparse
import pickle
from run import run_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Name of the model to be run', required=True)
parser.add_argument('--scenario', help='Training Data Case (A/B/C/D)', required=True)
parser.add_argument('--elmo', help='Use ELMo embeddings', action="store_true") # default: use GloVe embeddings
parser.add_argument('--save_model', help='Save model weights & vocabulary', action="store_true")
parser.add_argument('--eda', help='Use EDA for Data Augmentation', action="store_true") # Use EDA; default: no Data Augmentation
parser.add_argument('--cda', help='Use CDA for Data Augmentation', action="store_true") # Use CDA; default: no Data Augmentation
args = parser.parse_args()


results_path = '/path/Augmentation-for-Literary-Data/results/DL-results/'+args.model+'_ELMo'+str(args.elmo)+'_EDA'+str(args.eda)+'_Case_'+args.scenario+'.tsv'
# if os.path.exists(results_path):
#     sys.exit("Results file already exists: " + results_path)
results_file = open(results_path, "w")
results_file.write("Model\tF1-score\tAUROC\tWeighted F1\tPrecision\tRecall\tAccuracy\tAUPRC\n")

preds_path = '/path/Augmentation-for-Literary-Data/results/predictions/'+args.model+'_ELMo'+str(args.elmo)+'_EDA'+str(args.eda)+'_preds_for_Case_'+args.scenario+'.tsv'

print("Model = {} | Scenario = {} | EDA = {} | CDA = {} | ELMo = {} | SaveModel = {} | OutputResults = {} | SavePredictions = {}\n".format(args.model, args.scenario, args.eda, args.cda, args.elmo, args.save_model, results_path, preds_path))


if args.eda: # run with EDA
    f1, auroc, w_f1, precision, recall, accuracy, auprc, preds = run_model(name=args.model,
                                                                           case=args.scenario,
                                                                           augmentation='EDA',
                                                                           use_elmo=args.elmo,
                                                                           save_model=args.save_model)

elif args.cda: # run with CDA
    pass

else: # run wihtout any Data Augmentation
    f1, auroc, w_f1, precision, recall, accuracy, auprc, preds = run_model(name=args.model,
                                                                           case=args.scenario,
                                                                           augmentation=None,
                                                                           use_elmo=args.elmo,
                                                                           save_model=args.save_model)

results_file.write(args.model+'ELMo'+str(args.elmo)+'\t'+str(f1)+'\t'+str(auroc)+'\t'+str(w_f1)+'\t'+str(precision)+'\t'+str(recall)+'\t'+str(accuracy)+'\t'+str(auprc)+'\n')

# Save predictions:
print("Write predictions to:", preds_path)
with open(preds_path, 'w') as f:
    f.write('fname\tprobability_fiction\tlabel\n')
    for ID, prob in preds.items():
        if prob >= 0.5: # ordering is ['fic' 'non']
            f.write(ID+'\t'+str(prob)+'\tfic\n')
        else:
            f.write(ID+'\t'+str(prob)+'\tnon\n')
