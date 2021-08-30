# CASN
Co-attention span network with multi-task for BioNER  

### Requirements
* Python 3.6
  * PyTorch 1.9.0
  * transformers 2.1.1  
  
  
  ## Training Instructions

* Experiment configurations are found in `model*/experiments.conf`
* The parameter `main_metrics` can be selected from coref, ner, relation or any combination of the three, such as coref_ner_relation which indicates the F1 score for averaged F1 score for coref, ner and relation. The model is tuned and saved based on the resulting averaged F1 score.
* The parameters `ner_weight` and `relation_weight` are weights for the multi-task objective. If set the weight to 0 then the task is not trained.
* If coreference is used as an auxiliary task, `coref_weight` is always set to 0. The training of the main task and the training of coreference objective functions take turns. The frequency of how often the auxiliary task is trained is controlled by the parameter `coref_freq`.
* If training coreference as the main task, set `coref_weight` to 1 and `coref_only` flag to 1.
* The number of iteration in CorefProp is controlled by `coref_depth`, the number of iteration in RelProp is controlled by `rel_prop`
* Choose an experiment that you would like to run, e.g. `genia_best_ner`
