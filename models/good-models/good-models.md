

# roberta base 

cryptok-540
epoch=1-step=807.ckpt
using [SEP] during training between sentence and topic
mAP strict= 0.880274642049566 ; mAP relaxed = 0.962832459686845


cryptok-390
map strict 0.75, relax 0.904



with undecided pair and training like:
sentence1 = topic + arg
sentence2 = topic + kp
then embed as tokenizer(sentence1, sentence2)

less performance with text = topic + arg + kp then tokenizer(text)

cryptok-531
epoch=1-step=807.ckpt
mAP strict= 0.879076361656189 ; mAP relaxed = 0.9624009132385254




with sentence embeddings:

{'map_relaxed': 0.9501110911369324,
 'map_strict': 0.8497300148010254,
 'test_accuracy': 0.7196023464202881,
 'test_f1': 0.7196023464202881,
 'test_loss': 0.5933224558830261,
 'test_precision': 0.7196023464202881,
 'test_recall': 0.7196023464202881}



# roberta base kfold  - best
crypto 440
mAP strict= 0.8170706147673692 ; mAP relaxed = 0.9380859036320186

```
i.e., >15% annotators yet <60% of them marked the pair as a match
(as detailed in Bar-Haim et al., ACL-2020).
```

Kfold + using undecided paris with as match

{
 'map_relaxed': 0.9380859136581421,
 'map_strict': 0.8170706033706665,
 'test_loss': 0.8050061464309692,
 'val_accuracy': 0.6186591982841492,
 'val_f1': 0.6186591982841492,
 'val_precision': 0.6186591982841492,
 'val_recall': 0.6186591982841492
}

all_mpnet_base - CRYPTOK-527
{
 'map_relaxed': 0.7894099354743958,
 'map_strict': 0.6092702746391296,
 'test_accuracy': 0.6744838356971741,
 'test_f1': 0.6744838356971741,
 'test_loss': 0.8435215950012207,
 'test_precision': 0.6744838356971741,
 'test_recall': 0.6744838356971741
}


roberta-base => contrastive  - CRYPTOK-526

{
 'map_relaxed': 0.967033326625824,
 'map_strict': 0.8787113428115845,
 'test_accuracy': 0.6334437727928162,
 'test_f1': 0.6334437727928162,
 'test_loss': 0.027558164671063423,
 'test_precision': 0.6334437727928162,
 'test_recall': 0.6334437727928162
}


roberta-base => triplet - CRYPTOK-525
{
 'map_relaxed': 0.597618043422699,
 'map_strict': 0.4805447459220886,
 'test_accuracy': 0.267397403717041,
 'test_f1': 0.267397403717041,
 'test_loss': 4.6805901527404785,
 'test_precision': 0.267397403717041,
 'test_recall': 0.267397403717041
}

# Roberta-large (or other large models)
overfits, needs away to better tune it. 


