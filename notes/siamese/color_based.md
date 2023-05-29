Experiment with color-based features with variants of:
- use a single feature. Then, possibly add an ensemble step to combine information. [Details](#siamese-model-with-single-color-based-feature)
- combine 2-3 features directly in the training process (to see if a complicated model can outperform single models with ensemble step): [Details](#combing-different-features-during-training-process)
  + directly concat in feature vector (ref: [SiameseColorbased](../../model/siamese.py#L60))
  + use 2 FC blocks, each for 1 feature (ref: [SiameseColorbased2](../../model/siamese.py#L74))


## Siamese model with single color-based feature
Use one feature to train a model. Abbreviation:
- A: average-color
- B: brightness
- C: color-variance

### rank-k acc for first train
- siamese502-colorbased-A-on-0_45.npz: [0.3564356435643564, 0.594059405940594, 0.7029702970297029]
- siamese502-colorbased-B-on-0_45.npz: [0.36633663366336633, 0.45544554455445546, 0.504950495049505]
- siamese502-colorbased-C-on-0_45.npz: [0.37623762376237624, 0.5544554455445545, 0.6237623762376238]


### rank-k acc for second train
- siamese502-colorbased-A-on-0_45.npz: [0.32673267326732675, 0.5346534653465347, 0.6336633663366337]
- siamese502-colorbased-B-on-0_45.npz: [0.3564356435643564, 0.46534653465346537, 0.5346534653465347]
- siamese502-colorbased-C-on-0_45.npz: [0.32673267326732675, 0.45544554455445546, 0.5346534653465347]

### ensemble step (v1): same feature - different model
combine prediction from 2 models resulted from 2 different training.
#### ensemble result without normalization
Prediction of each model are directly concatenated into single final vector.

- A_45_avg.npz: [0.3465346534653465, 0.5445544554455446, 0.693069306930693]
- B_45_avg.npz: [0.33663366336633666, 0.48514851485148514, 0.5445544554455446]
- C_45_avg.npz: [0.37623762376237624, 0.5643564356435643, 0.6138613861386139]

- A_45.npz: [0.3465346534653465, 0.5445544554455446, 0.6831683168316832]
- B_45.npz: [0.32673267326732675, 0.48514851485148514, 0.5445544554455446]
- C_45.npz: [0.38613861386138615, 0.5544554455445545, 0.6237623762376238]


#### ensemble result with normalization
Each output vector are normalized to mean 0 std 1 before combined.
An interesting fact is, average and concatenate operator yield the same rank-k acc.
##### average operator
- A_45_avg.npz: [0.36633663366336633, 0.5247524752475248, 0.6534653465346535]
- B_45_avg.npz: [0.33663366336633666, 0.43564356435643564, 0.504950495049505]
- C_45_avg.npz: [0.38613861386138615, 0.5148514851485149, 0.5742574257425742]

##### concatenate operator
- A_45.npz: [0.36633663366336633, 0.5247524752475248, 0.6534653465346535]
- B_45.npz: [0.33663366336633666, 0.44554455445544555, 0.504950495049505]
- C_45.npz: [0.38613861386138615, 0.5148514851485149, 0.5742574257425742]

### ensemble step (v2): different features - different models
Combine prediction from 2 models which are trained independently with different features, obviously, it yielded the highest rank-k acc.
- AB_45_1.npz: [0.39603960396039606, 0.6633663366336634, 0.7227722772277227]
- BC_45_1.npz: [0.42574257425742573, 0.5841584158415841, 0.6633663366336634]


## SIFT + Brightness
siamese502-colorbased-B-on-0_5.npz: [0.3069306930693069, 0.38613861386138615, 0.42574257425742573]

## Combing different features during training process
Purpose of this experiment is trying to give more information (2 features, instead of 1) for model and hope model can learn better with more information. Comparison with model that is trained with single feature, and then performing ensemble step (v1, v2), the complicated model with more features are performing worse than expected.
Note:
- BatchNorm layer is used as replacement for normalization step (convert vector to mean 0 std 1).
- The last FC layer that combines output from 2 single FC blocks acts as ensemble operator.
We came to a conclusion that, there is an improvement with adding BatchNorm layer after each block, and it gave better results by adding a last FC layer to concat output of 2 blocks. However, still the complicated model can't beat the simple ensemble step (concat operator) with 2 single models.

### with BatchNorm layer
#### with last FC layer
- siamese502-colorbased2-AB-on-fc-0_40.npz: [0.37623762376237624, 0.504950495049505, 0.5841584158415841]
siamese502-colorbased2-BC-on-fc-0_50.npz: [0.36633663366336633, 0.5742574257425742, 0.6435643564356436]
##### train more epoch (more computation, higher acc)
- AB-on-fc-0_75.npz: [0.39603960396039606, 0.6435643564356436, 0.7821782178217822]
- BC-on-fc-0_65.npz: [0.39603960396039606, 0.5841584158415841, 0.6534653465346535]

#### without last FC layer
- siamese502-colorbased2-AB-on-0_15.npz: [0.33663366336633666, 0.49504950495049505, 0.5742574257425742]
- siamese502-colorbased2-BC-on-0_35.npz: [0.37623762376237624, 0.504950495049505, 0.5643564356435643]

### without BatchNorm layer, with last FC layer
- siamese502-colorbased2-BC-on-0_45.npz: [0.3564356435643564, 0.49504950495049505, 0.5742574257425742]
- siamese502-colorbased2-AB-on-0_30.npz: [0.3564356435643564, 0.5445544554455446, 0.6336633663366337]


## Selective training
In this experiment, instead of using all triplets, we select the hard triplets for next training iteration. Here, top _k %_ triplets with highest loss will are recorded.
The purpose is forcing model focuses on hard instance only and speed-up the training process while trying to maintain the same rank-k acc.
### k = 50%
- siamese502-colorbased2-AB-on-0_80.npz: [0.3564356435643564, 0.49504950495049505, 0.5346534653465347]
- siamese502-colorbased2-BC-on-fc-0_80.npz: [0.3564356435643564, 0.5445544554455446, 0.5841584158415841]

### k = 10%
- siamese502-colorbased2-AB-on-0_50.npz: [0.32673267326732675, 0.43564356435643564, 0.49504950495049505]
- siamese502-colorbased2-BC-on-fc-0_80.npz: [0.3564356435643564, 0.504950495049505, 0.5346534653465347]
