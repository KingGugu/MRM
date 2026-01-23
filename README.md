# Code for Review (CDConv as backbone)

Code for Enhancing Protein Representation Learning via Manifold Restore Mixing.

The complete code, dataset, and weights will be released upon acceptance of the paper.

## Run the Code

We take the Fold Classification task as an example. Other tasks are similar to it.

First, Extract the `fold.tar.gz` file in the `protein-data` directory.

Then, we need to train the base model using the original data, which is the first stage of `4.3 Two-Stage Regularized Training` in our paper.

```
python fold.py --num-epochs 200 --lr-milestones 100 150
```

Finally, we enable MRM to further enhance the model's performance with the manifold restored data. It is the second stage of `4.3 Two-Stage Regularized Training` in our paper. After running the follow command, the weight of the base model will be loaded.
```
python fold.py --num-epochs 200 --lr-milestones 100 --mrm --base_weight [Location of the .pt file]
```



