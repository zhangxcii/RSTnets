# deprecated/ — the pre-refactor script

The pre-refactor script and pretrained checkpoint for the periodic-hill
case with $\alpha = 1.0$. Tested with `tensorflow==2.12.0`, `keras==2.12.0`, `deepxde==1.9.1` using older-generation GPU hardware.

The folder contains
- **Data/** — training data at the domain boundaries & reference flow fields for evaluation.
- **Model-adam/**, **Model-lbfgs/** — the pretrained checkpoint (Adam phase, then final L-BFGS-refined model).
- **PINNs.py** — the model class (`RSTnet`).

## Examples

Infer the solution from the pretrained checkpoint:

```sh
python Phills_restore.py
```

Train the model from scratch:

```sh
DDE_BACKEND=tensorflow python Phills.py
```
