# Template example

Copy this directory to start a new flow case:

```sh
cp -r examples/template examples/my_flow
```

## Minimum required to get running

From `examples/my_flow/`:

1. Rename `template_config.json` to `my_flow_config.json` (must match the folder name) and set:
   - `physics.invRe` (required) — the inverse Reynolds number used in the RANS residual.
   - `description` (required) — a one-line description of the flow.
   - Everything else already has a working default — only touch it if your case needs
     something different (network width, precision, backend, coordinate scaling, iteration
     counts — see the top-level README for what each field does).

2. Add your data files under `data/`, matching the filenames/columns listed in the config's
   `data` section:
   - `domain_in.npy` — shape `(N, 2)`, columns `[x, y]`. The full field (used for
     physics-residual collocation points and for plotting/error comparison against `domain_data.npy`).
   - `domain_data.npy` — shape `(N, len(fields))`, columns matching `data.fields`
     (default `[u, v, p, uu, uv, vv]`).
   - `bc_in.npy` — shape `(M, 2)`, columns `[x, y]`. The sparse boundary/supervision point set.
   - `bc_data.npy` — shape `(M, len(fields))`, same column order as `domain_data.npy`.
   - Any files listed under `data.extras` (optional).

3. From the repository root, validate first (no training time spent):

   ```sh
   python rstnets/train.py --case my_flow --dry-run
   ```

4. Then train:

   ```sh
   python rstnets/train.py --case my_flow
   ```

   Checkpoints, loss history, and plots/errors are written into `examples/my_flow/results/`

To produce plots/errors from an already-trained checkpoint:

```sh
python rstnets/restore.py --case my_flow
```

## Optional: custom plots or geometry code

If you need extra figures or geometry-specific postprocessing beyond the default contour
comparison, add that code to your own example directory (not `rstnets/`) and point
`<name>_config.json`'s `postprocess_module` at a `.py` file in this directory (path relative to
the config file) that defines:

```python
def run(case_dir, case_config, data, model, pred, star, x_plot, y_plot, output_dir):
    ...  # pred/star are dicts of field name -> 1D numpy array; save extra figures under output_dir/results/
```

It's called automatically after the default contour-comparison plot. Anything your hook needs
that isn't a training/architecture setting (e.g. a geometry parameter) can go in the config's
free-form `custom` object, read back via `case_config.custom[...]`. See
`examples/phill/postprocess.py` for a worked example (periodic-hill geometry +
streamline/profile-line plots).
