# Reproduction Guide

To make reproduction easier, we provide a script which will download all data and models required for the experiments.

## Setup Script

To execute the script we assume that you followed the steps in the [Getting Started](getting_started.md) guide and have the environment activated.

The script will by default:
- Download the data for the convexity experiments
- Download the data for the path-connectedness experiments
- Download the pre-trained Unet models for the path-connectedness experiments, from [Kardoost et al.](https://pure.mpg.de/rest/items/item_3353327/component/file_3499084/content)
- Download the labels for the path-connectedness experiments
- Download the pre-trained model and code for [Soft-Semantic-Segmentation](https://yaksoy.github.io/sss/) and the features for convexity experiments
- Download pre-train states for our path-connectedness model

This might take a while, depending on your internet connection. You can also customize the script to only download specific parts of the data.
in the [SetupConfig.py](../awesome/util/setup_config.py) file you can set the following flags to True or False depending on what you want to download.
It will create marker files for each part of the data, so you can run the script multiple times without re-downloading everything.

The default execution of the script will download all data and models.

```bash
python scripts/setup.py
```

You can set the flags as specified in the [SetupConfig.py](../awesome/util/setup_config.py) by using cli arguments. Setting the flag will invert the default value.
E.g.
```bash
python scripts/setup.py --download-fbms-data
```
will download everything except the fbms data.

You can get a list of all available flags by using the `--help` argument.
```bash
python scripts/setup.py --help
```

In general, you can set all fields defined in the SetupConfig.py when you replace `_` with `-` in the argument name and provide it with "--" when invoking the script.

The fbms data and labels required for path-connectedness are quite large ~30 GB. To ignore these, use the following command:

```bash
python scripts/setup.py --download-fbms-data --download-uncertainty-multicut-models --download-uncertainty-multicut-labels
```

You can run the script multiple times to download different parts of the data, it will by default not overwrite existing files or download files that are already present.

## Experiment Execution

In general all our experiments have the same entry point, the [run.py](../scripts/run.py) script. The script will run the experiment specified by a configuration.
We are using the [AwesomeConfig](../awesome/run/awesome_config.py) object, which in combination with the [AwesomeRunner](../awesome/run/awesome_runner.py) will set everything up and make it ready for execution.

Similar to the Setup script, you can set arguments for run.py as cli-arguments, or you can set them in a configuration file (.yaml or .json) and provide the path to the configuration file as the `--config-path` argument.

We are providing our configuration files in the [config](../config) folder. The configuration files are structured in subfolders, where each subfolder represents a different experiment. The configuration files are named according to the experiment they are used for.

Example:
```bash
python scripts/run.py --config-path config/convexity/sequential/CNNET_benchmark+feat+convex.yaml
```

Would run the experiment for the CNN sequential fit of the convexity experiment, with compressed feature inputs from Soft-Semantic-Segmentation.

> Note: If you did not used the default paths during the setup, you might need to adjust the paths in the configuration files. We have a notebook [process_config_paths.ipynb](../notebooks/process_config_paths.ipynb) which will help you to adjust the paths in the configuration files.

For those who are using vscode, we also provide a launch configuration in the [.vscode](../.vscode) folder, which will allow you to run the experiments directly from vscode. The config file can be picked from the dropdown menu. This will require the [Task Shell Input](https://marketplace.visualstudio.com/items?itemName=augustocdias.tasks-shell-input) extension to be installed.

### Convexity

The configuration files for the convexity experiments are located in the `config/convexity` folder.
These are further divided into the following subfolders:
- sequential: for the sequential fit
- joint: for the joint fit training of the convex net and the normal predictor

For example, to run the CNN with feature inputs from Soft-Semantic-Segmentation and the convexity prior in a sequential fashion, you can use the following configuration file:

```bash
python scripts/run.py --config-path config/convexity/sequential/CNNET_benchmark+feat+convex.yaml
```


The primary implementation of the convex net can be found in awesome/models/convex_net.py

For demonstrating the sequential fit, we provide a notebook [convexity.ipynb](../notebooks/how_to/convexity.ipynb) which will guide you through the process of training the model and evaluating it.

### Path-Connectedness

The configuration files for the path-connectedness experiments are located in the `config/path-connectedness` folder.

Also here we have the subfolders:
- sequential: for the sequential fit
- joint: for the joint fit training of the path-connected net and the normal predictor
- refit-unet: Used to only refit the Unet model. Will produce somewhat equivalent models to the Kardoost et al. models
- spatio-temporal: for the spatio-temporal training of the cars3 sequence (from the paper appendix)
- noisy-spatio-temporal: for the noisy spatio-temporal training of the cars3 sequence (from paper appendix)
- ...

The primary implementation of the path-connected net can be found in awesome/model/path_connected_net.py

Also here we provide a notebook [path-connectedness.ipynb](../notebooks/how_to/path-connectedness.ipynb) which will guide you through the process of training the model and evaluating it.

## Evaluation

The evaluation as well as plot generation are done via multiple notebooks located in the `notebooks` folder.

To name a few:

Convexity:

- notebooks/unireps_evaluation.ipynb
- notebooks/unireps_qualitative_results.ipynb

For path-connectedness:

- notebooks/fbms_eval_icml.ipynb

We also provide a bunch of other notebooks which might be helpful for understanding the data or the models. Unfortunately, there is not much of an explanation in the notebooks, contact us if you have any questions.

The code for the teaser figures and our other constraints can be found in the `notebooks` folder as well.

- [Convexity](../notebooks/icml_teaser_code/convex/convex.ipynb)
- [Convexity in XY and Depth](../notebooks/icml_teaser_code/convex-depth/convex.ipynb)
- [Star-Shape](../notebooks/icml_teaser_code/star_shaped/star.ipynb)
- [Rotational Symmetry](../notebooks/icml_teaser_code/rotation_symmetric/rotation_symmetric.ipynb)
- [Periodic](../notebooks/icml_teaser_code/repeating/repeating.ipynb)
- [Path-Connectedness](../notebooks/icml_teaser_code/connectedness/diffeo_convex.ipynb)
- [Spatio-Temporal Connectedness](../notebooks/icml_teaser_code/temporal_connectedness/temporal_connection.ipynb)


## Remarks regarding Soft-Semantic-Segmentation

We used the features provided by [Soft-Semantic-Segmentation](https://yaksoy.github.io/sss/) for the convexity experiments as additional input for our base segmentors. The implementation is based on tensorflow and only runs on tensorflow versions < 2.0. Therefore, we provided pre-computed features in the setup script.

If you want to compute the features yourself, you must install a valid tensorflow version, which is < 2.0.
