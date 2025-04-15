# Model packaging for Replicate

This project holds model packaging folders to build containers to deploy to Replicate using the [cog](https://github.com/replicate/cog) project.

## cog

You will need to select the release of [cog](https://github.com/replicate/cog/releases) to use for a model.
I am using 0.14.4 as of this writing.
You will need to place the `cog` command on your PATH.

On any `cog` command you can add `--debug` for more output.

## uv

Install `uv`

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Model folder

Create folders whose path matches the Huggingface slug for the model.
For example, `ibm-granite/granite-3.3-8b-instruct`.
Change into this folder for the remaining steps.
You will want to populate this folder with the files in git from an existing model folder.
The new folder should contain:

```text
ibm-granite/granite-3.3-8b-instruct
├── .dockerignore
├── cog.yaml
├── predict.py
├── predictor_config.json
├── pyproject.toml
├── requirements.txt
└── weights
    └── .gitignore
```

## pyproject.toml

This file needs to be edited to specify the cog version matching the `cog` command installed earlier.
You will also need to specify the `python-version` for the container and any other python packages, such as `vllm` which are needed by `predict.py`.
Use `==` version specifications for build reproducibility.

## requirements.txt

This file is generated from the `pyproject.toml` file to capture the python package dependencies with exact versions.

```sh
uv pip compile pyproject.toml --output-file requirements.txt
```

If you need to add more packages, edit the `pyproject.toml` file and rerun the `uv pip compile` command.

Once you have built the `requirements.txt` file, you will want to create a virtual env and install the packages in the `requirements.txt` file into the virtual env. Use this virtual env in VSCode to enable code completion/etc.

Just don't put the virtual env folder in the model folder or `cog` will include it in the container image which we don't want.

```sh
uv venv --python 3.11 ../venv
source ../venv/bin/activate
uv pip install --requirements requirements.txt
```

Make sure to use the same python version as specified in `pyproject.toml`.

## cog.yaml

Edit the `image` key to the full name of the image to use when pushing to Replicate.
For example,

```yaml
image: "r8.im/ibm-granite/granite-3.3-8b-instruct:1.0.0"
```

Make sure to update the image version if you have already pushed that version.

Also update any other versions, cuda, python, as needed.
Make sure to use the same python version as specified in `pyproject.toml`.

### predictor_config.json

This file needs to be edited to specify the `served_model_name` in the `engine_arg` and any other desired engine args for vLLM.

## predict.py

This module contains the `setup` and `predict` methods invoked to setup and infer the model.
This code may need changes to support certain models and their parameters such as multimodal parameters.

## Model weights

Model weights are packaged in the container.
They need to be downloaded into the `weights` folder of the model folder.
For example,

```sh
huggingface-cli download --local-dir weights ibm-granite/granite-3.3-8b-instruct
```

These model weight files are not committed to the git repo but are added to the container image.

## Building

To build the container, use the following command.

```sh
cog build --progress plain --separate-weights
```

When the build is done, 2 docker images will be created.
One for the weights and another for the rest which includes the weights as a layer.

```sh
➜ docker image ls
REPOSITORY                                  TAG             IMAGE ID       CREATED       SIZE
r8.im/ibm-granite/granite-3.3-8b-instruct   1.0.0           d3f4991be79d   3 hours ago   34.2GB
r8.im/ibm-granite/granite-3.3-8b-instruct   1.0.0-weights   d643be567003   3 hours ago   16.3GB
```

## Testing

To test the container, you can use the `cog predict` command.

```sh
cog predict r8.im/ibm-granite/granite-3.3-8b-instruct:1.0.0 --progress plain --gpus 1 -i "prompt=What is your name?"
```

You will need to specify the `--gpus` argument to ensure the container can access a GPU.

This will start the container, call `setup`, and call the `predict` method with the specified prompt.

To test using `curl`, you can start the container with

```sh
docker run --rm -p 5000:5000 --gpus 1 r8.im/ibm-granite/granite-3.3-8b-instruct:1.0.0
```

Then from another shell

```sh
curl -s http://localhost:5000/health-check | jq .
curl -s http://localhost:5000/openapi.json | jq .
curl -s http://localhost:5000/predictions -X POST -H 'Content-Type: application/json' -d '{"input": {"prompt": "Who is the all-time winner of the Masters Golf Tournament?"}}' | jq '.output | join("")'
curl -s http://localhost:5000/shutdown -X POST | jq .
```

Use `LocalForward 5000 localhost:5000` in your `.ssh/config` file if you ssh into the build/docker host so you can curl from your local system.

## Deploying

When you are ready to deploy to Replicate, you must first [create the model](https://replicate.com/create) if this is the first container deployment for the model.

Then login to Replicate and push the container to the Replicate container repository.

```sh
cog login
cog push --progress plain --separate-weights
```

After the container is pushed, you will need to go to the Replicate web site and configure the model settings and create a deployment for the model.
