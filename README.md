# Model packaging for Replicate

This project holds model packaging folders to build containers to deploy to Replicate using the [cog](https://github.com/replicate/cog) project.

## Tools

### cog

You will need to select the release of [cog](https://github.com/replicate/cog/releases) to use for a model.
I am using 0.21.0 as of this writing.
You will need to place the `cog` command on your PATH.

On any `cog` command you can add `--debug` for more output.

### uv

Install `uv`

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### docker

You will need to install Docker CE, and its `buildx` plugin, and the Nvidia container toolkit.

For example,

```sh
sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo systemctl start docker
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
docker run --rm  --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility ubuntu nvidia-smi
```

Also see <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html>.
You may need to edit `/etc/nvidia-container-runtime/config.toml` to change `supported-driver-capabilities` and restart Docker.

```toml
supported-driver-capabilities = "compute,utility"
```

## Model folder

Create folders whose path matches the Huggingface slug for the model.
For example, `ibm-granite/granite-4.2-8b`.
Change into this folder for the remaining steps.
You will want to populate this folder with the files in git from an existing model folder.
The new folder should contain:

```text
ibm-granite/granite-4.2-8b
├── .dockerignore
├── cog.yaml
├── pyproject.toml
├── requirements.txt
├── run.py
├── runner_config.json
└── tests
    └── *.json
└── weights
    └── .gitignore
```

### pyproject.toml

This file needs to be edited to specify the cog version matching the `cog` command installed earlier in `sdk_version`.
You will also need to specify the `python-version` for the container and any other python packages, such as `vllm` which are needed by `run.py`.
Use `==` version specifications for build reproducibility.

### requirements.txt

This file is generated from the `pyproject.toml` file to capture the python package dependencies with exact versions.

```sh
uv pip compile pyproject.toml --output-file requirements.txt
```

If you need to add more packages, edit the `pyproject.toml` file and rerun the `uv pip compile` command.

Once you have built the `requirements.txt` file, you will want to create a virtual env and install the packages in the `requirements.txt` file into the virtual env. Use this virtual env in VSCode to enable code completion/etc.

Just don't put the virtual env folder in the model folder or `cog` will include it in the container image which we don't want.

```sh
mkdir -p ../venv/$(basename $PWD)
uv venv --python 3.13 ../venv/$(basename $PWD)
ln -s ../venv/$(basename $PWD) .venv
source .venv/bin/activate
uv pip install --requirements requirements.txt
uv pip install cog==0.21.0
```

Make sure to use the same python version as specified in `pyproject.toml`.

### cog.yaml

Edit the `image` key to the name of the image to use when pushing to Replicate.
For example,

```yaml
image: "r8.im/ibm-granite/granite-4.2-8b"
```

Make sure to update the image version if you have already pushed that version.

Also update any other versions, cuda, python, as needed.
Make sure to use the same python version as specified in `pyproject.toml`.

#### runner_config.json

This file needs to be edited to specify the `served_model_name` in the `engine_arg` and any other desired engine args for vLLM.

### run.py

This module contains the `setup` and `run` methods invoked to setup and infer the model.
This code may need changes to support certain models and their parameters such as multimodal parameters.

### weights

Model weights are packaged in the container and are also needed for local testing.
They need to be downloaded into the `weights` folder of the model folder.
For example,

```sh
hf download --local-dir weights ibm-granite/granite-4.2-8b
```

These model weight files are not committed to the git repo but are added to the container image.

## Local testing

The `__main__` block at the bottom of `run.py` lets you run inference directly from the command line, loading the model from the local `weights` folder.

```sh
# Run against one or more JSON test fixtures
python run.py tests/prompt.json
python run.py tests/chat.json tests/chat_thinking.json
```

Each fixture file is a plain JSON object whose keys map to `run()` input parameters. The test files in `tests` folder cover the main usage patterns.

## Building container

To build the container, use the following command.

```sh
cog build --progress plain --separate-weights --use-cog-base-image
```

When the build is done, 2 docker images will be created.
One for the weights and another for the rest which includes the weights as a layer.

```sh
➜ docker image ls
IMAGE
r8.im/ibm-granite/granite-4.2-8b:latest
r8.im/ibm-granite/granite-4.2-8b-weights:latest
```

## Testing container

To test the container, you can use the `cog run` command.

```sh
cog run r8.im/ibm-granite/granite-4.2-8b-instruct:latest --progress plain --gpus all --json @tests/chat.json
```

You will need to specify the `--gpus` argument to ensure the container can access a GPU.

This will start the container, call `setup`, and call the `run` method with the specified prompt.

To test using `curl`, you can start the container with

```sh
docker run --rm -p 5000:5000 --gpus 1 r8.im/ibm-granite/granite-4.2-8b:latest
```

Then from another shell

```sh
curl -s http://localhost:5000/health-check | jq .
curl -s http://localhost:5000/openapi.json | jq .
curl -s http://localhost:5000/predictions -X POST -H 'Content-Type: application/json' -d '{"input": {"prompt": "Who is the all-time winner of the Masters Golf Tournament?"}}' | jq '.output | join("")'
curl -s http://localhost:5000/shutdown -X POST | jq .
```

Use `LocalForward 5000 localhost:5000` in your `.ssh/config` file if you ssh into the build/docker host so you can curl from your local system.

You can inspect the generated openapi schema for the run methods inputs with

```sh
docker inspect r8.im/ibm-granite/granite-4.2-8b:latest --format='{{index .Config.Labels "run.cog.openapi_schema"}}' | jq '.'
```

## Deploying container

When you are ready to deploy to Replicate, login to Replicate and push the container to the Replicate container repository.

```sh
cog login
docker tag r8.im/ibm-granite/granite-4.2-8b:latest r8.im/ibm-granite/granite-4.2-8b:1.0.0
docker push r8.im/ibm-granite/granite-4.2-8b:1.0.0
```

After the container is pushed, you will need to go to the Replicate web site and configure the model settings and create a deployment for the model.
