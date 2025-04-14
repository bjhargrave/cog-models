# Model packaging for Replicate

This project holds model packaging to deploy to Replicate using the [cog](https://github.com/replicate/cog/tree/main) project.

## Model weights

Model weights are packaged in the container. They need to be downloaded into the `weights` folder of the model folder. For example,

```sh
huggingface-cli download --local-dir weights ibm-granite/granite-3.3-8b-instruct
```

## Building

TBD

## Testing

TBD

## Deploying

TBD
