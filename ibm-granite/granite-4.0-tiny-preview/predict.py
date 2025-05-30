# SPDX-License-Identifier: Apache-2.0

# Prediction interface for Cog ⚙️
# https://cog.run/python

# pylint: disable=missing-module-docstring, missing-class-docstring, no-name-in-module, attribute-defined-outside-init

import inspect
import json
import pathlib
import random
import time
import typing

from dataclasses import dataclass, field

import cog
import structlog
import torch

from cog import BasePredictor, ConcatenateIterator, Input, Path as CogPath
from transformers import PreTrainedTokenizerBase
from vllm import (
    EngineArgs,
    LLMEngine,
    RequestOutput,
    SamplingParams,
)
from vllm.sampling_params import RequestOutputKind
from vllm.utils import Counter

class UserError(Exception):
    pass


@dataclass
class PredictorConfig:
    """
    PredictorConfig is a configuration class for the Predictor.

    Attributes:
        chat_template (str | None): A chat template to format the prompt with. If not provided,
                                         the default chat template will be used.
        engine_args (dict[str, str] | None): A dictionary of engine arguments. If not provided,
                                      an empty dictionary will be used.
    """

    chat_template: str | None = None
    engine_args: dict[str, typing.Any] | None = field(default_factory=dict)

    def __post_init__(self):
        if self.engine_args is None:
            self.engine_args = {}
        elif not isinstance(self.engine_args, dict):
            raise UserError(
                "Invalid predictor_config.json: engine_args must be "
                "a valid JSON object that maps to a dictionary."
            )


# pylint: disable=invalid-overridden-method, signature-differs
class Predictor(BasePredictor):
    logger = structlog.get_logger(__name__)

    def setup(self, weights: CogPath) -> None:
        # Model weights must be in the "weights" folder.
        # This can be overridden with the COG_WEIGHTS env var.
        structlog.contextvars.clear_contextvars()
        self.config = self.load_config(weights)
        log = self.logger.bind()
        log.info("setup() commencing")

        engine_args = self.config.engine_args or {}
        engine_args["model"] = weights.resolve().as_posix()
        if "dtype" not in engine_args:
            engine_args["dtype"] = "auto"
        if "tensor_parallel_size" not in engine_args:
            engine_args["tensor_parallel_size"] = max(torch.cuda.device_count(), 1)

        engine_args = EngineArgs(**engine_args)
        log.debug("EngineArgs", engine_args=engine_args)

        try:
            self.engine = LLMEngine.from_engine_args(engine_args)
        except TypeError as e:
            log.error("Unexpected EngineArg", exc_info=e)
            raise
        except Exception as e:
            log.error("VLLM Unknown Error", exc_info=e)
            raise

        self.tokenizer = typing.cast(
            PreTrainedTokenizerBase, self.engine.get_tokenizer_group().get_lora_tokenizer()
        )

        if self.config.chat_template:
            self.chat_template = self.config.chat_template
            log.debug(
                "Using chat template from predictor_config.json",
                chat_template=self.chat_template,
            )
        elif self.tokenizer.chat_template:
            self.chat_template = self.tokenizer.get_chat_template()
            log.debug(
                "Using chat template from tokenizer", chat_template=self.chat_template
            )
        else:
            raise UserError(
                "No prompt template specified in predictor_config.json or tokenizer"
            )

        self.request_counter = Counter()

        self._testing = True
        generator = self.predict(
            **dict(self._defaults, **{"max_tokens": 50, "prompt": "What is your name?"})
        )
        test_output = "".join(generator)
        structlog.contextvars.clear_contextvars()
        log.debug("Test prediction output", test_output=test_output)
        self._testing = False
        log.info("setup() complete")

    def predict(  # pylint: disable=invalid-overridden-method, arguments-differ, too-many-arguments, too-many-positional-arguments, too-many-locals
        self,
        *,
        # prompt must be the first argument
        # The LangChain Replicate class will use the first argument to supply the prompt
        prompt: str = Input(
            description="User prompt to send to the model.", default=""
        ),
        system_prompt: str | None = Input(
            description="System prompt to send to the model."
            "The chat template provides a good default.",
            default=None,
        ),
        min_tokens: int = Input(
            description="The minimum number of tokens the model should generate as output.",
            default=0,
        ),
        max_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=512,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=0.6,
        ),
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep "
            "the top tokens with cumulative probability >= top_p (nucleus filtering). "
            "Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=0.9,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating "
            "the output. If > 0, only keep the top k tokens with highest probability "
            "(top-k filtering).",
            default=50,
        ),
        presence_penalty: float = Input(description="Presence penalty", default=0.0),
        frequency_penalty: float = Input(description="Frequency penalty", default=0.0),
        stop_sequences: str | None = Input(
            description="A comma-separated list of sequences to stop generation at. "
            "For example, '<end>,<stop>' will stop generation at the first instance of "
            "'end' or '<stop>'.",
            default=None,
        ),
        chat_template: str | None = Input(
            description="A template to format the prompt with. If not provided, "
            "the default prompt template will be used.",
            default=None,
        ),
        seed: int | None = Input(
            description="Random seed. Leave blank to randomize the seed.",
            default=None,
        ),
    ) -> ConcatenateIterator[str]:  # type: ignore
        start_time = time.time()
        structlog.contextvars.clear_contextvars()
        request_id = str(next(self.request_counter))
        if not seed:
            seed = int(random.randint(0, 100000))
        structlog.contextvars.bind_contextvars(request_id=request_id, user_prompt=prompt)
        log = self.logger.bind()
        log.info("predict() commencing")

        if not system_prompt and prompt.lstrip().startswith("<|start_of_role|>"):
            formatted_prompt = prompt
            log.debug(
                "Using user prompt as formatted prompt ",
                formatted_prompt=formatted_prompt,
            )
        else:
            if not chat_template:
                chat_template = self.chat_template  # type: ignore
            conversation = []
            if system_prompt:
                conversation.append({"role": "system", "content": system_prompt})
            conversation.append({"role": "user", "content": prompt})

            formatted_prompt = typing.cast(
                str,
                self.tokenizer.apply_chat_template(
                    conversation=conversation,
                    chat_template=chat_template,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )
            log.debug(
                "Formatted prompt using chat template",
                formatted_prompt=formatted_prompt,
            )

        sampling_params = SamplingParams(
            n=1,
            top_k=(-1 if (top_k or 0) == 0 else top_k),
            top_p=top_p,
            temperature=temperature,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],  # type: ignore
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        if isinstance(stop_sequences, str) and stop_sequences:
            sampling_params.stop = stop_sequences.split(",")
        else:
            sampling_params.stop = (
                list(stop_sequences) if isinstance(stop_sequences, list) else []
            )
        log.debug("SamplingParams", sampling_params=sampling_params)

        self.engine.add_request(
            request_id,
            formatted_prompt,
            sampling_params,
        )
        while self.engine.has_unfinished_requests():
            for result in type(self.engine).validate_outputs(self.engine.step(), RequestOutput):
                if not result.finished:
                    continue
                assert len(result.outputs) == 1, (
                    "Expected exactly one output from generation request."
                )
                output = result.outputs[0]
                text = output.text
                # Normalize text by removing any incomplete surrogate pairs (common with emojis)
                text = text.replace("\N{REPLACEMENT CHARACTER}", "")

                yield text  # type: ignore
                log.debug("result", text=text, finish_reason=output.finish_reason)

        log.info(f"Generation took {time.time() - start_time:.2f}s")

        if not self._testing:
            # pylint: disable=undefined-loop-variable
            cog.current_scope().record_metric(
                "input_token_count", len(result.prompt_token_ids or [])
            )
            cog.current_scope().record_metric(
                "output_token_count", len(result.outputs[0].token_ids)
            )

        log.info("predict() complete")

    _defaults = {
        key: param.default.default
        for key, param in inspect.signature(predict).parameters.items()
        if hasattr(param.default, "default")
    }

    def load_config(self, weights: pathlib.Path) -> PredictorConfig:
        """
        Load the predictor configuration from the specified weights directory or
        the current directory.

        Load `predictor_config.json` from the weights directory or current directory.
        Return a default PredictorConfig object if not found or an error occurs.

        Priority:
        1. Load `predictor_config.json` from the specified weights directory.
        2. If not found, load `predictor_config.json` from the current directory.
        3. If not found or an error occurs, return a default PredictorConfig object.

        Args:
            weights (pathlib.Path): The path to the weights directory.

        Returns:
            PredictorConfig: The loaded predictor configuration.
        """
        log = self.logger.bind()

        predictor_config_path = weights.joinpath("predictor_config.json")
        if not predictor_config_path.exists():
            predictor_config_path = pathlib.Path("predictor_config.json")
            if not predictor_config_path.exists():
                predictor_config_path = None
        if predictor_config_path:
            try:
                log.debug("Loading predictor_config.json", path=predictor_config_path)
                with predictor_config_path.open(
                    mode="r",
                    encoding="utf-8",
                ) as f:
                    config = json.load(f)
                config = PredictorConfig(**config)
            except Exception as e:
                raise UserError(f"Invalid predictor_config.json: {e}") from e
        else:
            config = PredictorConfig()

        log.debug("PredictorConfig", config=config)
        return config
