# SPDX-License-Identifier: Apache-2.0

# Prediction interface for Cog ⚙️
# https://cog.run/python

# pylint: disable=missing-module-docstring, missing-class-docstring, no-name-in-module, attribute-defined-outside-init, wrong-import-position

import asyncio
import inspect
import json
import logging
import pathlib
import os
import sys
import typing

from cog import AsyncConcatenateIterator, BasePredictor, Input
from cog import Path as CogPath
from cog.coder import Coder
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from openai.types.chat import ChatCompletionToolParam

# Set before importing any of vLLM's packages
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from vllm import envs
from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    CustomChatCompletionMessageParam,
    load_chat_template,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.utils.counter import Counter
from vllm.v1.engine.async_llm import AsyncLLM


def init_logger(name: str) -> logging.Logger:
    """Create a Logger for the specified name.

    Args:
        name (str): The name for the Logger.

    Returns:
        logging.Logger: A configured Logger for the specified name.
    """
    _logger = logging.Logger(name)

    _logger.setLevel(os.environ.get("PREDICTOR_LOG_LEVEL", "DEBUG").upper())
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.DEBUG)
    _logger.addHandler(handler)

    return _logger


logger = init_logger(__name__)


class ResponseError(Exception):
    pass


class PredictorConfig(BaseModel):
    """
    PredictorConfig is a configuration class for the Predictor.
    """

    model_config = ConfigDict(extra="allow")

    chat_template: str | None = Field(default=None)
    chat_template_content_format: ChatTemplateContentFormatOption = Field(
        default="auto"
    )
    enable_log_requests: bool = Field(default=False)
    max_log_len: int | None = Field(default=None)
    log_error_stack: bool = Field(default=envs.VLLM_SERVER_DEV_MODE)
    engine_args: dict[str, typing.Any] = Field(default_factory=dict)


type Embedding = list[float]


class ChatCompletionDocumentParam(typing.TypedDict, total=False):
    """
    Chat Completion API documents.
    """

    text: str
    doc_id: str | int | None
    title: str | None


class JsonSchemaResponseFormat(typing.TypedDict, total=False):
    name: typing.Required[str]
    description: str | None
    schema: dict[str, typing.Any] | None


class ResponseFormat(typing.TypedDict, total=False):
    type: typing.Required[typing.Literal["text", "json_object", "json_schema"]]
    json_schema: JsonSchemaResponseFormat | None


class TypedDictCoder(Coder):
    """This class lets cog accept TypedDict as a dict."""

    @staticmethod
    def factory(tpe: type) -> typing.Optional["TypedDictCoder"]:
        if issubclass(tpe, dict):
            return TypedDictCoder()
        return None

    def encode(self, x: dict) -> dict:
        return x

    def decode(self, x: dict) -> dict:
        return x


# pylint: disable=invalid-overridden-method, signature-differs
class Predictor(BasePredictor):
    async def setup(
        self,
        weights: CogPath | str | None,
    ) -> None:
        if not weights:
            weights = CogPath("/src/weights")
        elif isinstance(weights, str):
            weights = CogPath(weights)
        self.config = self.load_config(weights)
        logger.info("setup() commencing")

        self.resolved_chat_template = load_chat_template(self.config.chat_template)
        if self.resolved_chat_template:
            logger.debug("Using chat template from predictor_config.json")

        engine_args = AsyncEngineArgs(**self.config.engine_args)
        if "model" not in self.config.engine_args:
            engine_args.model = weights.resolve().as_posix()
        if "tensor_parallel_size" not in self.config.engine_args:
            engine_args.tensor_parallel_size = max(torch.cuda.device_count(), 1)

        logger.debug("AsyncEngineArgs engine_args=%s", engine_args)

        self.engine = AsyncLLM.from_engine_args(engine_args)
        # try:
        # except TypeError:
        #     logger.error("Unexpected EngineArg")
        #     raise
        # except Exception:
        #     logger.error("VLLM Unknown Error")
        #     raise

        vllm_config: VllmConfig = self.engine.vllm_config
        model_config: ModelConfig = vllm_config.model_config

        supported_tasks = await self.engine.get_supported_tasks()
        logger.debug("Supported_tasks: %s", supported_tasks)

        request_logger = (
            RequestLogger(max_log_len=self.config.max_log_len)
            if self.config.enable_log_requests
            else None
        )

        self.request_counter = Counter()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # async def predict(  # pylint: disable=arguments-differ
    #     self,
    #     texts: list[str] = Input(
    #         description="A list of texts to embed.",
    #         default=[],
    #     ),  # pyright: ignore[reportArgumentType]
    #     normalize: bool = Input(
    #         description="Normalize the embeddings",
    #         default=True,
    #     ),  # pyright: ignore[reportArgumentType]
    # ) -> AsyncIterator[Any]:
    async def predict(  # pylint: disable=invalid-overridden-method, arguments-differ, too-many-arguments, too-many-positional-arguments, too-many-locals, unused-argument
        self,
        # prompt must be the first argument
        # The LangChain Replicate class will use the first argument to supply the prompt
        prompt: str | None = Input(
            description="Completion API user prompt.", default=None
        ),  # pyright: ignore[reportArgumentType]
        messages: list[CustomChatCompletionMessageParam] = Input(
            description="Chat completion API messages.",
            default=[],
        ),  # pyright: ignore[reportArgumentType]
        documents: list[ChatCompletionDocumentParam] = Input(
            description="Documents for request. Passed to the chat template.",
            default=[],
        ),  # pyright: ignore[reportArgumentType]
        tools: list[ChatCompletionToolParam] = Input(
            description="Tools for request. Passed to the chat template.",
            default=[],
        ),  # pyright: ignore[reportArgumentType]
        tool_choice: str | None = Input(
            description="Tool choice for request. "
            "If the choice is a specific function, this should be specified as a JSON string.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        response_format: ResponseFormat | None = Input(
            description="An object specifying the format that the model must output.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        system_prompt: str | None = Input(
            description="Completion API system prompt. "
            "The chat template provides a good default.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        chat_template: str | None = Input(
            description="A template to format the prompt with. If not specified, "
            "the chat template provided by the model will be used.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        add_generation_prompt: bool = Input(
            description="Add generation prompt. Passed to the chat template. Defaults to True.",
            default=True,
        ),  # pyright: ignore[reportArgumentType]
        chat_template_kwargs: dict[str, typing.Any] = Input(
            description="Additional arguments to be passed to the chat template.",
            default={},
        ),  # pyright: ignore[reportArgumentType]
        min_tokens: int = Input(
            description="The minimum number of tokens the model should generate as output.",
            default=0,
        ),  # pyright: ignore[reportArgumentType]
        max_tokens: int | None = Input(
            description="max_tokens is deprecated in favor of the max_completion_tokens field.",
            default=None,
            deprecated=True,
        ),  # pyright: ignore[reportArgumentType]
        max_completion_tokens: int | None = Input(
            description="An upper bound for the number of tokens that can be generated for a completion, "
            "including visible output tokens and reasoning tokens.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=0.0,
        ),  # pyright: ignore[reportArgumentType]
        top_p: float = Input(
            description="A probability threshold for generating the output. If < 1.0, only keep "
            "the top tokens with cumulative probability >= top_p (nucleus filtering). "
            "Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751).",
            default=0.9,
        ),  # pyright: ignore[reportArgumentType]
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating "
            "the output. If > 0, only keep the top k tokens with highest probability "
            "(top-k filtering).",
            default=50,
        ),  # pyright: ignore[reportArgumentType]
        presence_penalty: float | None = Input(
            description="Presence penalty", default=None
        ),  # pyright: ignore[reportArgumentType]
        frequency_penalty: float | None = Input(
            description="Frequency penalty", default=None
        ),  # pyright: ignore[reportArgumentType]
        repetition_penalty: float | None = Input(
            description="Repetition penalty", default=None
        ),  # pyright: ignore[reportArgumentType]
        stop: list[str] = Input(
            description="A list of sequences to stop generation at. "
            'For example, ["<end>","<stop>"] will stop generation at the first instance of '
            '"<end>" or "<stop>".',
            default=[],
        ),  # pyright: ignore[reportArgumentType]
        seed: int | None = Input(
            description="Random seed. Leave unspecified to randomize the seed.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        stream: bool = Input(
            description="Request streaming response. Defaults to False.",
            default=False,
        ),  # pyright: ignore[reportArgumentType]
    ) -> AsyncConcatenateIterator[str]:  # type: ignore

        import importlib.metadata as metadata

        content = f"{self.device=} {torch.version.cuda=}"
        for dist in sorted(
            metadata.distributions(), key=lambda x: x.metadata["Name"].lower()
        ):
            content += f" {dist.metadata['Name']}=={dist.metadata['Version']}"

        yield content  # type: ignore

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
        predictor_config_path = weights.joinpath("predictor_config.json")
        if not predictor_config_path.exists():
            predictor_config_path = pathlib.Path("predictor_config.json")
            if not predictor_config_path.exists():
                predictor_config_path = None
        if predictor_config_path:
            logger.debug("Loading predictor_config.json path=%s", predictor_config_path)
            json_data = predictor_config_path.read_text(encoding="utf-8")
            config = PredictorConfig.model_validate_json(json_data)
        else:
            config = PredictorConfig()

        logger.debug("PredictorConfig config=%s", config)
        return config


# For testing
if __name__ == "__main__":

    async def main():
        """Async main method for direct testing."""
        predictor = Predictor()
        await predictor.setup("weights")

        if len(sys.argv) >= 2:
            file_paths = sys.argv[1:]
            defaults = predictor._defaults
            print()
            for path in file_paths:
                print(f"### Test file: {path}")
                json_str = pathlib.Path(path).read_text(encoding="utf-8")
                json_dict = json.loads(json_str)
                inputs = dict(defaults, **json_dict)
                generator = predictor.predict(**inputs)
                async for output in generator:  # type: ignore
                    if output.startswith("{") and output.endswith("}"):
                        try:
                            print(json.dumps(json.loads(output), indent=4))
                            continue
                        except json.JSONDecodeError:
                            pass
                    print(output, end="")
                print()

    asyncio.run(main())
