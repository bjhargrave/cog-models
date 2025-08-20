# SPDX-License-Identifier: Apache-2.0

# Prediction interface for Cog ⚙️
# https://cog.run/python

# pylint: disable=missing-module-docstring, missing-class-docstring, no-name-in-module, attribute-defined-outside-init

import inspect
import json
import logging
import os
import pathlib
import time
import typing
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import dataclass, field

import cog
import torch

from cog import AsyncConcatenateIterator, BasePredictor, Input, Path as CogPath
from cog.coder import json_coder  # pylint: disable=unused-import # noqa: F401
from openai.types.chat import ChatCompletionToolParam
from pydantic import ValidationError
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    envs,
)
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    CustomChatCompletionMessageParam,
    apply_hf_chat_template,
    load_chat_template,
    parse_chat_messages_futures,
    resolve_chat_template_content_format,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatCompletionResponse,
    ChatCompletionToolsParam,
    CompletionRequest,
    CompletionStreamResponse,
    CompletionResponse,
    ErrorResponse,
    StreamOptions,
    UsageInfo,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    OpenAIServingModels,
)
from vllm.utils import Counter


class UserError(Exception):
    pass


class ResponseError(Exception):
    pass


@dataclass
class PredictorConfig:
    """PredictorConfig is the configuration class for the Predictor."""

    chat_template: str | None = field(default=None)
    chat_template_content_format: ChatTemplateContentFormatOption = field(
        default="auto"
    )
    enable_log_requests: bool = field(default=False)
    max_log_len: int | None = field(default=None)
    enable_force_include_usage: bool = field(default=False)
    enable_auto_tool_choice: bool = field(default=False)
    tool_call_parser: str | None = field(default=None)
    reasoning_parser: str = field(default="")
    response_role: str = field(default="assistant")
    engine_args: dict[str, typing.Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.engine_args is None:
            self.engine_args = {}
        elif not isinstance(self.engine_args, dict):
            raise UserError(
                "Invalid predictor_config.json: engine_args must be "
                "a valid JSON object that maps to a dictionary."
            )


class ChatCompletionDocumentParam(typing.TypedDict, total=False):
    """
    Chat Completion API documents.

    We need to define our own type since list[dict] is not supported by cog-runtime.
    """

    text: str
    doc_id: str | None
    title: str | None


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


# pylint: disable=invalid-overridden-method, signature-differs, abstract-method
class Predictor(BasePredictor):
    async def setup(self, weights: CogPath) -> None:
        # Model weights must be in the "weights" folder.
        # This can be overridden with the COG_WEIGHTS env var.
        self.config = self.load_config(weights)
        logger.info("setup() commencing")

        self.resolved_chat_template = load_chat_template(self.config.chat_template)
        if self.resolved_chat_template:
            logger.debug("Using chat template from predictor_config.json")

        engine_args = self.config.engine_args
        if "model" not in engine_args:
            engine_args["model"] = weights.resolve().as_posix()
        if "dtype" not in engine_args:
            engine_args["dtype"] = "auto"
        if "tensor_parallel_size" not in engine_args:
            engine_args["tensor_parallel_size"] = max(torch.cuda.device_count(), 1)

        engine_args = AsyncEngineArgs(**engine_args)
        logger.debug("AsyncEngineArgs engine_args=%s", engine_args)

        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        except TypeError:
            logger.exception("Unexpected EngineArg")
            raise
        except Exception:
            logger.exception("VLLM Unknown Error")
            raise

        vllm_config: VllmConfig = await self.engine.get_vllm_config()
        model_config: ModelConfig = vllm_config.model_config

        if envs.VLLM_USE_V1:
            supported_tasks = await self.engine.get_supported_tasks()  # type: ignore
        else:
            supported_tasks = model_config.supported_tasks
        logger.debug("Supported_tasks: %s", supported_tasks)

        request_logger = (
            RequestLogger(max_log_len=self.config.max_log_len)
            if self.config.enable_log_requests
            else None
        )

        default_mm_loras = (
            vllm_config.lora_config.default_mm_loras
            if vllm_config.lora_config is not None
            else None
        )
        lora_modules = (
            [
                LoRAModulePath(
                    name=modality,
                    path=lora_path,
                )
                for modality, lora_path in default_mm_loras.items()
            ]
            if default_mm_loras
            else None
        )
        served_model_names = (
            model_config.served_model_name
            if isinstance(model_config.served_model_name, list)
            else [model_config.served_model_name]
            if model_config.served_model_name
            else [model_config.model]
        )
        base_model_paths = [
            BaseModelPath(name=name, model_path=model_config.model)
            for name in served_model_names
        ]
        self.serving_models = OpenAIServingModels(
            engine_client=self.engine,
            model_config=model_config,
            base_model_paths=base_model_paths,
            lora_modules=lora_modules,
        )
        await self.serving_models.init_static_loras()
        self.serving_chat = (
            OpenAIServingChat(
                self.engine,
                model_config,
                self.serving_models,
                self.config.response_role,
                request_logger=request_logger,
                chat_template=self.resolved_chat_template,
                chat_template_content_format=self.config.chat_template_content_format,
                enable_force_include_usage=self.config.enable_force_include_usage,
                enable_auto_tools=self.config.enable_auto_tool_choice,
                tool_parser=self.config.tool_call_parser,
                reasoning_parser=self.config.reasoning_parser,
            )
            if "generate" in supported_tasks
            else None
        )
        self.serving_completion = (
            OpenAIServingCompletion(
                self.engine,
                model_config,
                self.serving_models,
                request_logger=request_logger,
                enable_force_include_usage=self.config.enable_force_include_usage,
            )
            if "generate" in supported_tasks
            else None
        )

        self.request_counter = Counter()

        self._testing = True
        generator = self.predict(
            **dict(self._defaults, max_tokens=50, prompt="What is your name?")
        )
        test_output = "".join([tok async for tok in generator])  # type: ignore
        self._testing = False
        logger.debug("Test prediction output test_output=%s", test_output)
        logger.info("setup() complete")

    def get_model_name(self) -> str:
        """Return the default model name."""
        return self.serving_models.base_model_paths[0].name

    async def predict(  # pylint: disable=invalid-overridden-method, arguments-differ, too-many-arguments, too-many-positional-arguments, too-many-locals
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
            description="Documents for request.",
            default=[],
        ),  # pyright: ignore[reportArgumentType]
        tools: list[ChatCompletionToolParam] = Input(
            description="Tools for request.",
            default=[],
        ),  # pyright: ignore[reportArgumentType]
        system_prompt: str | None = Input(
            description="Completion API system prompt. "
            "The chat template provides a good default.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        chat_template: str | None = Input(
            description="A template to format the prompt with. If not provided, "
            "the default prompt template will be used.",
            default=None,
        ),  # pyright: ignore[reportArgumentType]
        add_generation_prompt: bool = Input(
            description="Add generation prompt. Passed to chat template. Defaults to True.",
            default=True,
        ),  # pyright: ignore[reportArgumentType]
        thinking: bool = Input(
            description="Whether to enable thinking. Passed to chat template. Defaults to False.",
            default=False,
        ),  # pyright: ignore[reportArgumentType]
        min_tokens: int = Input(
            description="The minimum number of tokens the model should generate as output.",
            default=0,
        ),  # pyright: ignore[reportArgumentType]
        max_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output.",
            default=512,
        ),  # pyright: ignore[reportArgumentType]
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=0.6,
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
        presence_penalty: float = Input(description="Presence penalty", default=0.0),  # pyright: ignore[reportArgumentType]
        frequency_penalty: float = Input(description="Frequency penalty", default=0.0),  # pyright: ignore[reportArgumentType]
        stop_sequences: str | None = Input(
            description="A comma-separated list of sequences to stop generation at. "
            "For example, '<end>,<stop>' will stop generation at the first instance of "
            "'<end>' or '<stop>'.",
            default=None,
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
        start_time = time.time()
        request_id = str(next(self.request_counter))
        logger.info("predict() commencing request_id=%s", request_id)

        top_k = -1 if (top_k or 0) == 0 else top_k
        stop = stop_sequences.split(",") if stop_sequences else []
        stream_options = StreamOptions() if stream else None

        usage: UsageInfo | None = None
        finish_reason: str | None = None
        responses: list[str] = []
        responses_joiner: Callable[[list[str]], str]

        if messages:  # Chat completion API
            if prompt or system_prompt:
                logger.warning(
                    "Mutually exclusive messages and prompt/system prompt are specified. "
                    "Only messages will be used."
                )

            request = ChatCompletionRequest(
                model=self.get_model_name(),
                messages=messages,  # pyright: ignore[reportArgumentType]
                tools=[ChatCompletionToolsParam.model_validate(tool) for tool in tools]
                if tools
                else None,
                documents=documents or None,  # pyright: ignore[reportArgumentType]
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs={
                    "thinking": thinking,
                },
                n=1,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                min_tokens=min_tokens,
                max_completion_tokens=max_tokens,
                stop=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                stream=stream,
                stream_options=stream_options,
                request_id=request_id,
            )

            responses_joiner = json.dumps

            generator = await self.create_chat_completion(request)
            match generator:
                case ChatCompletionResponse():
                    usage = generator.usage
                    for choice in generator.choices:
                        finish_reason = choice.finish_reason
                        message_text = choice.message.model_dump_json(
                            exclude_unset=True, exclude_none=True
                        )
                        responses.append(message_text)
                        yield message_text  # type: ignore
                case AsyncGenerator():
                    async for response in generator:
                        usage = response.usage
                        for choice in response.choices:
                            finish_reason = choice.finish_reason
                            delta_text = choice.delta.model_dump_json(
                                exclude_unset=True, exclude_none=True
                            )
                            responses.append(delta_text)
                            yield delta_text  # type: ignore
                case ErrorResponse():
                    logger.error("%r", generator)
                    raise ResponseError(generator.model_dump_json())

        elif prompt or system_prompt:  # Completion API
            if (
                not system_prompt
                and prompt
                and prompt.lstrip().startswith("<|start_of_role|>")
            ):
                request_prompt = prompt
            else:
                conversation = []
                if system_prompt:
                    conversation.append({"role": "system", "content": system_prompt})
                conversation.append({"role": "user", "content": prompt})

                tokenizer = await self.engine.get_tokenizer()
                model_config = self.serving_models.model_config
                resolved_content_format = resolve_chat_template_content_format(
                    chat_template=chat_template or self.resolved_chat_template,
                    tools=tools or None,  # pyright: ignore[reportArgumentType]
                    given_format=self.config.chat_template_content_format,
                    tokenizer=tokenizer,  # pyright: ignore[reportArgumentType]
                    model_config=model_config,
                )
                conversation, mm_data_future = parse_chat_messages_futures(
                    messages=conversation,
                    model_config=model_config,
                    tokenizer=tokenizer,
                    content_format=resolved_content_format,
                )
                request_prompt = apply_hf_chat_template(
                    tokenizer=tokenizer,  # pyright: ignore[reportArgumentType]
                    conversation=conversation,
                    chat_template=chat_template or self.resolved_chat_template,
                    tools=tools or None,  # pyright: ignore[reportArgumentType]
                    model_config=model_config,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    documents=documents or None,
                    thinking=thinking,
                )
                mm_data = await mm_data_future
                # CompletionRequest does not support multimodal data
                if mm_data:
                    logger.debug("mm_data %r", mm_data)

            request = CompletionRequest(
                model=self.get_model_name(),
                prompt=request_prompt,
                n=1,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                stop=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                stream=stream,
                stream_options=stream_options,
                request_id=request_id,
            )

            responses_joiner = "".join

            generator = await self.create_completion(request)
            match generator:
                case CompletionResponse():
                    usage = generator.usage
                    for choice in generator.choices:
                        finish_reason = choice.finish_reason
                        choice_text = choice.text
                        if choice_text:
                            responses.append(choice_text)
                            yield choice_text  # type: ignore
                case AsyncGenerator():
                    async for response in generator:
                        usage = response.usage
                        for choice in response.choices:
                            finish_reason = choice.finish_reason
                            choice_text = choice.text
                            if choice_text:
                                responses.append(choice_text)
                                yield choice_text  # type: ignore
                case ErrorResponse():
                    logger.error("%r", generator)
                    raise ResponseError(generator.model_dump_json())

        else:  # Error
            error_message = "No messages or prompt inputs specified"
            logger.error("%s", error_message)
            raise ResponseError(error_message)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "response_text=%s finish_reason=%s",
                responses_joiner(responses),
                finish_reason,
            )
        logger.info("Generation took %.2fs", time.time() - start_time)

        if usage:
            logger.debug(
                "prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
            if not self._testing:
                current_scope = cog.current_scope()
                for name, value in usage.model_dump(
                    exclude_unset=True, exclude_none=True
                ).items():
                    current_scope.record_metric(name, value)

        logger.info("predict() complete")

    def completion(self) -> OpenAIServingCompletion:
        """Return completion handler"""
        handler = self.serving_completion
        assert handler is not None, (
            f"Completion API not supported by model {self.get_model_name()}"
        )
        return handler

    async def create_completion(
        self, request: CompletionRequest
    ) -> (
        AsyncGenerator[CompletionStreamResponse, None]
        | CompletionResponse
        | ErrorResponse
    ):
        """Create completion response generator"""
        handler = self.completion()
        generator = await handler.create_completion(request, None)

        match generator:
            case ErrorResponse() | CompletionResponse():
                return generator
            case AsyncGenerator():
                return self.completion_stream_generator(generator)

    async def completion_stream_generator(
        self, response_generator: AsyncIterator[str]
    ) -> AsyncGenerator[CompletionStreamResponse, None]:
        """Create a generator for streaming completion responses"""
        async for response_str in response_generator:
            if response_str.startswith("data: "):
                data_str = response_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    # data_str could represent CompletionResponse or CompletionStreamResponse
                    # But they are similar enough to use CompletionStreamResponse
                    yield CompletionStreamResponse.model_validate_json(data_str)
                except ValidationError:  # It could be an ErrorResponse
                    raise ResponseError(data_str)  # pylint: disable=raise-missing-from

    def chat(self) -> OpenAIServingChat:
        """Return chat completion handler"""
        handler = self.serving_chat
        assert handler is not None, (
            f"Chat Completion API not supported by model {self.get_model_name()}"
        )
        return handler

    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> (
        AsyncGenerator[ChatCompletionStreamResponse, None]
        | ChatCompletionResponse
        | ErrorResponse
    ):
        """Create chat completion response generator"""
        handler = self.chat()
        generator = await handler.create_chat_completion(request, None)

        match generator:
            case ErrorResponse() | ChatCompletionResponse():
                return generator
            case AsyncGenerator():
                return self.chat_completion_stream_generator(generator)

    async def chat_completion_stream_generator(
        self, response_generator: AsyncIterator[str]
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """Create a generator for streaming chat completion responses"""
        async for response_str in response_generator:
            if response_str.startswith("data: "):
                data_str = response_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    yield ChatCompletionStreamResponse.model_validate_json(data_str)
                except ValidationError:  # It could be an ErrorResponse
                    raise ResponseError(data_str)  # pylint: disable=raise-missing-from

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
            try:
                logger.debug(
                    "Loading predictor_config.json path=%s", predictor_config_path
                )
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

        logger.debug("PredictorConfig config=%s", config)
        return config
