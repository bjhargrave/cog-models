# SPDX-License-Identifier: Apache-2.0

# Prediction interface for Cog ⚙️
# https://cog.run/python

# pylint: disable=missing-module-docstring, missing-class-docstring, no-name-in-module, attribute-defined-outside-init
# mypy: disable-error-code="import-untyped"

import asyncio
import inspect
import json
import logging
import os
import pathlib
import sys
import time
import typing
from collections.abc import AsyncGenerator, AsyncIterator

import torch
from cog import AsyncConcatenateIterator, BasePredictor, Input
from cog import Path as CogPath
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import (
    AnyResponseFormat,
    ErrorResponse,
    StreamOptions,
    UsageInfo,
)
from vllm.entrypoints.openai.models.protocol import (
    BaseModelPath,
    LoRAModulePath,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
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
    """PredictorConfig is the configuration class for the Predictor."""

    model_config = ConfigDict(extra="allow")

    chat_template: str | None = Field(default=None)
    chat_template_content_format: ChatTemplateContentFormatOption = Field(
        default="auto"
    )
    enable_log_requests: bool = Field(default=False)
    max_log_len: int | None = Field(default=None)
    enable_force_include_usage: bool = Field(default=False)
    enable_auto_tool_choice: bool = Field(default=False)
    tool_call_parser: str | None = Field(default=None)
    reasoning_parser: str = Field(default="")
    response_role: str = Field(default="assistant")
    log_error_stack: bool = Field(default=envs.VLLM_SERVER_DEV_MODE)
    trust_request_chat_template: bool = Field(default=False)
    default_chat_template_kwargs: dict[str, typing.Any] = Field(default_factory=dict)
    engine_args: dict[str, typing.Any] = Field(default_factory=dict)


class ChatCompletionDocumentParam(typing.TypedDict, total=False):
    """
    Chat Completion API documents.
    """

    text: str
    doc_id: str | int | None
    title: str | None


def process_documents(
    documents: list[ChatCompletionDocumentParam],
) -> list[dict[str, str]] | None:
    """Convert all document values to str."""
    if documents:
        return [{k: str(v) for k, v in document.items()} for document in documents]
    return None


def process_tool_choice(
    tool_choice: str | None,
) -> (
    ChatCompletionNamedToolChoiceParam
    | typing.Literal["none", "auto", "required"]
    | None
):
    """Convert string tool_choice value to the desired usable value."""
    match tool_choice:
        case "none" | "auto" | "required" | None:
            return tool_choice
        case _:
            try:
                return json.loads(tool_choice)
            except json.JSONDecodeError:
                logger.exception("Invalid tool_choice value")
                return None


class JsonSchemaResponseFormat(typing.TypedDict, total=False):
    name: typing.Required[str]
    description: str | None
    schema: dict[str, typing.Any] | None


class ResponseFormat(typing.TypedDict, total=False):
    type: typing.Required[typing.Literal["text", "json_object", "json_schema"]]
    json_schema: JsonSchemaResponseFormat | None


# pylint: disable=invalid-overridden-method, signature-differs, abstract-method, too-many-instance-attributes, arguments-differ
class Predictor(BasePredictor):
    async def setup(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, weights: CogPath | str | None
    ) -> None:
        # Model weights must be in the "weights" folder.
        # This can be overridden with the COG_WEIGHTS env var.
        if not weights:
            weights = CogPath("/src/weights")  # default location
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

        vllm_config: VllmConfig = self.engine.vllm_config
        model_config: ModelConfig = vllm_config.model_config

        supported_tasks = await self.engine.get_supported_tasks()
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
            base_model_paths=base_model_paths,
            lora_modules=lora_modules,
        )
        await self.serving_models.init_static_loras()

        self.serving_render = OpenAIServingRender(
            model_config=self.engine.model_config,
            renderer=self.engine.renderer,
            io_processor=self.engine.io_processor,
            model_registry=self.serving_models.registry,
            request_logger=request_logger,
            chat_template=self.resolved_chat_template,
            chat_template_content_format=self.config.chat_template_content_format,
            trust_request_chat_template=self.config.trust_request_chat_template,
            enable_auto_tools=self.config.enable_auto_tool_choice,
            tool_parser=self.config.tool_call_parser,
            default_chat_template_kwargs=self.config.default_chat_template_kwargs,
            log_error_stack=self.config.log_error_stack,
        )

        self.serving_chat = (
            OpenAIServingChat(
                engine_client=self.engine,
                models=self.serving_models,
                response_role=self.config.response_role,
                openai_serving_render=self.serving_render,
                request_logger=request_logger,
                chat_template=self.resolved_chat_template,
                chat_template_content_format=self.config.chat_template_content_format,
                trust_request_chat_template=self.config.trust_request_chat_template,
                enable_force_include_usage=self.config.enable_force_include_usage,
                enable_auto_tools=self.config.enable_auto_tool_choice,
                tool_parser=self.config.tool_call_parser,
                reasoning_parser=self.config.reasoning_parser,
                default_chat_template_kwargs=self.config.default_chat_template_kwargs,
            )
            if "generate" in supported_tasks
            else None
        )
        if self.serving_chat is not None:
            self.serving_chat.warmup()

        self.serving_completion = (
            OpenAIServingCompletion(
                engine_client=self.engine,
                models=self.serving_models,
                openai_serving_render=self.serving_render,
                request_logger=request_logger,
                enable_force_include_usage=self.config.enable_force_include_usage,
            )
            if "generate" in supported_tasks
            else None
        )

        self.request_counter = Counter(1)

        generator = self.predict(
            **dict(
                self._defaults, max_completion_tokens=50, prompt="What is your name?"
            )
        )
        test_output = "".join([tok async for tok in generator])  # type: ignore
        logger.debug("Test prediction output test_output=%s", test_output)

        logger.info("setup() complete")

    async def predict(  # pyright: ignore[reportIncompatibleMethodOverride]
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
            description="An upper bound for the number of tokens that can be generated for a "
            "completion, including visible output tokens and reasoning tokens.",
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
        start_time = time.time()
        request_id = str(next(self.request_counter))
        logger.info("predict() commencing request_id=%s", request_id)

        top_k = -1 if (top_k or 0) == 0 else top_k
        stream_options = StreamOptions() if stream else None
        if max_completion_tokens is None:
            max_completion_tokens = max_tokens

        chat_completion = bool(messages)
        if prompt or system_prompt:
            if chat_completion:
                logger.warning(
                    "Mutually exclusive messages and prompt/system prompt are specified. "
                    "Only messages will be used."
                )
            else:
                messages = []  # new list
                if system_prompt:
                    messages.append(
                        CustomChatCompletionMessageParam(
                            role="system", content=system_prompt
                        )
                    )
                if prompt and (
                    system_prompt or not prompt.lstrip().startswith("<|start_of_role|>")
                ):
                    messages.append(
                        CustomChatCompletionMessageParam(role="user", content=prompt)
                    )
        elif not chat_completion:
            error_message = "No messages or prompt inputs specified"
            logger.error("%s", error_message)
            raise ResponseError(error_message)

        usage: UsageInfo | None = None
        finish_reason: str | None = None

        async def create_chat_completion_response() -> AsyncGenerator[str, None]:
            nonlocal usage, finish_reason
            request = ChatCompletionRequest(
                model=self.serving_models.model_name(),
                messages=messages,  # type: ignore[arg-type]
                tools=tools or None,  # type: ignore[arg-type]
                tool_choice=process_tool_choice(tool_choice),
                documents=process_documents(documents),
                response_format=typing.cast(AnyResponseFormat, response_format),
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                chat_template_kwargs=chat_template_kwargs,
                n=1,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                min_tokens=min_tokens,
                max_completion_tokens=max_completion_tokens,
                stop=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stream=stream,
                stream_options=stream_options,
                request_id=request_id,
            )

            generator = await self.create_chat_completion(request)
            match generator:
                case ChatCompletionResponse():

                    async def chat_completion_response() -> AsyncGenerator[str, None]:
                        nonlocal usage, finish_reason
                        usage = generator.usage
                        assert len(generator.choices) == 1, (
                            "Expected exactly one output from generation request."
                        )
                        choice = generator.choices[0]
                        finish_reason = choice.finish_reason
                        response_text = (
                            generator.model_dump_json()
                            if chat_completion
                            else choice.message.content
                        )
                        if response_text:
                            yield response_text

                    return chat_completion_response()
                case AsyncGenerator():

                    async def chat_completion_stream_response() -> AsyncGenerator[
                        str, None
                    ]:
                        nonlocal usage, finish_reason
                        async for response in generator:
                            if not finish_reason:
                                if response.usage:
                                    usage = response.usage
                                assert len(response.choices) == 1, (
                                    "Expected exactly one output from generation request."
                                )
                                choice = response.choices[0]
                                if choice.finish_reason:
                                    finish_reason = choice.finish_reason
                                response_text = (
                                    response.model_dump_json()
                                    if chat_completion
                                    else choice.delta.content
                                )
                                if response_text:
                                    yield response_text

                    return chat_completion_stream_response()
                case ErrorResponse():
                    logger.error("%r", generator)
                    raise ResponseError(generator.model_dump_json())

        async def create_completion_response() -> AsyncGenerator[str, None]:
            nonlocal usage, finish_reason
            max_tokens = (
                max_completion_tokens if max_completion_tokens is not None else 512
            )
            request = CompletionRequest(
                model=self.serving_models.model_name(),
                prompt=prompt,
                response_format=typing.cast(AnyResponseFormat, response_format),
                n=1,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                stop=stop,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stream=stream,
                stream_options=stream_options,
                request_id=request_id,
            )

            generator = await self.create_completion(request)
            match generator:
                case CompletionResponse():

                    async def completion_response() -> AsyncGenerator[str, None]:
                        nonlocal usage, finish_reason
                        usage = generator.usage
                        assert len(generator.choices) == 1, (
                            "Expected exactly one output from generation request."
                        )
                        choice = generator.choices[0]
                        finish_reason = choice.finish_reason
                        choice_text = choice.text
                        if choice_text:
                            yield choice_text

                    return completion_response()
                case AsyncGenerator():

                    async def completion_stream_response() -> AsyncGenerator[str, None]:
                        nonlocal usage, finish_reason
                        async for response in generator:
                            if not finish_reason:
                                if response.usage:
                                    usage = response.usage
                                assert len(response.choices) == 1, (
                                    "Expected exactly one output from generation request."
                                )
                                choice = response.choices[0]
                                if choice.finish_reason:
                                    finish_reason = choice.finish_reason
                                choice_text = choice.text
                                if choice_text:
                                    yield choice_text

                    return completion_stream_response()
                case ErrorResponse():
                    logger.error("%r", generator)
                    raise ResponseError(generator.model_dump_json())

        responses: list[str] = []
        response = await (
            create_chat_completion_response()
            if messages
            else create_completion_response()
        )
        async for response_text in response:
            if response_text:
                responses.append(response_text)
                yield response_text  # type: ignore

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "finish_reason=%s response_text=%s",
                finish_reason,
                json.dumps(responses) if chat_completion else "".join(responses),
            )

        logger.info("Generation took %.2fs", time.time() - start_time)

        if usage:
            logger.debug(
                "prompt_tokens=%s, completion_tokens=%s, total_tokens=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )
            self.record_metric("token_input_count", usage.prompt_tokens)
            self.record_metric("token_output_count", usage.completion_tokens)

        logger.info("predict() completed request_id=%s", request_id)

    def completion(self) -> OpenAIServingCompletion:
        """Return completion handler"""
        handler = self.serving_completion
        assert handler is not None, (
            f"generate task is not supported by model {self.serving_models.model_name()}"
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
                    response = CompletionStreamResponse.model_validate_json(data_str)
                    yield response
                except ValidationError:  # It could be an ErrorResponse
                    raise ResponseError(data_str)  # pylint: disable=raise-missing-from

    def chat(self) -> OpenAIServingChat:
        """Return chat completion handler"""
        handler = self.serving_chat
        assert handler is not None, (
            f"generate task is not supported by model {self.serving_models.model_name()}"
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
                    response = ChatCompletionStreamResponse.model_validate_json(
                        data_str
                    )
                    yield response
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
        if predictor_config_path.exists():
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
            defaults = predictor._defaults  # pylint: disable=protected-access
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
