# SPDX-License-Identifier: Apache-2.0

# Prediction interface for Cog ⚙️
# https://cog.run/python

# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring, no-name-in-module, attribute-defined-outside-init

import inspect
import json
import pathlib
import time
import typing

from collections import abc
from dataclasses import dataclass, field
from uuid import uuid4

import cog
import structlog
import torch

from cog import BasePredictor, Input, Path as CogPath
from pydantic import json_schema, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import core_schema
from structlog.contextvars import bind_contextvars, clear_contextvars
from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    EmbeddingRequestOutput,
    PoolingParams,
)


class UserError(Exception):
    pass


@dataclass
class PredictorConfig:
    """
    PredictorConfig is a configuration class for the Predictor.

    Attributes:
        engine_args (dict[str, str] | None): A dictionary of engine arguments. If not provided,
                                      an empty dictionary will be used.
    """

    engine_args: dict[str, typing.Any] | None = field(default_factory=dict)

    def __post_init__(self):
        if self.engine_args is None:
            self.engine_args = {}
        elif not isinstance(self.engine_args, dict):
            raise UserError(
                "Invalid predictor_config.json: engine_args must be "
                "a valid JSON object that maps to a dictionary."
            )


type Embedding = list[float]


class AsyncIterator[T: Embedding](abc.AsyncIterator[T]):  # pylint: disable=abstract-method, too-few-public-methods
    """Subclass to enable JSON schema generation by Pydantic"""

    @classmethod
    def validate(cls, value: abc.AsyncIterator[T]) -> abc.AsyncIterator[T]:
        return value

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: typing.Type[typing.Any],  # pylint: disable=unused-argument
        handler: GetCoreSchemaHandler,  # pylint: disable=unused-argument
    ) -> core_schema.CoreSchema:
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(abc.AsyncIterator),
                core_schema.no_info_plain_validator_function(cls.validate),
            ]
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: core_schema.CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> json_schema.JsonSchemaValue:
        json_schema_value = handler(schema)
        json_schema_value.update(
            {
                "items": {
                    # This assumes T is Embedding
                    "items": {"type": "number"},
                    "type": "array",
                },
                "type": "array",
                "x-cog-array-type": "iterator",
            }
        )
        return json_schema_value


# pylint: disable=invalid-overridden-method, signature-differs
class Predictor(BasePredictor):
    logger = structlog.get_logger(__name__)

    async def setup(self, weights: CogPath) -> None:
        # Model weights must be in the "weights" folder.
        # This can be overridden with the COG_WEIGHTS env var.
        clear_contextvars()
        self.config = self.load_config(weights)
        log = self.logger.bind()
        log.info("setup() commencing")

        engine_args = self.config.engine_args or {}
        engine_args["model"] = weights.resolve().as_posix()
        if "dtype" not in engine_args:
            engine_args["dtype"] = "auto"
        if "tensor_parallel_size" not in engine_args:
            engine_args["tensor_parallel_size"] = max(torch.cuda.device_count(), 1)

        engine_args = AsyncEngineArgs(**engine_args)
        log.debug("AsyncEngineArgs", engine_args=engine_args)

        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        except TypeError as e:
            log.error("Unexpected EngineArg", exc_info=e)
            raise
        except Exception as e:
            log.error("VLLM Unknown Error", exc_info=e)
            raise

        self._testing = True
        generator = self.predict(
            **dict(
                self._defaults,
                **{"texts": ["Dogs are mammals.", "Fish are not mammals."]},
            )
        )
        embedding_lengths = [len(embedding) async for embedding in generator]  # type: ignore
        self._testing = False
        clear_contextvars()
        log.debug(
            "Test prediction output",
            embedding_count=len(embedding_lengths),
            embedding_lengths=embedding_lengths,
        )
        log.info("setup() complete")

    async def predict(  # pylint: disable=invalid-overridden-method, arguments-differ, too-many-locals
        self,
        *,
        # texts must be the first argument
        # The LangChain Replicate class will use the first argument to supply the input
        texts: list[str] = Input(  # pylint: disable=redefined-builtin
            description="A list of texts to embed.",
            default=[],
        ),
    ) -> AsyncIterator[Embedding]:  # type: ignore
        start_time = time.time()
        clear_contextvars()
        request_id = uuid4().hex
        bind_contextvars(input_count=len(texts), request_id=request_id)
        log = self.logger.bind()
        log.info("predict() commencing")

        pooling_params = PoolingParams()
        log.debug("PoolingParams", pooling_params=pooling_params)

        generators = [
            self.engine.encode(
                text,
                pooling_params=pooling_params,
                request_id=f"{request_id}/{i}",
            )
            for i, text in enumerate(texts)
        ]
        input_token_count: int = 0
        output_token_count: int = 0
        for i, generator in enumerate(generators):
            async for result in generator:
                request_output = EmbeddingRequestOutput.from_base(result)
            embedding: Embedding = request_output.outputs.embedding
            embedding_length = len(embedding)
            input_tokens = len(request_output.prompt_token_ids)
            log.debug(
                "Embedding result",
                text=texts[i],
                input_tokens=input_tokens,
                embedding_length=embedding_length,
                request_id=request_output.request_id,
                finished=request_output.finished,
            )
            input_token_count += input_tokens
            output_token_count += embedding_length
            yield embedding  # type: ignore

        log.info(f"Embedding took {time.time() - start_time:.2f}s")

        if not self._testing:
            cog.current_scope().record_metric("input_token_count", input_token_count)
            cog.current_scope().record_metric("output_token_count", output_token_count)

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
