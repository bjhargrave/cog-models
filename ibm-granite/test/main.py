import logging
import pathlib
import os
from typing import Any, AsyncIterator

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

# Set before importing any of vLLM's packages
os.environ.setdefault("VLLM_LOGGING_STREAM", "ext://sys.stderr")

from vllm import (
    AsyncEngineArgs,
    AsyncLLMEngine,
    envs,
)
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    load_chat_template,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.utils.counter import Counter

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
    engine_args: dict[str, Any] = Field(default_factory=dict)


class Test:
    def test(self):
        #weights = pathlib.Path("../granite-4.0-h-small/weights")
        weights = pathlib.Path("weights")
        self.config = self.load_config(weights)
        logger.info("setup() commencing")
        torch._dynamo.config.dynamic_shapes = False

        engine_args = AsyncEngineArgs(**self.config.engine_args)
        if "model" not in self.config.engine_args:
            engine_args.model = weights.resolve().as_posix()
        if "tensor_parallel_size" not in self.config.engine_args:
            engine_args.tensor_parallel_size = max(torch.cuda.device_count(), 1)

        logger.debug("AsyncEngineArgs engine_args=%s", engine_args)

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        logger.debug("Engine created")



    def load_config(self, weights: pathlib.Path) -> PredictorConfig:
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

if __name__ == "__main__":
    Test().test()
