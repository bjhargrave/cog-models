# SPDX-License-Identifier: Apache-2.0

# Prediction interface for Cog ⚙️
# https://cog.run/python

# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring, no-name-in-module, attribute-defined-outside-init

import asyncio
import inspect
import json
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Coroutine

import torch
from cog import BasePredictor, Input
from cog import Path as CogPath
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from transformers import AutoModel, AutoTokenizer


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


@dataclass
class Request[T, R]:
    """
    A request wrapper that holds the data and a future for the result.

    Attributes:
        data: The request data to be processed
        result_future: Future that will hold the processing result
    """

    data: T
    request_id: str
    result_future: asyncio.Future[R]


class ProducerConsumer[T, R]:
    """
    Producer-Consumer pattern implementation using asyncio.

    This class manages a queue of requests and a background consumer task
    that processes them. Producers can submit requests and wait for results.

    Example:
        async def process_request(data: int) -> int:
            await asyncio.sleep(0.1)  # Simulate work
            return data * 2

        pc = ProducerConsumer(process_request, max_queue_size=100)
        await pc.start()

        # Producer submits and waits for result
        result = await pc.submit(42)
        print(f"Result: {result}")  # Output: Result: 84

        await pc.stop()
    """

    def __init__(
        self,
        processor: Callable[[T], Coroutine[Any, Any, R]],
        max_queue_size: int = 0,
        name: str = "ProducerConsumer",
    ) -> None:
        """
        Initialize the producer-consumer.

        Args:
            processor: Async function that processes each request
            max_queue_size: Maximum queue size (0 = unlimited)
            name: Name for logging purposes
        """
        self.processor = processor
        self.name = name
        self.queue: asyncio.Queue[Request[T, R]] = asyncio.Queue(maxsize=max_queue_size)
        self._consumer_task: asyncio.Task | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """
        Start the background consumer loop.

        Raises:
            RuntimeError: If already started
        """
        if self._running:
            raise RuntimeError(f"{self.name} is already running")

        self._running = True
        self._shutdown_event.clear()
        self._consumer_task = asyncio.create_task(
            self._consumer_loop(), name=f"{self.name}-consumer"
        )
        logger.debug("%s started", self.name)

    async def stop(self, timeout: float | None = None) -> None:
        """
        Stop the background consumer loop gracefully.

        Waits for all pending requests to be processed before stopping.

        Args:
            timeout: Maximum time to wait for shutdown (None = wait forever)

        Raises:
            asyncio.TimeoutError: If shutdown times out
        """
        if not self._running:
            logger.debug("%s is not running", self.name)
            return

        logger.debug("%s stopping...", self.name)
        self._running = False

        # Wait for queue to be empty
        await self.queue.join()

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for consumer task to finish
        if self._consumer_task:
            try:
                await asyncio.wait_for(self._consumer_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.error("%s shutdown timed out, cancelling task", self.name)
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass

        logger.debug("%s stopped", self.name)

    async def submit(self, data: T, request_id: str) -> R:
        """
        Submit a request and wait for the result.

        This method blocks until the consumer processes the request
        and returns the result.

        Args:
            data: The request data to process

        Returns:
            The processing result

        Raises:
            RuntimeError: If not started
            Exception: Any exception raised during processing
        """
        if not self._running:
            raise RuntimeError(f"{self.name} is not running")

        # Create a future for the result
        result_future: asyncio.Future[R] = asyncio.Future()

        # Create and enqueue the request
        request = Request(request_id=request_id, data=data, result_future=result_future)
        await self.queue.put(request)

        # Return the result
        return await result_future

    async def _consumer_loop(self) -> None:
        """
        Background consumer loop that processes requests from the queue.

        This runs continuously until shutdown is signaled and the queue is empty.
        """
        logger.debug("%s consumer loop started", self.name)

        try:
            while self._running or not self.queue.empty():
                try:
                    # Wait for a request with timeout to check shutdown
                    request = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Check if we should shutdown
                    if self._shutdown_event.is_set() and self.queue.empty():
                        break
                    continue

                # Process the request
                try:
                    logger.debug(
                        "%s processing request: %s", self.name, request.request_id
                    )
                    result = await self.processor(request.data)
                    request.result_future.set_result(result)
                    logger.debug(
                        "%s completed request: %s", self.name, request.request_id
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "%s error processing request: %s", self.name, e, exc_info=True
                    )
                    request.result_future.set_exception(e)
                finally:
                    self.queue.task_done()

        except asyncio.CancelledError:
            logger.info("%s consumer loop cancelled", self.name)
            raise
        except Exception as e:
            logger.error("%s consumer loop error: %s", self.name, e, exc_info=True)
            raise
        finally:
            logger.info("%s consumer loop stopped", self.name)

    @property
    def is_running(self) -> bool:
        """Check if the consumer is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get the current queue size."""
        return self.queue.qsize()


class PredictorConfig(BaseModel):
    """
    PredictorConfig is a configuration class for the Predictor.
    """

    model_config = ConfigDict(extra="allow")

    enable_log_requests: bool = Field(default=False)
    served_model_name: str | None = Field(default=None)
    pretrained_model_name_or_path: str | None = Field(default=None)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.counter = start
        self._lock = asyncio.Lock()

    async def next(self) -> int:
        async with self._lock:
            i = self.counter
            self.counter += 1
            return i

    async def reset(self, start: int = 0) -> None:
        async with self._lock:
            self.counter = start


type Embedding = list[float]


@dataclass
class EmbeddingRequest:
    inputs: list[str]
    normalize: bool


@dataclass
class EmbeddingResponse:
    outputs: list[Embedding]


# pylint: disable=invalid-overridden-method, signature-differs, abstract-method
class Predictor(BasePredictor):
    async def get_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for a list of strings using a Hugging Face embedding model.

        Args:
            texts: List of strings to embed

        Returns:
            List of embeddings, one per input string
        """
        # tokenize inputs
        tokenized_queries = self.tokenizer(
            request.inputs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)

        # encode queries
        with torch.no_grad():
            # Queries
            model_output = self.model(**tokenized_queries)
            # Perform pooling
            query_embeddings = model_output[0][:, 0]

        if request.normalize:
            # normalize the embeddings
            query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)

        return EmbeddingResponse(outputs=query_embeddings.cpu().tolist())

    async def setup(self, weights: CogPath | str | None) -> None:
        # Model weights must be in the "weights" folder.
        # This can be overridden with the COG_WEIGHTS env var.
        # self.config = self.load_config(weights)
        if not weights:
            weights = CogPath("/src/weights")  # default location
        elif isinstance(weights, str):
            weights = CogPath(weights)
        self.config = self.load_config(weights)
        logger.info("setup() commencing")

        model_path = self.config.pretrained_model_name_or_path or weights.resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, **self.config.model_kwargs)
        logger.debug(
            "Using model %s on device %s",
            self.config.served_model_name,
            self.model.device,
        )
        self.model.eval()

        self.request_counter = Counter(1)

        self.queue_worker = ProducerConsumer(
            self.get_embeddings, max_queue_size=10, name="EmbeddingWorker"
        )
        await self.queue_worker.start()

        self._testing = True
        output = await self.predict(
            **dict(
                self._defaults,
                **{"texts": ["Dogs are mammals.", "Fish are not mammals."]},
            )
        )
        embedding_lengths = [len(embedding) for embedding in output]  # type: ignore
        self._testing = False
        logger.debug(
            "Test prediction output embedding_count=%s, embedding_lengths=%s",
            len(embedding_lengths),
            embedding_lengths,
        )
        logger.info("setup() complete")

    async def stop(self, timeout: float | None = None) -> None:
        await self.queue_worker.stop(timeout)

    async def predict(  # pylint: disable=invalid-overridden-method, arguments-differ, too-many-arguments, too-many-positional-arguments, too-many-locals
        self,
        # texts must be the first argument
        # The LangChain Replicate class will use the first argument to supply the input
        texts: list[str] = Input(
            description="A list of text strings to embed.",
            default=[],
        ),  # pyright: ignore[reportArgumentType]
        normalize: bool = Input(
            description="Normalize the embeddings.",
            default=True,
        ),  # pyright: ignore[reportArgumentType]
    ) -> list[Any]:  # type: ignore
        start_time = time.time()
        request_id = f"embd-{await self.request_counter.next()}"
        logger.info("predict() commencing request_id=%s", request_id)

        request = EmbeddingRequest(
            inputs=texts,
            normalize=normalize,
        )

        if self.config.enable_log_requests:
            logger.info("Embedding request=%s", request)

        response = await self.queue_worker.submit(data=request, request_id=request_id)

        if logger.isEnabledFor(logging.DEBUG):
            for text, embedding in zip(request.inputs, response.outputs):
                logger.debug(
                    "Embedding result text='%s', embedding_length=%s",
                    text,
                    len(embedding),
                )

        logger.info("Generation took %.2fs", time.time() - start_time)

        logger.info("predict() complete")

        return response.outputs

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
        try:
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
                    output = await predictor.predict(**inputs)
                    print(output)
        finally:
            await predictor.stop()

    asyncio.run(main())
