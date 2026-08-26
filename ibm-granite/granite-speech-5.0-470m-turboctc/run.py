# SPDX-License-Identifier: Apache-2.0

# Runner interface for Cog ⚙️
# https://cog.run/python


import asyncio
import base64
import contextlib
import inspect
import io
import json
import logging
import os
import pathlib
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, override

import soundfile as sf
import soxr
import torch
from cog import BaseRunner, Input, Path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from soundfile import AudioData, AudioData_2d, FileDescriptorOrPath
from transformers import AutoModelForCTC, AutoProcessor


def init_logger(name: str) -> logging.Logger:
    """Create a Logger for the specified name.

    Args:
        name (str): The name for the Logger.

    Returns:
        logging.Logger: A configured Logger for the specified name.
    """
    _logger = logging.getLogger(name)

    _logger.setLevel(os.environ.get("RUNNER_LOG_LEVEL", "DEBUG").upper())
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
        self._consumer_task: asyncio.Task[None] | None = None
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
        self._consumer_task = asyncio.create_task(self._consumer_loop(), name=f"{self.name}-consumer")
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
            except TimeoutError:
                logger.error("%s shutdown timed out, cancelling task", self.name)
                self._consumer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._consumer_task

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
        result_future: asyncio.Future[R] = asyncio.get_event_loop().create_future()

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
            while not self._shutdown_event.is_set() or not self.queue.empty():
                try:
                    # Wait for a request with timeout to check shutdown
                    request = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                except TimeoutError:
                    # Check if we should shutdown
                    if self._shutdown_event.is_set() and self.queue.empty():
                        break
                    continue

                # Process the request
                try:
                    logger.debug("%s processing request: %s", self.name, request.request_id)
                    result = await self.processor(request.data)
                    request.result_future.set_result(result)
                    logger.debug("%s completed request: %s", self.name, request.request_id)
                except Exception as e:
                    logger.error("%s error processing request: %s", self.name, e, exc_info=True)
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


class RunnerConfig(BaseModel):
    """
    RunnerConfig is a configuration class for the Runner.
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


@dataclass
class AudioRequest:
    inputs: list[Path]


@dataclass
class AudioResponse:
    outputs: list[str]
    input_tokens: int
    output_tokens: int


class Runner(BaseRunner):
    def _resolve_weights_path(self, weights: Path | str | None) -> Path:
        """Resolve and validate the weights path."""
        if not weights:
            return Path("./weights")
        return Path(weights) if isinstance(weights, str) else weights

    def _audio_to_float(self, source: FileDescriptorOrPath, target_sampling_rate: int | None = None) -> AudioData | AudioData_2d:
        """Read audio from a file path or data: URI into a float32 numpy array.

        Args:
            source (FileDescriptorOrPath): The audio source. Can be a file path or a
                data: URI (e.g. ``data:audio/wav;base64,<b64>``).
            target_sampling_rate (int | None): If provided and different from the
                audio's native sampling rate, the audio is resampled to this rate
                using ``soxr``. Defaults to ``None`` (no resampling).

        Returns:
            AudioData | AudioData_2d: A float32 numpy array of audio samples.
        """
        if isinstance(source, str) and source.startswith("data:"):
            # Split off the "data:audio/wav;base64," header
            _, b64data = source.split(",", 1)
            # Decode base64 to raw bytes
            audio_bytes = base64.b64decode(b64data)
            source = io.BytesIO(audio_bytes)

        # Read into a numpy array using soundfile (handles WAV headers, bit depth, etc.)
        data, sampling_rate = sf.read(source, dtype="float32")

        if target_sampling_rate is not None and target_sampling_rate != sampling_rate:
            data = soxr.resample(data, sampling_rate, target_sampling_rate)

        return data

    async def transcribe(self, request: AudioRequest) -> AudioResponse:
        """Transcribe a batch of audio inputs to text.

        Decodes each audio file in the request to a float32 numpy array,
        runs the CTC model, and returns one transcript string per input.

        Args:
            request: An :class:`AudioRequest` whose ``inputs`` is a list of
                :class:`~cog.Path` objects pointing to the audio files to
                transcribe.

        Returns:
            An :class:`AudioResponse` with:

            * ``outputs`` — a list of transcript strings, one per input audio
              file, in the same order as ``request.inputs``.
            * ``input_tokens`` — the total number of non-padded encoder
              time-steps consumed across the batch.
            * ``output_tokens`` — the total number of non-blank/pad output
              tokens produced across the batch.
        """
        # Decode each audio file to a float32 numpy array, resampling if needed
        speech = [self._audio_to_float(input, self.processor.feature_extractor.sampling_rate) for input in request.inputs]

        # Extract log-mel features and build attention masks for the model
        inputs = self.processor(speech, sampling_rate=self.processor.feature_extractor.sampling_rate, device=self.model.device)
        inputs.to(self.model.device, dtype=self.model.dtype)

        # Total input frames across the batch (non-padded encoder time steps)
        input_tokens: int = int(inputs["attention_mask"].sum())

        # Run the CTC model forward pass without gradient tracking
        with torch.no_grad():
            model_output = self.model.generate(**inputs)

        # Total output tokens across the batch (excluding pad/blank token)
        pad_id: int = self.processor.tokenizer.pad_token_id or 0
        output_tokens: int = int((model_output != pad_id).sum())

        # Decode token ids to transcript strings, removing special tokens
        outputs = self.processor.batch_decode(model_output, skip_special_tokens=True)

        return AudioResponse(outputs=outputs, input_tokens=input_tokens, output_tokens=output_tokens)

    async def _run_warmup_test(self, audio: list[Path]) -> None:
        """Run a warmup inference to ensure the model is ready."""
        inputs: dict[str, Any] = self._defaults | {
            "audio": audio,
        }
        generator = self.run(**inputs)
        test_output = "\n".join([output async for output in generator])
        logger.debug("Test inference output test_output=%s", test_output)

    @override
    # pyrefly: ignore[bad-override]
    async def setup(self, weights: Path | str | None) -> None:
        logger.info("setup() commencing")

        # Load configuration
        self.config = self.load_config()

        # Initialize model
        weights = self._resolve_weights_path(weights)
        model_path = self.config.pretrained_model_name_or_path or weights.resolve().as_posix()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCTC.from_pretrained(model_path, **self.config.model_kwargs)
        logger.debug(
            "Using model %s on device %s",
            self.config.served_model_name,
            self.model.device,
        )
        self.model.eval()

        # Initialize request counter
        self.request_counter = Counter(1)

        # Start queue worker
        self.queue_worker = ProducerConsumer(self.transcribe, max_queue_size=10, name="AudioWorker")
        await self.queue_worker.start()

        # Run warmup test
        await self._run_warmup_test([Path("./multilingual_sample.wav")])

        logger.info("setup() complete")

    async def stop(self, timeout: float | None = None) -> None:
        await self.queue_worker.stop(timeout)

    @override
    # pyrefly: ignore[bad-override]
    async def run(
        self,
        # audio must be the first argument
        # The LangChain Replicate class will use the first argument to supply the input
        audio: list[Path] | None = Input(
            description="Audio input.",
            default=None,
        ),
    ) -> AsyncIterator[str]:
        start_time = time.time()
        request_id = f"speech-{await self.request_counter.next()}"
        logger.info("run() commencing request_id=%s", request_id)

        if audio is None:
            audio = []
        request = AudioRequest(
            inputs=audio,
        )

        if self.config.enable_log_requests:
            logger.info("Audio request=%s", request)

        response = await self.queue_worker.submit(data=request, request_id=request_id)

        for output in response.outputs:
            yield output

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Transcription result=%s",
                "\n".join(response.outputs),
            )

        logger.info("Generation took %.2fs", time.time() - start_time)

        logger.debug(
            "input_tokens=%s, output_tokens=%s, total_tokens=%s",
            response.input_tokens,
            response.output_tokens,
            response.input_tokens + response.output_tokens,
        )
        self.record_metric("token_input_count", response.input_tokens)
        self.record_metric("token_output_count", response.output_tokens)

        logger.info("run() complete")

    _defaults: dict[str, Any] = {key: param.default.default for key, param in inspect.signature(run).parameters.items() if hasattr(param.default, "default")}

    def load_config(self) -> RunnerConfig:
        """
        Load the runner configuration from the current directory.

        Load `runner_config.json` from the current directory.
        Return a default RunnerConfig object if not found or an error occurs.

        Priority:
        1. Load `runner_config.json` from the  current directory.
        2. If not found or an error occurs, return a default RunnerConfig object.

        Returns:
            RunnerConfig: The loaded runner configuration.
        """

        runner_config_path = pathlib.Path("runner_config.json")
        if runner_config_path.exists():
            logger.debug("Loading runner_config.json path=%s", runner_config_path)
            json_data = runner_config_path.read_text(encoding="utf-8")
            config = RunnerConfig.model_validate_json(json_data)
        else:
            config = RunnerConfig()

        logger.debug("RunnerConfig config=%s", config)
        return config


# For testing
if __name__ == "__main__":

    async def main():
        """Async main method for direct testing."""
        config_path = "./weights/config.json"
        weights = Path(str(config_path)).parent
        runner = Runner()
        try:
            await runner.setup(weights)

            if len(sys.argv) >= 2:
                file_paths = sys.argv[1:]
                defaults = runner._defaults
                print()
                for path in file_paths:
                    print(f"### Test file: {path}")
                    json_str = pathlib.Path(path).read_text(encoding="utf-8")
                    json_dict: dict[str, Any] = json.loads(json_str)
                    inputs = defaults | json_dict
                    generator = runner.run(**inputs)
                    async for output in generator:
                        if output.startswith("{") and output.endswith("}"):
                            try:
                                print(json.dumps(json.loads(output), indent=4))
                                continue
                            except json.JSONDecodeError:
                                pass
                        print(output)
                    print()
        finally:
            await runner.stop()

    asyncio.run(main())
