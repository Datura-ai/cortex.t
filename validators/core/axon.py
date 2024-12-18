import os
import uuid
import copy
import time
import asyncio
import inspect
import uvicorn
import traceback
import threading
import contextlib

from inspect import Signature

from cursor.app.core.query_to_validator import axon_to_use
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, APIRouter, Request, Response, Depends
from starlette.responses import Response
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing import List, Optional, Tuple, Callable, Any, Dict
from bittensor.core.synapse import Synapse
from bittensor.core.settings import DEFAULTS, version_as_int

from bittensor.core.errors import (
    InvalidRequestNameError,
    SynapseDendriteNoneException,
    SynapseParsingError,
    UnknownSynapseError,
    NotVerifiedException,
    BlacklistedException,
    PriorityException,
    RunException,
    PostProcessException,
    InternalServerError,
    SynapseException
)
from bittensor.core.threadpool import PriorityThreadPoolExecutor
from bittensor.core.axon import AxonMiddleware
import bittensor
from bittensor.utils import networking
import bittensor as bt
from bittensor_wallet import Wallet
from substrateinterface import Keypair
from cursor.app.core.config import config


class FastAPIThreadedServer(uvicorn.Server):
    """
    The ``FastAPIThreadedServer`` class is a specialized server implementation for the Axon server in the Bittensor network.

    It extends the functionality of :func:`uvicorn.Server` to run the FastAPI application in a separate thread, allowing the Axon server to handle HTTP requests concurrently and non-blocking.

    This class is designed to facilitate the integration of FastAPI with the Axon's asynchronous architecture, ensuring efficient and scalable handling of network requests.

    Importance and Functionality
        Threaded Execution
            The class allows the FastAPI application to run in a separate thread, enabling concurrent handling of HTTP requests which is crucial for the performance and scalability of the Axon server.

        Seamless Integration
            By running FastAPI in a threaded manner, this class ensures seamless integration of FastAPI's capabilities with the Axon server's asynchronous and multi-threaded architecture.

        Controlled Server Management
            The methods start and stop provide controlled management of the server's lifecycle, ensuring that the server can be started and stopped as needed, which is vital for maintaining the Axon server's reliability and availability.

        Signal Handling
            Overriding the default signal handlers prevents potential conflicts with the Axon server's main application flow, ensuring stable operation in various network conditions.

    Use Cases
        Starting the Server
            When the Axon server is initialized, it can use this class to start the FastAPI application in a separate thread, enabling it to begin handling HTTP requests immediately.

        Stopping the Server
            During shutdown or maintenance of the Axon server, this class can be used to stop the FastAPI application gracefully, ensuring that all resources are properly released.

    Args:
        should_exit (bool): Flag to indicate whether the server should stop running.
        is_running (bool): Flag to indicate whether the server is currently running.

    The server overrides the default signal handlers to prevent interference with the main application flow and provides methods to start and stop the server in a controlled manner.
    """

    should_exit: bool = False
    is_running: bool = False

    def install_signal_handlers(self):
        """
        Overrides the default signal handlers provided by ``uvicorn.Server``. This method is essential to ensure that the signal handling in the threaded server does not interfere with the main application's flow, especially in a complex asynchronous environment like the Axon server.
        """
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        """
        Manages the execution of the server in a separate thread, allowing the FastAPI application to run asynchronously without blocking the main thread of the Axon server. This method is a key component in enabling concurrent request handling in the Axon server.

        Yields:
            None: This method yields control back to the caller while the server is running in the background thread.
        """
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

    def _wrapper_run(self):
        """
        A wrapper method for the :func:`run_in_thread` context manager. This method is used internally by the ``start`` method to initiate the server's execution in a separate thread.
        """
        with self.run_in_thread():
            while not self.should_exit:
                time.sleep(1e-3)

    def start(self):
        """
        Starts the FastAPI server in a separate thread if it is not already running. This method sets up the server to handle HTTP requests concurrently, enabling the Axon server to efficiently manage
        incoming network requests.

        The method ensures that the server starts running in a non-blocking manner, allowing the Axon server to continue its other operations seamlessly.
        """
        if not self.is_running:
            self.should_exit = False
            thread = threading.Thread(target=self._wrapper_run, daemon=True)
            thread.start()
            self.is_running = True

    def stop(self):
        """
        Signals the FastAPI server to stop running. This method sets the :func:`should_exit` flag to ``True``, indicating that the server should cease its operations and exit the running thread.

        Stopping the server is essential for controlled shutdowns and resource management in the Axon server, especially during maintenance or when redeploying with updated configurations.
        """
        if self.is_running:
            self.should_exit = True


class CortexAxon(bt.axon):
    def __init__(
            self,
            wallet: Optional["Wallet"] = None,
            config: Optional["Config"] = None,
            port: Optional[int] = None,
            ip: Optional[str] = None,
            external_ip: Optional[str] = None,
            external_port: Optional[int] = None,
            max_workers: Optional[int] = None,
    ):
        r"""Creates a new bittensor.Axon object from passed arguments.
        Args:
            config (:obj:`Optional[bittensor.config]`, `optional`):
                bittensor.axon.config()
            wallet (:obj:`Optional[bittensor.wallet]`, `optional`):
                bittensor wallet with hotkey and coldkeypub.
            port (:type:`Optional[int]`, `optional`):
                Binding port.
            ip (:type:`Optional[str]`, `optional`):
                Binding ip.
            external_ip (:type:`Optional[str]`, `optional`):
                The external ip of the server to broadcast to the network.
            external_port (:type:`Optional[int]`, `optional`):
                The external port of the server to broadcast to the network.
            max_workers (:type:`Optional[int]`, `optional`):
                Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
        """
        # Build and check config.
        if config is None:
            config = CortexAxon.config()
        config = copy.deepcopy(config)
        config.axon.ip = ip or config.axon.ip
        config.axon.port = port or config.axon.port
        config.axon.external_ip = external_ip or config.axon.external_ip
        config.axon.external_port = external_port or config.axon.external_port
        config.axon.max_workers = max_workers or config.axon.max_workers
        CortexAxon.check_config(config)
        self.config = config

        # Get wallet or use default.
        self.wallet = wallet or Wallet()

        # Build axon objects.
        self.uuid = str(uuid.uuid1())
        self.ip = self.config.axon.ip
        self.port = self.config.axon.port
        self.external_ip = (
            self.config.axon.external_ip  # type: ignore
            if self.config.axon.external_ip is not None  # type: ignore
            else networking.get_external_ip()
        )
        self.external_port = (
            self.config.axon.external_port  # type: ignore
            if self.config.axon.external_port is not None  # type: ignore
            else self.config.axon.port  # type: ignore
        )
        self.full_address = str(self.config.axon.ip) + ":" + str(self.config.axon.port)
        self.started = False

        # Build middleware
        self.thread_pool = bittensor.PriorityThreadPoolExecutor(
            max_workers=self.config.axon.max_workers
        )
        self.nonces: Dict[str, int] = {}

        # Request default functions.
        self.forward_class_types: Dict[str, List[Signature]] = {}
        self.blacklist_fns: Dict[str, Optional[Callable]] = {}
        self.priority_fns: Dict[str, Optional[Callable]] = {}
        self.forward_fns: Dict[str, Optional[Callable]] = {}
        self.verify_fns: Dict[str, Optional[Callable]] = {}
        self.required_hash_fields: Dict[str, str] = {}

        # Instantiate FastAPI
        self.app = FastAPI()
        log_level = "trace" if bittensor.logging.__trace_on__ else "critical"
        self.fast_config = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.config.axon.port, log_level=log_level
        )
        self.fast_server = FastAPIThreadedServer(config=self.fast_config)
        self.router = APIRouter()
        self.app.include_router(self.router)

        # Attach default forward.
        def ping(r: bittensor.Synapse) -> bittensor.Synapse:
            return r

        self.attach(
            forward_fn=ping, verify_fn=None, blacklist_fn=None, priority_fn=None
        )
        self.middleware_cls = CortexAxonMiddleware
        self.app.add_middleware(self.middleware_cls, axon=self)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all HTTP methods
            allow_headers=["*"],  # Allows all headers
        )

    def default_verify(self, synapse: bittensor.Synapse):
        if synapse.dendrite is not None:
            keypair = Keypair(ss58_address=synapse.dendrite.hotkey)

            # Build the signature messages.
            message = f"{synapse.dendrite.nonce}.{synapse.dendrite.hotkey}.{self.wallet.hotkey.ss58_address}.{synapse.dendrite.uuid}.{synapse.computed_body_hash}"

            # Build the unique endpoint key.
            endpoint_key = f"{synapse.dendrite.hotkey}:{synapse.dendrite.uuid}"

            if not keypair.verify(message, synapse.dendrite.signature):
                raise Exception(
                    f"Signature mismatch with {message} and {synapse.dendrite.signature}"
                )

            # Success
            self.nonces[endpoint_key] = synapse.dendrite.nonce  # type: ignore
        else:
            raise SynapseDendriteNoneException()


def create_error_response(synapse: bittensor.Synapse):
    if synapse.axon is None:
        return JSONResponse(
            status_code=400,
            headers=synapse.to_headers(),
            content={"message": "Invalid request name"},
        )
    else:
        return JSONResponse(
            status_code=synapse.axon.status_code or 400,
            headers=synapse.to_headers(),
            content={"message": synapse.axon.status_message},
        )


def log_and_handle_error(
        synapse: bittensor.Synapse,
        exception: Exception,
        status_code: int,
        start_time: float,
):
    # Display the traceback for user clarity.
    bittensor.logging.trace(f"Forward exception: {traceback.format_exc()}")

    # Set the status code of the synapse to the given status code.
    error_type = exception.__class__.__name__
    error_message = str(exception)
    detailed_error_message = f"{error_type}: {error_message}"

    # Log the detailed error message for internal use
    bittensor.logging.error(detailed_error_message)

    if synapse.axon is None:
        raise SynapseParsingError(detailed_error_message)
    # Set a user-friendly error message
    synapse.axon.status_code = status_code
    synapse.axon.status_message = error_message

    # Calculate the processing time by subtracting the start time from the current time.
    synapse.axon.process_time = str(time.time() - start_time)  # type: ignore

    return synapse


class CortexAxonMiddleware(AxonMiddleware):
    """
    The `AxonMiddleware` class is a key component in the Axon server, responsible for processing all incoming requests.

    It handles the essential tasks of verifying requests, executing blacklist checks,
    running priority functions, and managing the logging of messages and errors. Additionally, the class
    is responsible for updating the headers of the response and executing the requested functions.

    This middleware acts as an intermediary layer in request handling, ensuring that each request is
    processed according to the defined rules and protocols of the Bittensor network. It plays a pivotal
    role in maintaining the integrity and security of the network communication.

    Args:
        app (FastAPI): An instance of the FastAPI application to which this middleware is attached.
        axon (bittensor.axon): The Axon instance that will process the requests.

    The middleware operates by intercepting incoming requests, performing necessary preprocessing
    (like verification and priority assessment), executing the request through the Axon's endpoints, and
    then handling any postprocessing steps such as response header updating and logging.
    """

    async def dispatch(
            self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Asynchronously processes incoming HTTP requests and returns the corresponding responses. This
        method acts as the central processing unit of the AxonMiddleware, handling each step in the
        request lifecycle.

        Args:
            request (Request): The incoming HTTP request to be processed.
            call_next (RequestResponseEndpoint): A callable that processes the request and returns a response.

        Returns:
            Response: The HTTP response generated after processing the request.

        This method performs several key functions:

        1. Request Preprocessing: Sets up Synapse object from request headers and fills necessary information.
        2. Logging: Logs the start of request processing.
        3. Blacklist Checking: Verifies if the request is blacklisted.
        4. Request Verification: Ensures the authenticity and integrity of the request.
        5. Priority Assessment: Evaluates and assigns priority to the request.
        6. Request Execution: Calls the next function in the middleware chain to process the request.
        7. Response Postprocessing: Updates response headers and logs the end of the request processing.

        The method also handles exceptions and errors that might occur during each stage, ensuring that
        appropriate responses are returned to the client.
        """
        # Records the start time of the request processing.
        start_time = time.time()

        if "v1" in request.url.path:
            if request.method == "OPTIONS":
                return await call_next(request)
            try:
                api_key = request.headers.get("Authorization").split(" ")[1]
                if not api_key or api_key != config.api_key:
                    return JSONResponse(
                        {"detail": "Invalid or missing API Key"}, status_code=401
                    )
                return await call_next(request)
            except Exception:
                return JSONResponse(
                    {"detail": "Invalid or missing API Key"}, status_code=401
                )

        try:
            # Set up the synapse from its headers.
            try:
                synapse: "Synapse" = await self.preprocess(request)
            except Exception as exc:
                if isinstance(exc, SynapseException) and exc.synapse is not None:
                    synapse = exc.synapse
                else:
                    synapse = Synapse()
                raise

            # Logs the start of the request processing
            if synapse.dendrite is not None:
                bittensor.logging.trace(
                    f"axon     | <-- | {request.headers.get('content-length', -1)} B | {synapse.name} | {synapse.dendrite.hotkey} | {synapse.dendrite.ip}:{synapse.dendrite.port} | 200 | Success "
                )
            else:
                bittensor.logging.trace(
                    f"axon     | <-- | {request.headers.get('content-length', -1)} B | {synapse.name} | None | None | 200 | Success "
                )

            # Call the blacklist function
            await self.blacklist(synapse)

            # Call verify and return the verified request
            await self.verify(synapse)

            # Call the priority function
            await self.priority(synapse)

            # Call the run function
            response = await self.run(synapse, call_next, request)

        # Handle errors related to preprocess.
        except InvalidRequestNameError as e:
            if "synapse" not in locals():
                synapse: bittensor.Synapse = bittensor.Synapse()  # type: ignore
            log_and_handle_error(synapse, e, 400, start_time)
            response = create_error_response(synapse)

        except SynapseParsingError as e:
            if "synapse" not in locals():
                synapse = bittensor.Synapse()
            log_and_handle_error(synapse, e, 400, start_time)
            response = create_error_response(synapse)

        except UnknownSynapseError as e:
            if "synapse" not in locals():
                synapse = bittensor.Synapse()
            log_and_handle_error(synapse, e, 404, start_time)
            response = create_error_response(synapse)

        # Handle errors related to verify.
        except NotVerifiedException as e:
            log_and_handle_error(synapse, e, 401, start_time)
            response = create_error_response(synapse)

        # Handle errors related to blacklist.
        except BlacklistedException as e:
            log_and_handle_error(synapse, e, 403, start_time)
            response = create_error_response(synapse)

        # Handle errors related to priority.
        except PriorityException as e:
            log_and_handle_error(synapse, e, 503, start_time)
            response = create_error_response(synapse)

        # Handle errors related to run.
        except RunException as e:
            log_and_handle_error(synapse, e, 500, start_time)
            response = create_error_response(synapse)

        # Handle errors related to postprocess.
        except PostProcessException as e:
            log_and_handle_error(synapse, e, 500, start_time)
            response = create_error_response(synapse)

        # Handle all other errors.
        except Exception as e:
            log_and_handle_error(synapse, InternalServerError(str(e)), 500, start_time)
            response = create_error_response(synapse)

        # Logs the end of request processing and returns the response
        finally:
            # Log the details of the processed synapse, including total size, name, hotkey, IP, port,
            # status code, and status message, using the debug level of the logger.
            if synapse.dendrite is not None and synapse.axon is not None:
                bittensor.logging.trace(
                    f"axon     | --> | {response.headers.get('content-length', -1)} B | {synapse.name} | {synapse.dendrite.hotkey} | {synapse.dendrite.ip}:{synapse.dendrite.port}  | {synapse.axon.status_code} | {synapse.axon.status_message}"
                )
            elif synapse.axon is not None:
                bittensor.logging.trace(
                    f"axon     | --> | {response.headers.get('content-length', -1)} B | {synapse.name} | None | None | {synapse.axon.status_code} | {synapse.axon.status_message}"
                )
            else:
                bittensor.logging.trace(
                    f"axon     | --> | {response.headers.get('content-length', -1)} B | {synapse.name} | None | None | 200 | Success "
                )

            # Return the response to the requester.
            return response
