"""
Resource management for DSCIM pipeline.

Manages Dask cluster lifecycle and other computational resources.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DaskManager:
    """
    Manage Dask cluster lifecycle for pipeline execution.

    Provides context manager interface for automatic setup and cleanup
    of Dask distributed computing resources.

    Parameters
    ----------
    use_dask : bool, optional
        Whether to use Dask (default: True)
    n_workers : int, optional
        Number of workers. If None, uses CPU count
    threads_per_worker : int, optional
        Threads per worker (default: 1)
    memory_limit : str, optional
        Memory limit per worker (default: "4GB")

    Examples
    --------
    >>> # Automatic lifecycle management
    >>> with DaskManager(n_workers=4) as manager:
    ...     # Dask cluster is running
    ...     print(manager.client.dashboard_link)
    ...     # Process data
    ... # Cluster automatically shut down

    >>> # Manual control
    >>> manager = DaskManager()
    >>> manager.start()
    >>> # ... do work ...
    >>> manager.stop()
    """

    def __init__(
        self,
        use_dask: bool = True,
        n_workers: Optional[int] = None,
        threads_per_worker: int = 1,
        memory_limit: str = "4GB",
        verbose: bool = True,
    ):
        self.use_dask = use_dask
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.memory_limit = memory_limit
        self.verbose = verbose
        self._client = None
        self._cluster = None

    def start(self):
        """Start Dask cluster."""
        if not self.use_dask:
            if self.verbose:
                logger.info("Dask disabled, running in local mode")
            return

        try:
            from dask.distributed import Client, LocalCluster

            if self.verbose:
                logger.info("Starting Dask cluster...")

            self._cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
                silence_logs=not self.verbose,
            )
            self._client = Client(self._cluster)

            if self.verbose:
                logger.info(f"✓ Dask cluster started")
                logger.info(f"  Workers: {len(self._cluster.workers)}")
                logger.info(f"  Dashboard: {self._client.dashboard_link}")

        except ImportError:
            logger.warning(
                "Dask not available. Install with: pip install dask[distributed]"
            )
            self.use_dask = False

    def stop(self):
        """Stop Dask cluster."""
        if self._client:
            if self.verbose:
                logger.info("Shutting down Dask cluster...")
            self._client.close()
            self._client = None

        if self._cluster:
            self._cluster.close()
            self._cluster = None

            if self.verbose:
                logger.info("✓ Dask cluster shut down")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()

    @property
    def client(self):
        """Get Dask client."""
        return self._client

    @property
    def cluster(self):
        """Get Dask cluster."""
        return self._cluster
