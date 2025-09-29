"""
Custom HTTP client with optimized connection pooling.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

# Configure logging
logging.getLogger("urllib3").setLevel(logging.WARNING)

def create_http_session(
    pool_connections=10,
    pool_maxsize=20,
    max_retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    pool_block=False
):
    """
    Create a requests Session with connection pooling optimized for multiple concurrent requests.
    
    Args:
        pool_connections: The number of urllib3 connection pools to cache
        pool_maxsize: The maximum number of connections to save in the pool
        max_retries: Maximum number of retries for failed requests
        backoff_factor: Backoff factor for retries
        status_forcelist: HTTP status codes that should trigger a retry
        pool_block: Whether the connection pool should block for connections
        
    Returns:
        A configured requests.Session object
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
    )
    
    # Create and mount the adapter
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry_strategy,
        pool_block=pool_block
    )
    
    # Mount the adapter for both http and https
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Create a global session that can be imported and used throughout the application
http_session = create_http_session()
