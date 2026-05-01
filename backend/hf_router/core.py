import os
import re
import json
import logging
import requests
import functools
from pathlib import Path
from typing import Dict, Optional, Any, Union

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, HfApi
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# --- Security ---
HF_TOKEN_PATTERN = re.compile(r'(hf_[a-zA-Z0-9]{10,})')

def mask_tokens(text: str) -> str:
    """Masks Hugging Face tokens in a string to prevent secret leakage."""
    if not isinstance(text, str):
        return text
    return HF_TOKEN_PATTERN.sub('hf_***MASKED***', text)

class SecureFormatter(logging.Formatter):
    """A logging formatter that automatically masks HF tokens."""
    def format(self, record: logging.LogRecord) -> str:
        original_msg = super().format(record)
        return mask_tokens(original_msg)

def get_secure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = SecureFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

class SecureException(Exception):
    """An exception class that masks HF tokens in its message."""
    def __init__(self, message: str, *args: Any):
        super().__init__(mask_tokens(message), *args)

logger = get_secure_logger(__name__)

# --- Configuration & Token Management ---
class TokenManager:
    """
    Manages loading and resolving Hugging Face tokens securely from .env files,
    environment variables, and JSON configuration maps.
    """
    def __init__(self, tokens_file: str = "tokens.json", load_env: bool = True):
        self.rules: Dict[str, str] = {}
        if load_env:
            load_dotenv()
            
        self._load_tokens(tokens_file)

    def _resolve_value(self, val: str) -> Optional[str]:
        if val.startswith("env:"):
            env_var = val[4:]
            return os.getenv(env_var)
        return val

    def _load_tokens(self, tokens_file: str):
        path = Path(tokens_file)
        if path.exists():
            try:
                with open(path, "r") as f:
                    raw_rules = json.load(f)
                
                for pattern, raw_token in raw_rules.items():
                    resolved = self._resolve_value(raw_token)
                    if resolved:
                        self.rules[pattern] = resolved
                logger.info(f"Loaded {len(self.rules)} routing rules from {tokens_file}.")
            except Exception as e:
                logger.error(f"Failed to load {tokens_file}: {e}")
        else:
            logger.info(f"Tokens file {tokens_file} not found. Relying on environment variables.")
            default_token = os.getenv("HF_DEFAULT_TOKEN")
            if default_token:
                self.rules["*"] = default_token

    def get_token_for_repo(self, repo_id: str) -> Optional[str]:
        """Routes to the correct token based on repository ID."""
        if repo_id in self.rules:
            return self.rules[repo_id]
            
        parts = repo_id.split("/")
        if len(parts) == 2:
            org, _ = parts
            org_wildcard = f"{org}/*"
            if org_wildcard in self.rules:
                return self.rules[org_wildcard]
                
        return self.rules.get("*")

# --- Routing ---
class ModelRouter:
    """Intelligently routes requests to appropriate endpoints or tokens."""
    def __init__(self, token_manager: Optional[TokenManager] = None):
        self.token_manager = token_manager or TokenManager()

    def get_routing_info(self, repo_id: str) -> dict:
        """
        Determines the routing details for a given repository.
        Returns a dict containing token and endpoint preferences.
        """
        token = self.token_manager.get_token_for_repo(repo_id)
        
        endpoint_preference = "default"
        if repo_id.startswith("enterprise/"):
            endpoint_preference = "enterprise_secure"
        elif "70b" in repo_id.lower() or "8x7b" in repo_id.lower():
            endpoint_preference = "high_memory"
            
        return {
            "token": token,
            "endpoint_preference": endpoint_preference
        }

# --- Gateway ---
class UnifiedGateway:
    def __init__(self, router: Optional[ModelRouter] = None):
        self.router = router or ModelRouter()
        self._client_cache: Dict[str, InferenceClient] = {}

    def _get_client(self, repo_id: str, token: str) -> InferenceClient:
        """Retrieves a cached InferenceClient or instantiates a new one."""
        cache_key = f"{repo_id}:{token}"
        if cache_key not in self._client_cache:
            self._client_cache[cache_key] = InferenceClient(model=repo_id, token=token)
        return self._client_cache[cache_key]

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        reraise=True
    )
    def generate(self, repo_id: str, prompt: str, **kwargs: Any) -> str:
        """
        A unified generation abstraction that automatically resolves the correct
        authentication token and handles transient failures.
        """
        routing_info = self.router.get_routing_info(repo_id)
        token = routing_info.get("token")
        
        if not token:
            logger.warning(f"No token found for {repo_id}. Inference may fail if repo is private.")

        logger.info(f"Routing inference for '{repo_id}' using endpoint preference: {routing_info['endpoint_preference']}")
        
        try:
            client = self._get_client(repo_id, token)
            response = client.text_generation(prompt, **kwargs)
            return response
        except Exception as e:
            raise SecureException(f"Inference failed for {repo_id}: {str(e)}") from None

# --- Upload Utilities ---
class ModelUploader:
    def __init__(self, token_manager: Optional[TokenManager] = None):
        self.token_manager = token_manager or TokenManager()

    def upload_model(
        self, 
        repo_id: str, 
        folder_path: Union[str, Path], 
        private: bool = True,
        commit_message: str = "Upload model via HF Router"
    ) -> str:
        """
        Uploads a local folder to the Hugging Face Hub, automatically resolving the token.
        """
        token = self.token_manager.get_token_for_repo(repo_id)
        if not token:
            raise SecureException(f"Cannot upload to {repo_id}: No valid token resolved.")

        api = HfApi(token=token)
        
        try:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            logger.info(f"Repository {repo_id} ready (private={private}).")
            
            url = api.upload_folder(
                folder_path=str(folder_path),
                repo_id=repo_id,
                commit_message=commit_message
            )
            logger.info(f"Successfully uploaded {folder_path} to {repo_id}")
            return url
        except Exception as e:
            raise SecureException(f"Failed to upload model to {repo_id}: {str(e)}") from None
