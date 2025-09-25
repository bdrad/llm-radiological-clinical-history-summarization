import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import torch
from transformers import pipeline

class HuggingFaceProvider:
    """Provider for HuggingFace transformer models."""
    
    def __init__(self, model_name: str, max_tokens: int = 200):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.pipeline = None
        self._load_credentials()
        self._initialize_pipeline()
    
    def _load_credentials(self):
        """Load HuggingFace credentials."""
        env_path = "/mnt/sohn2022/Adrian/Utils/Credentials/.env"
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token:
            try:
                from huggingface_hub import login
                login(token=token)
            except ImportError:
                print("Warning: huggingface_hub not available for authentication")
    
    def _initialize_pipeline(self):        
        try:
            # Initialize pipeline with appropriate settings
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,  # Some models may require this
            )
            
            print(f"Initialized HuggingFace pipeline for {self.model_name}")
            
        except ImportError as e:
            raise ImportError(f"Required packages not installed for HuggingFace provider: {e}")
        except Exception as e:
            raise Exception(f"Failed to initialize HuggingFace pipeline: {e}")
    
    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Generate text using HuggingFace model.
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter for sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            generation_params = {
                "max_new_tokens": self.max_tokens,
                "pad_token_id": self.pipeline.tokenizer.eos_token_id,
                "eos_token_id": self.pipeline.tokenizer.eos_token_id,
            }
            
            if temperature is not None:
                generation_params["temperature"] = temperature
            else:
                generation_params["temperature"] = 0
            
            result = self.pipeline(prompt, **generation_params)            
            if result and len(result) > 0:
                return result[0]["generated_text"].strip()
            else:
                return "ERROR: No response generated"
                
        except Exception as e:
            print(f"Error with HuggingFace model: {e}")
            return f"ERROR: {str(e)}"

__all__ = ["HuggingFaceProvider"]
