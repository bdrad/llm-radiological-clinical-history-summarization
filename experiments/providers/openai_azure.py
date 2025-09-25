import os
import json
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import requests
from openai import AzureOpenAI

REASONING_MODELS = ["o1", "o1_mini"]

class AzureOpenAIProvider:    
	"""Provider for Azure OpenAI transformer models."""

	def __init__(self, model_name: str = "gpt4o", max_tokens: int = None):
		self.api_details = {
			"o1": {
				"DEPLOYMENT_ID": "o1-2024-12-17",
				"API_VERSION": "2025-02-01-preview"
			},
			"o1_mini": {
				"DEPLOYMENT_ID": "o1-mini-2024-09-12",
				"API_VERSION": "2025-02-01-preview"
			},
			"gpt4o": {
				"DEPLOYMENT_ID": "gpt-4o-2024-05-13",
				"API_VERSION": "2025-02-01-preview"
			},
			"gpt4o_mini": {
				"DEPLOYMENT_ID": "gpt-4o-mini-2024-07-18",
				"API_VERSION": "2025-02-01-preview"
			}
		}
		
		self.model_name = model_name
		if max_tokens:
			self.max_tokens = max_tokens
		elif self.model_name in REASONING_MODELS:
			self.max_tokens = 500
		else:
			self.max_tokens = 200
		self.client = None
		self._initialize_client()
		
	def _initialize_client(self):
		env_path = "/mnt/sohn2022/Adrian/Utils/Credentials/.env"
		if os.path.exists(env_path):
			load_dotenv(env_path)
		else:
			load_dotenv()
					
		versa_api_key = os.environ.get('VERSA_API_KEY')
		versa_endpoint = os.environ.get('RESOURCE_ENDPOINT')
		api_version = self.api_details[self.model_name]["API_VERSION"]
		
		self.client = AzureOpenAI(
			api_key=versa_api_key,
			api_version=api_version,
			azure_endpoint=versa_endpoint,
		)
	
	def generate(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> str:
		"""
		Generate text using Azure OpenAI.
		
		Args:
			prompt: Input prompt
			temperature: Temperature parameter for sampling
			**kwargs: Additional parameters
			
		Returns:
			Generated text
		"""
		try:
			messages =  [{"role": "user", "content": prompt}]
			deployment = self.api_details[self.model_name]["DEPLOYMENT_ID"]

			if self.supports_temperature():
				response = self.client.chat.completions.create(
					model=deployment,
					messages=messages,
					temperature=temperature,
					# max_completion_tokens=self.max_tokens,
				)
			else:
				response = self.client.chat.completions.create(
					model=deployment,
					messages=messages,
					# max_completion_tokens=self.max_tokens,
				)
			return response.choices[0].message.content
		except Exception as e:
			print(f"Error with Azure OpenAI: {e}")
			return f"ERROR: {str(e)}"
	
	def supports_temperature(self) -> bool:
		"""Check if provider supports temperature parameter."""
		# OpenAI's o-series reasoning models don't support temperature
		return self.model_name not in REASONING_MODELS

__all__ = ["AzureOpenAIProvider"]