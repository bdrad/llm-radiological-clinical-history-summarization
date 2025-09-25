from typing import Optional, Dict, Any, Union
from .anthropic_aws_bedrock import AWSClaudeProvider
from .openai_azure import AzureOpenAIProvider  
from .huggingface import HuggingFaceProvider

class ProviderFactory:
    """Factory for creating and managing LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, model_name: str, max_tokens: int = 200) -> Union[AWSClaudeProvider, AzureOpenAIProvider, HuggingFaceProvider]:
        """
        Create a provider instance based on type.
        
        Args:
            provider_type: Type of provider ("aws_claude", "azure_openai", "huggingface")
            model_name: Name of the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Provider instance
        """
        if provider_type == "anthropic_aws":
            return AWSClaudeProvider(model_name=model_name, max_tokens=max_tokens)
        elif provider_type == "openai_azure":
            return AzureOpenAIProvider(model_name=model_name, max_tokens=max_tokens)
        elif provider_type == "huggingface":
            return HuggingFaceProvider(model_name=model_name, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def get_available_models() -> Dict[str, Dict[str, str]]:
        """Get dictionary of available models by provider."""
        return {
            "aws_claude": {
                "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
            },
            "azure_openai": {
                "gpt4o": "GPT-4o",
                "gpt4o_mini": "GPT-4o mini",
                "o1": "o1",
                "o1_mini": "o1 mini"
            },
            "huggingface": {
				"llama3_1_8b",
                "mistral_7b",
                "biomistral_7b",
                "meditron3_8b",
                "qwen2_5_7b",				
				"deepseek_r1_llama_8b",
                "deepseek_r1_qwen_7b"
			}
        }
    
    @staticmethod
    def get_provider_info(provider_type: str, model_name: str) -> Dict[str, Any]:
        """Get information about a specific provider/model combination."""
        try:
            provider = ProviderFactory.create_provider(provider_type, model_name)
            return provider.get_model_info()
        except Exception as e:
            return {
                "provider": provider_type,
                "model_name": model_name,
                "error": str(e),
                "available": False
            }

class UnifiedProvider:
    """Unified interface for all providers with consistent API."""
    
    def __init__(self, provider_type: str, model_name: str, max_tokens: int = 200):
        self.provider_type = provider_type
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.provider = ProviderFactory.create_provider(provider_type, model_name, max_tokens)
    
    def generate(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> str:
        """Generate text using the configured provider."""
        return self.provider.generate(prompt, temperature=temperature, **kwargs)
    
    def supports_temperature(self) -> bool:
        """Check if the provider supports temperature parameter."""
        return self.provider.supports_temperature()
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return self.provider.get_model_info()
    
    def cleanup(self):
        """Clean up provider resources."""
        if hasattr(self.provider, 'cleanup'):
            self.provider.cleanup()

# Export main classes and functions
__all__ = [
    "ProviderFactory", 
    "UnifiedProvider",
    "AWSClaudeProvider",
    "AzureOpenAIProvider", 
    "HuggingFaceProvider"
]
