import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from anthropic import AnthropicBedrock

class AWSClaudeProvider:    
    """Provider for Claude models accessed through AWS Bedrock."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", max_tokens: int = 200):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        env_path = "/mnt/sohn2022/Adrian/Utils/Credentials/.env"
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()
                    
        aws_access_key = os.environ.get('AWS_ACCESS_KEY')
        aws_secret_key = os.environ.get('AWS_SECRET_KEY')
        aws_endpoint = os.environ.get('AWS_ENDPOINT')
        aws_region = os.environ.get('AWS_REGION', 'us-west-2')
        
        self.client = AnthropicBedrock(
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key,
            aws_region=aws_region,
            base_url=aws_endpoint,
        )
    
    def generate(self, prompt: str, temperature: Optional[float] = None, **kwargs) -> str:
        try:
            model_id_map = {
                "claude3_5": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            }
            model_id = model_id_map.get(self.model_name, "anthropic.claude-3-5-sonnet-20240620-v1:0")            
            message_params = {
                "model": model_id,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if temperature is not None:
                message_params["temperature"] = temperature
            else:
                message_params["temperature"] = 0
            
            message_params.update(kwargs)            
            message = self.client.messages.create(**message_params)
            
            if message and hasattr(message, 'content') and len(message.content) >= 1:
                return message.content[0].text.strip()
            else:
                return "ERROR: No response generated"
                
        except Exception as e:
            print(f"Error with AWS Claude: {e}")
            return f"ERROR: {str(e)}"

__all__ = ["AWSClaudeProvider"]
