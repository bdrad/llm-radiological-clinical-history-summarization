# Experiment configurations
EXPERIMENT_CONFIGS = {
    # Prompting strategy ablation
    "zero_shot_standard": {
        "prompting_strategy": "zero_shot",
        "prompt_type": "standard",
        "temperature": 0, 
        "description": "Zero-shot prompting with standard prompt"
    },
	"zero_shot_augmented": {
        "prompting_strategy": "zero_shot",
        "prompt_type": "augmented", 
        "temperature": 0,
        "description": "One-shot prompting with augmented comprehensiveness rules"
    },
    "one_shot_standard": {
        "prompting_strategy": "one_shot", 
        "prompt_type": "standard",
        "temperature": 0,
        "description": "One-shot prompting with standard prompt"
    },
	"one_shot_augmented": {
        "prompting_strategy": "one_shot", 
        "prompt_type": "augmented",
        "temperature": 0,
        "description": "One-shot prompting with standard prompt"
    },
    "temp_0.0": {
        "prompting_strategy": "zero_shot",
        "prompt_type": "standard",
        "temperature": 0,
        "description": "Temperature 0.0 (deterministic)",
        "requires_temperature_support": True
    },
    "temp_0.5": {
        "prompting_strategy": "zero_shot", 
        "prompt_type": "standard",
        "temperature": 0.5,
        "description": "Temperature 0.5 (moderate variability)",
        "requires_temperature_support": True
    },
    "temp_1.0": {
        "prompting_strategy": "zero_shot",
        "prompt_type": "standard",
        "temperature": 1,
        "description": "Temperature 1.0 (high variability)",
        "requires_temperature_support": True
    }
}

# Model configurations
MODEL_CONFIGS = {
    # AWS Claude models
    "claude3_5": {
        "provider": "anthropic_aws",
        "model_name": "claude3_5",
        "max_tokens": 200,
    },
    
    # Azure OpenAI models
    "gpt4o": {
        "provider": "openai_azure",
        "model_name": "gpt4o",
        "max_tokens": 200,
    },
    "gpt4o_mini": {
        "provider": "openai_azure",
        "model_name": "gpt4o_mini",
        "max_tokens": 200,
    },
    "o1": {
        "provider": "openai_azure",
        "model_name": "o1",
        "max_tokens": 500,
    },
    "o1_mini": {
        "provider": "openai_azure",
        "model_name": "o1_mini",
        "max_tokens": 500,
    },
    
    # HuggingFace models
    "llama3_1_8b": {
        "provider": "huggingface",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "max_tokens": 200,
    },
    "mistral_7b": {
        "provider": "huggingface",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_tokens": 200,
    },
    "biomistral_7b": {
        "provider": "huggingface",
        "model_name": "BioMistral/BioMistral-7B",
        "max_tokens": 200,
    },
    "meditron3_8b": {
        "provider": "huggingface",
        "model_name": "OpenMeditron/Meditron3-8B",
        "max_tokens": 200,
    },
    "qwen2_5_7b": {
        "provider": "huggingface",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_tokens": 200,
    },
    "deepseek_r1_llama_8b": {
        "provider": "huggingface",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "max_tokens": 500,
    },
    "deepseek_r1_qwen_7b": {
        "provider": "huggingface",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "max_tokens": 500,
    }
}