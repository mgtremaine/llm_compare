# llm_compare

Python script to compare LLM prompt responses. Uses Google, OpenAI, and Ollama interfaces.

## Description

`llm_compare` is a Python script designed to parse a prompt and use multiple Large Language Models (LLMs) via Google's Generative AI API, OpenAI, and Ollama models. It supports processing text prompts and optionally includes image data to enhance the responses.

## Features

- Supports multiple LLMs: Google Generative AI, OpenAI, and Ollama models.
- Processes text prompts and optionally includes image data.
- Configurable via a JSON file.
- Verbose output option for detailed response information.

## Requirements

- Python 3.6+
- Libraries: `google-generativeai`, `Pillow`, `openai`, `ollama`, `pdf2image`

## Installation

Install the required libraries using pip:

```sh
pip3 install google-generativeai Pillow openai ollama pdf2image# llm_compare
```

## Usage
Run the script with the following command:

Arguments
-p, --prompt: Prompt for the AI models (required).
-i, --image_file: Path to the image file (optional).
-c, --config: Path to the config JSON file (default: arena_config.json).
-v, --verbose: Enable verbose output (optional).

## Example
```
python3 arena_prompt.py -p "What is the capital of France?" -c arena_config.json -v
```

## Configuration
Create a configuration file (arena_config.json) with the following structure:
```
{
    "GOOGLE_KEY": "<INSERT KEY HERE>",
    "GOOGLE_MODEL": "gemini-1.5-flash",
    "OPENAI_KEY": "<INSERT KEY HERE>",
    "OPENAI_MODEL": "gpt-4o",
    "OLLAMA_MODEL": ["llava", "llama3"]
}
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
Mike Tremaine mgt@stellarcore.net


This README.md provides a comprehensive overview of the project, including its description, features, requirements, installation instructions, usage, configuration, license, and author information.
