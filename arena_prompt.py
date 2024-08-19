#!/usr/bin/env python3

# arena_prompt.py - A simple script to parse a prompt and use multiple LLM via Google's Generative AI API, OpenAI, and Ollama models

# Usage: python3 arena_prompt.py <image_file>
# Requires: Python 3.6+, google-generativeai, Pillow, openai, ollama
# Install: pip3 install google-generativeai Pillow openai ollama
# Author: Mike Tremaine <mgt@stellarcore.net>
# License: MIT

# Import libraries
import os, sys, json, re
#Our AIs
import google.generativeai as genai
import openai
import ollama
#Image processing
import PIL.Image, io
from pdf2image import convert_from_path, convert_from_bytes
#Support
from datetime import datetime
import argparse
import base64

# Parse command line arguments
parser = argparse.ArgumentParser(description='A simple script to parse a prompt and use multiple LLM via Google\'s Generative AI API, OpenAI, and Ollama models')
parser.add_argument('-p', '--prompt', type=str, help='Prompt for the AI models')
parser.add_argument('-i', '--image_file', type=str, help='Path to the image file', nargs='?')
parser.add_argument('-c', '--config', type=str, default='arena_config.json', help='Path to the config json file (default: arena_config.json)')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()

# Import image file from command line
image_file = args.image_file

#Import json from config_path
def load_json_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

# Load the config file
if not os.path.exists(args.config):
    print(f"Error: Config file {args.config} does not exist.")
    sys.exit(1)

config = load_json_config(args.config)

#Set Models from Config
GOOGLE_API_KEY = config['GOOGLE_KEY']
GOOGLE_MODEL =  config['GOOGLE_MODEL'] 
OPENAI_API_KEY = config['OPENAI_KEY']
OPENAI_MODEL = config['OPENAI_MODEL']
#OLLAMA_MODELS is json array
OLLAMA_MODELS = config["OLLAMA_MODEL"]

if 'SYSTEM' in config:
    SYSTEM = config['SYSTEM']
else:
    # Set default SYSTEM
    SYSTEM = """You are an expert assist.
        Answer the question based on the context. Be concise and do not hallucinate. 
        Respond with the information you have in polite and professional manner. """

# Detect file type
image = None
if image_file is not None and os.path.exists(image_file):
    if image_file.endswith('.pdf'):
        # Returns a list for each page
        image = convert_from_path(image_file, dpi=200, fmt="jpeg", jpegopt={"quality": 100}, thread_count=4)
        print("PDF file detected. Converting to image...")
    else:
        image = [PIL.Image.open(image_file)] #make sure it is 200dpi
        image = image[0].convert('RGB')
        image = image.resize((200,200))
        print("Image found Converting to RGB  200dpi...")


def get_google_response(prompt, image=None):
    #Initialize the API
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(GOOGLE_MODEL)
    # Call the API
    if image is None:
        response = model.generate_content([prompt])
    else:
        response = model.generate_content([prompt, image])
    return response

def get_openai_response(prompt, image=None):
    if image is None:
        #If no image, just call the API
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config['OPENAI_MODEL'],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response
    else:
        #image is PIL image make base_64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        base64_image = img_byte_arr.getvalue()

        #Initialize the API
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        #Call the API
        try:
            response = client.chat.completions.create(
                model=config['OPENAI_MODEL'],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            { "type": "text",
                              "text": prompt
                            },
                            { "type": "image_url",
                              "image_url": {
                                  "url": f"data:image/jpeg;base64,{base64_image}",
                                  "detail": "high"
                              }
                            }
                        ]
                    }
                ]
            )
            return response
        except Exception as e:
            print(f"Error occurred while calling OpenAI API: {e}")
            return None 

def get_ollama_response(prompt, model, image=None):
    # Convert image to bytes if it's a PIL Image
    if isinstance(image, PIL.Image.Image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        image = img_byte_arr.getvalue()

    # Use ollama library to get response
    messages = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]
    if image:
        messages[0]['image'] = {
            'data': base64.b64encode(image).decode('utf-8'),
            'mime_type': 'image/jpeg'
        }

    response = ollama.chat(
        model=model,
        messages=messages,
    )
    return response

#Main
# Prompt
prompt = SYSTEM + args.prompt

# Process with OpenAI Vision (dummy implementation)
response_openai = get_openai_response(prompt, image)
if args.verbose:
    print("OpenAI:")
    print(response_openai)
else:
    if response_openai is None:
        print("OpenAI: Error occurred while calling OpenAI API")
    else :
        print("OpenAI: " + response_openai.choices[0].message.content)

# Process with Google Generative AI
response_google = get_google_response(prompt, image)
if args.verbose:
    print("Google:")
    print(response_google)
else:
    print("Google: " + response_google.candidates[0].content.parts[0].text)

# Process each image with each model
for model in OLLAMA_MODELS:
    # Process with Ollama model
    response_ollama = get_ollama_response(prompt, model, image)
    if args.verbose:
        print(f"Ollama {model}:")
        print(response_ollama)
    else:
        print(f"Ollama {model}: {response_ollama['message']['content']}")