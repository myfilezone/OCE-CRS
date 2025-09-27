import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
import time
import json
import os
import logging
import re  # Used to sanitize filenames

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model configuration: map model names to local paths ---
# !!! Important: replace the placeholder paths with the actual local locations of your model files !!!
MODELS_TO_PROCESS = {
    "Qwen2.5-3B": "/path/to/Qwen2.5-3B"
}
# ---------------------------------------------------

# --- Dataset configuration ---
# Use a local dataset path
LOCAL_DATASET_PATH = "/path/to/local/dataset"  # <--- Replace with your local dataset path
# DATASET_NAME = "tatsu-lab/alpaca" # No longer pulling from the Hub
DATASET_SPLIT = "train[:50000]"  # Use a smaller sample for faster testing; increase as needed
INPUT_FIELD = "instruction"  # Field containing the instruction in the Alpaca dataset

# --- Validate local dataset path ---
if not os.path.isdir(LOCAL_DATASET_PATH):
    logging.error(f"Local dataset path not found: {LOCAL_DATASET_PATH}")
    logging.error("Please ensure the path is correct and the dataset files are present.")
    exit()
# -------------------------------------

# Inference configuration
MAX_NEW_TOKENS = 4096  # Cap the generated token count

# Output directory configuration
OUTPUT_DIR = 'inference_results'  # Directory to store result files
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the output directory

# --- Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    try:
        device_name = torch.cuda.get_device_name(0)
    except Exception as e:
        logging.warning(f"Could not get GPU name: {e}")
        device_name = "cuda"  # Fallback name
else:
    device = torch.device("cpu")
    device_name = "cpu"
logging.info(f"Using device: {device} ({device_name})")


# --- Load dataset from the local path ---
logging.info(f"Loading dataset from local path '{LOCAL_DATASET_PATH}', split '{DATASET_SPLIT}'...")
try:
    # Pass the local path directly to load_dataset
    dataset = load_dataset(LOCAL_DATASET_PATH, split=DATASET_SPLIT)
    logging.info(f"Dataset loaded successfully from local path with {len(dataset)} samples.")
except Exception as e:
    logging.error(f"Failed to load dataset from local path {LOCAL_DATASET_PATH}: {e}", exc_info=True)
    exit()  # Exit if the dataset fails to load


# --- Main data generation loop ---
total_processed_files = 0
for model_name, model_path in MODELS_TO_PROCESS.items():
    logging.info(f"\n--- Processing model: {model_name} from path: {model_path} ---")

    if not os.path.isdir(model_path):
        logging.error(f"Model path not found: {model_path}. Skipping model {model_name}.")
        continue

    model_inference_data = []

    trust_remote = any(sub in model_name.lower() for sub in ['phi', 'falcon', 'mpt', 'mistral'])
    if trust_remote:
        logging.info(f"Setting trust_remote_code=True for model {model_name}")

    tokenizer = None  # Initialize so they can be cleaned up on failure
    model = None
    try:
        logging.info(f"Loading tokenizer for {model_name} from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                logging.warning(f"Tokenizer for {model_name} has no pad_token_id. Using eos_token_id ({tokenizer.eos_token_id}) instead.")
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                logging.warning(f"Tokenizer for {model_name} has no pad_token_id and no eos_token_id. Setting pad_token_id to 0.")
                tokenizer.pad_token_id = 0 # Fallback, may need adjustment

        logging.info(f"Loading model {model_name} from {model_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote,
            torch_dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        ).to(device)
        model.eval()
        logging.info(f"Model {model_name} loaded to {device}.")

    except Exception as e:
        logging.error(f"Failed to load model or tokenizer for {model_name} from {model_path}: {e}", exc_info=True)
        del tokenizer
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        continue

    processed_count = 0
    for i, sample in enumerate(dataset):
        request_text = sample.get(INPUT_FIELD)

        if not request_text or not isinstance(request_text, str):
            logging.warning(f"Skipping sample {i} for model {model_name} due to empty or invalid request_text.")
            continue

        try:
            input_tokens = tokenizer(request_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            input_token_count = input_tokens['input_ids'].shape[1]
        except Exception as e:
            logging.error(f"Tokenization failed for sample {i} with model {model_name}: {e}. Text: '{request_text[:100]}...'")
            continue

        generation_config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

        inference_time = -1.0
        outputs = None
        try:
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.no_grad():
                    outputs = model.generate(**input_tokens, generation_config=generation_config)
                end_event.record()
                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                start_time = time.time()
                with torch.no_grad():
                     outputs = model.generate(**input_tokens, generation_config=generation_config)
                end_time = time.time()
                inference_time = end_time - start_time

        except torch.cuda.OutOfMemoryError:
             logging.error(f"CUDA OutOfMemoryError for sample {i} with model {model_name}. Input length: {input_token_count}. Skipping.")
             torch.cuda.empty_cache()
             continue
        except Exception as e:
            logging.error(f"Inference failed for sample {i} with model {model_name}: {e}", exc_info=True)
            continue

        try:
            output_token_ids = outputs[0]
            response_only_token_ids = output_token_ids[input_token_count:]
            response_text = tokenizer.decode(response_only_token_ids, skip_special_tokens=True)
            output_token_count = len(response_only_token_ids)

        except Exception as e:
            logging.error(f"Decoding failed for sample {i} with model {model_name}: {e}")
            response_text = "[Decoding Error]"
            output_token_count = -1

        data_point = {
            "request_text": request_text,
            "model_name": model_name,
            "response_text": response_text,
            "device": device_name,
            "inference_time": round(inference_time, 4),
            "input_token_count": input_token_count,
            "output_token_count": output_token_count
        }
        model_inference_data.append(data_point)
        processed_count += 1

        if (processed_count) % 50 == 0:
             logging.info(f"  Model {model_name}: Processed {processed_count}/{len(dataset)} samples.")

    logging.info(f"Finished processing model {model_name}. Generated {processed_count} data points.")

    # --- Save data for the current model ---
    if model_inference_data:
        sanitized_model_name = re.sub(r'[^\w\-]+', '_', model_name)
        output_filename = os.path.join(OUTPUT_DIR, f'inference_data_{sanitized_model_name}.jsonl')

        logging.info(f"Saving data for model {model_name} to {output_filename}...")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                for item in model_inference_data:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')
            logging.info(f"Data successfully saved to: {os.path.abspath(output_filename)}")
            total_processed_files += 1
        except IOError as e:
            logging.error(f"Failed to write data to file {output_filename}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during file writing for {output_filename}: {e}")
    else:
        logging.warning(f"No data generated for model {model_name}, skipping file save.")

    del model
    del tokenizer
    del model_inference_data
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logging.info("CUDA cache cleared.")


# --- Example: load generated data ---
def load_data_from_jsonl(filename):
    """Load data from a .jsonl file."""
    data = []
    if not os.path.exists(filename):
        logging.error(f"Error: File '{filename}' not found for loading example.")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in {filename}: {line.strip()}")
    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}")
        return None
    return data

# Example: load the first successfully processed model and print a preview
logging.info("\n--- Reading data example ---")
first_successful_model_name = None
example_filename = ""
for model_name in MODELS_TO_PROCESS.keys():
     sanitized_model_name = re.sub(r'[^\w\-]+', '_', model_name)
     potential_filename = os.path.join(OUTPUT_DIR, f'inference_data_{sanitized_model_name}.jsonl')
     if os.path.exists(potential_filename):
         first_successful_model_name = model_name
         example_filename = potential_filename  # Store the located file name
         break

if first_successful_model_name:
    logging.info(f"Attempting to load example data from: {example_filename}")
    loaded_data = load_data_from_jsonl(example_filename)
    if loaded_data:
        logging.info(f"Successfully loaded {len(loaded_data)} records from {example_filename}.")
        print(f"\nFirst 2 loaded data points from {first_successful_model_name}:")
        for record in loaded_data[:2]:
            print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        logging.warning(f"Could not load data from {example_filename} for the example.")
else:
    logging.warning("No output files found in the specified directory to load for the example.")


logging.info(f"\n--- Script finished. Processed and saved data for {total_processed_files} models. ---")
