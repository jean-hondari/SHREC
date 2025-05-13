
import transformers 
# from transformers import transformers.AutoConfig, transformers.AutoModelForCausalLM, transformers.AutoTokenizer, transformers.Trainer, transformers.TrainingArguments, transformers.BitsAndBytesConfig,  transformers.TrainerCallback, transformers.Trainer
print('transformers import complete')
from datasets import DatasetDict, load_dataset, Dataset

import os
import logging
from typing import Optional, Union, Dict, Any, Tuple
from huggingface_hub import upload_file
import torch
from sklearn.model_selection import train_test_split
import pdb
print('import complete')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_name: str, trust_remote_code: bool = True, device_map= "auto", quantization_type: str = "bitsandbytes_8bit", cache_dir: str = "./deepseek", output_dir: str = "./fine_tuned_deepseek"):
        # device_map: Optional[Union[str, Dict[str, Any]]] = "auto"
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.device_map = device_map
        self.quantization_type = quantization_type
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None

    def _load_quantization_config(self):
        """
        Load the appropriate quantization configuration based on the desired quantization type.
        """
        try:
            if self.quantization_type == "bitsandbytes_8bit":
                return transformers.BitsAndBytesConfig(load_in_8bit=True)
            else:
                logger.warning(f"Quantization type '{self.quantization_type}' is not directly supported. Proceeding without quantization.")
                return None
        except Exception as e:
            logger.error(f"Error in loading quantization configuration: {str(e)}")
            return None

    def _load_model_with_fallback(self):
        """
        Load the model with a fallback mechanism in case quantization is not supported.
        """
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computations
            bnb_4bit_use_double_quant=True,  # Enable double quantization
        )
        try:
            # Try loading the model with the specified quantization config
            # quantization_config = self._load_quantization_config()
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                trust_remote_code=self.trust_remote_code,
                device_map=self.device_map, #{"": "cpu"},
                offload_state_dict=False,  # Disable offloading                
                cache_dir=self.cache_dir
            )
            logger.info("Model loaded successfully with quantization.")
            return model
        except ValueError as e:
            if "Unknown quantization type" in str(e):
                logger.warning("Quantization type not supported directly. Attempting to load without quantization.")
                
                # Fallback: Remove quantization from config
                config = transformers.AutoConfig.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
                if hasattr(config, "quantization_config"):
                    delattr(config, "quantization_config")

                try:
                    # Try loading the model without quantization
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        config=config,
                        trust_remote_code=self.trust_remote_code,
                        offload_state_dict=False,  # Disable offloading
                        device_map=self.device_map, #{"": "cpu"},
                        cache_dir=self.cache_dir
                    )
                    logger.info("Model loaded successfully without quantization.")
                    return model
                except Exception as inner_e:
                    logger.error(f"Failed to load model without quantization: {str(inner_e)}")
                    raise
            else:
                logger.error(f"Unexpected error during model loading: {str(e)}")
                raise

    def _load_tokenizer(self):
        """
        Load the tokenizer for the model.
        """
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=self.trust_remote_code,
                offload_state_dict=False,  # Disable offloading
                cache_dir=self.cache_dir)
            logger.info("Tokenizer loaded successfully.")
            return tokenizer
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            raise

    def load(self) -> Tuple:
        """
        Load both model and tokenizer with fallbacks and quantization handling.
        """
        logger.info("Starting model and tokenizer loading...")
        
        # Load the model with fallback if needed
        self.model = self._load_model_with_fallback()
        # Ensure the model is fully on the GPU (or CPU if no GPU is available)
        # self.model = self.model.to("cpu")
        # Load the tokenizer
        self.tokenizer = self._load_tokenizer()

        return self.model, self.tokenizer

    def fine_tune(self, dataset_name: str, output_dir: str = None):
        """
        Fine-tune the model on the specified dataset.
        """
        if output_dir is None:
            output_dir = self.output_dir

        logger.info("Loading dataset...")
        dataset = load_dataset(dataset_name)

        # If the dataset has no validation split, split the train set
        if "validation" not in dataset:
            train_data = dataset["train"]
            train_data = [example for example in train_data if example["source"] != "..." and example["target"] != "..."]

            train_data, val_data = train_test_split(train_data, test_size=0.1)
            
            # Convert train_data and val_data to the correct dictionary format
            train_data_dict = {
                "source": [item["source"] for item in train_data],
                "target": [item["target"] for item in train_data],
            }

            val_data_dict = {
                "source": [item["source"] for item in val_data],
                "target": [item["target"] for item in val_data],
            }

            # Create DatasetDict with the correct format
            dataset = DatasetDict({
                'train': Dataset.from_dict(train_data_dict),
                'validation': Dataset.from_dict(val_data_dict),
            })

        # Preprocess the data

        dataset = self.preprocess_data(dataset)

        # Set up training arguments
        
        training_args = transformers.TrainingArguments(
            evaluation_strategy="epoch",
            num_train_epochs=3,
            output_dir=output_dir,
            per_device_train_batch_size=2,  
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,  
            dataloader_num_workers=4,  
            dataloader_pin_memory=True,  
            fsdp="full_shard auto_wrap",  # Enables Fully Sharded Data Parallelism
            fsdp_config={"offload_params": True},  # Offloads parameters to CPU
            no_cuda=True  # Forces CPU execution
        )

        # Initialize transformers.Trainer
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
        )

        # Start training
        trainer.train()

        # Save the fine-tuned model
        logger.info("Saving fine-tuned model...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def upload_to_hub(self, repo_id: str, token: str):
        """
            Upload the model to Hugging Face Model Hub after ensuring it's on the same device.
        """
        logger.info("Ensuring the model is fully loaded on the device...")

        # Ensure the model is on a single device (either GPU or CPU)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")

        logger.info("Uploading model to Hugging Face Model Hub...")
        self.model.push_to_hub(repo_id=repo_id, use_auth_token=token)
        self.tokenizer.push_to_hub(repo_id=repo_id, use_auth_token=token)
        logger.info("Model uploaded successfully to Hugging Face Hub.")


    def evaluate(self, dataset_name: str):
        """Evaluate the model on a test dataset."""
        logger.info("Evaluating model...")
        dataset = load_dataset(dataset_name)
        eval_dataset = dataset["test"]  # Assuming the dataset has a 'test' split

        # Use the transformers.Trainer to evaluate the model
        trainer = transformers.Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Evaluate on test dataset
        results = trainer.evaluate(eval_dataset=eval_dataset)

        # For classification tasks, accuracy might be available directly
        logger.info(f"Evaluation results: {results}")
        return results

    def generate(self, prompt: str):
        """Generate a response using the fine-tuned model based on the given prompt."""
        logger.info(f"Generating response for prompt: {prompt}")
        
        # Generate a response from the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        output = self.model.generate(inputs['input_ids'], max_length=4096, num_return_sequences=1)

        # Decode the output
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Response: {response}")
        return response

    def preprocess_data(self, dataset):
        """Preprocess the dataset for training and evaluation."""
        # Map the "source" and "target" columns into inputs and labels for training
        def preprocess_function(examples):
            # Ensure that "source" is the input and "target" is the label
            inputs = examples["source"]
            targets = examples["target"]
            model_inputs = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=2048)
            labels = self.tokenizer(targets, padding="max_length", truncation=True, max_length=2048)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Apply preprocessing
        dataset = dataset.map(preprocess_function, batched=True)
        return dataset

if __name__ == "__main__":
    
    deepseek = ModelLoader(model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B") #"deepseek-ai/DeepSeek-R1")
    model, tokenizer = deepseek.load()
    deepseek.test_model_with_prompt("hello")
