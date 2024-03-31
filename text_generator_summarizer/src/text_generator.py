from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("allenai/dolma")
import pandas as pd

# Step 1: Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df["input"].tolist(), df["output"].tolist()

# Step 2: Tokenization
def tokenize_data(input_texts, output_texts, tokenizer):
    # Check if padding token is set, if not, set it to the end-of-sequence token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_tokenized = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    output_tokenized = tokenizer(output_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    return input_tokenized, output_tokenized

# Step 3: Model Training
def train_model(input_tokenized, output_tokenized):
    model_name = "gpt2"  # You can specify different GPT-2 model sizes here
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir="./models/chatbot_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=input_tokenized,
        eval_dataset=output_tokenized,  # You can use the same dataset for evaluation
    )
    trainer.train()
    return model

# Step 4: Generate Text
def generate_response(model, tokenizer, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")  # Tokenize input text
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, early_stopping=True)  # Generate text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)  # Decode the generated text
    return generated_text

if __name__ == "__main__":
    file_path = "../data/chatbot_data.csv"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input_texts, output_texts = load_data(file_path)
    input_tokenized, output_tokenized = tokenize_data(input_texts, output_texts, tokenizer)
    model = train_model(input_tokenized, output_tokenized)

    # Test the trained model
    input_text = "Hello!"
    response = generate_response(model, tokenizer, input_text)
    print("User Input:", input_text)
    print("Model Response:", response)




# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# def generate_text(input_text, max_length=50):
#     model_name = "gpt2"  # You can specify different GPT-2 model sizes here
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2LMHeadModel.from_pretrained(model_name)

#     input_ids = tokenizer.encode(input_text, return_tensors="pt")  # Tokenize input text
#     output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, early_stopping=True)  # Generate text

#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)  # Decode the generated text
#     return generated_text

# if __name__ == "__main__":
#     input_text = "Once upon a time"  # Initial prompt for text generation
#     generated_text = generate_text(input_text)
#     print("Generated Text:", generated_text)



# from data_processing import process_chat_data
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers import Trainer, TrainingArguments

# def train_chatbot_model():
#     # Process the chatbot data
#     input_texts, target_texts = process_chat_data("../data/chatbot_data.csv")
#     if input_texts is None or target_texts is None:
#         return

#     # Initialize tokenizer and model
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token

#     # Tokenize inputs and targets
#     tokenized_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
#     tokenized_targets = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")

#     # Initialize model
#     model = GPT2LMHeadModel.from_pretrained("gpt2")

#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir="./models/chatbot_model",
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         save_steps=1000,
#         save_total_limit=2,
#     )

#     # Initialize trainer and train the model
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_inputs,  # Use tokenized inputs for training
#         eval_dataset=tokenized_inputs,   # Use tokenized inputs for evaluation
#         tokenizer=tokenizer,  # Pass the tokenizer to the trainer
#     )
#     trainer.train()
#     print("[INFO] Model trained successfully.")
#     trainer.save_model("./models/chatbot_model")

# if __name__ == "__main__":
#     train_chatbot_model()




# from data_processing import process_chat_data
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers import Trainer, TrainingArguments

# import torch


# def get_data():
#     try:
#         # import pdb;pdb.set_trace()
#         data_path = "../data/chatbot_data.csv"
#         input_texts, target_texts = process_chat_data(data_path)
#         return input_texts, target_texts
#     except Exception as e:
#         return {"Error" : str(e)}

# def train_text_gen(model):
#     input_texts, target_texts = get_data()

#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token

#     # Tokenize input and target texts
#     inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
#     targets = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)["input_ids"]


#     print("[INFO] Training model...")
#     training_args = TrainingArguments(
#         output_dir="./models/chatbot_model",
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=4,
#         save_steps=1000,
#         save_total_limit=2,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=inputs,
#         eval_dataset=inputs,
#     )
#     trainer.train()
#     print("[INFO] Model trained successfully.")
#     trainer.save_model("./models/text_generator_model")


# if __name__ == "__main__":
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#     train_text_gen(model)




