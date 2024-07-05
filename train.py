"""
This script is used to train the tokenizer model using the combined content of the dataset.
The trained model is saved in the tokenizer_model directory.
"""

import os
from tokenizer.basic_bpe import BasicTokenizer

# reading the input file
combined_file_path = "hindi_combined.txt"
combined_file_path = os.path.join(os.getcwd(), "data", combined_file_path)
with open(combined_file_path, "r") as file:
    combined_content = file.read()

basic_tokenizer = BasicTokenizer()
print("Training the tokenizer model...")
basic_tokenizer.train(combined_content, vocab_size=5000, verbose=True)

print("Saving the model...")
model_path = os.path.join(os.getcwd(), "tokenizer_model")
os.makedirs(model_path, exist_ok=True)
prefix = os.path.join(model_path, "hindi_sentiments_basic")
basic_tokenizer.save(prefix)
print("Model saved at:", prefix)
