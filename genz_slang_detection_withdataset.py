

# Tokenize the dataset and add labels
def tokenize_function(examples):
    # Tokenize the input and set labels to match input_ids for loss calculation
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)
    inputs["labels"] = inputs["input_ids"].copy()  # Set labels to be the same as input_ids
    return inputs
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set pad token (GPT-2 does not have a default pad token)
tokenizer.pad_token = tokenizer.eos_token  

# Load dataset (change to your dataset if different)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenize the dataset and add labels
def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)
    inputs["labels"] = inputs["input_ids"].copy()  # Labels are same as input_ids for causal LM
    return inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)



tokenized_dataset = dataset.map(tokenize_function, batched=True)
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
)

# Initialize the Trainer with tokenized dataset containing labels
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune the model
trainer.train()
# Save the fine-tuned model and tokenizer
trainer.save_model("./results")  # This will save in the ./results directory
tokenizer.save_pretrained("./results")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer from the saved directory
model = GPT2LMHeadModel.from_pretrained("./results")
tokenizer = GPT2Tokenizer.from_pretrained("./results")

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token to avoid padding issues

# Ensure the model is in evaluation mode
model.eval()
import torch

def generate_text(input_text, max_new_words=2):
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate predictions with limited new words
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_words,  # Generate only the next 2 tokens
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )

    # Decode and print the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Custom text input
custom_text = input("Enter your text: ")

# Generate and display the prediction
predicted_text = generate_text(custom_text, max_new_words=2)  # Only next 2 words
print("Input Text:", custom_text)
print("Generated Text:", predicted_text)

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()  # Set model to evaluation mode
def generate_text(input_text, max_length=50):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
def calculate_accuracy(predictions, references):
    correct = sum([1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip()])
    accuracy = correct / len(references)
    return accuracy * 100  # Convert to percentage
def calculate_bleu(predictions, references):
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        reference = [ref.split()]  # BLEU expects references to be a list of lists
        candidate = pred.split()
        bleu_scores.append(sentence_bleu(reference, candidate))
    return sum(bleu_scores) / len(bleu_scores)
def calculate_perplexity(texts):
    perplexities = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())
    return sum(perplexities) / len(perplexities)
def calculate_wer(predictions, references):
    wer_scores = [wer(ref, pred) for ref, pred in zip(references, predictions)]
    return sum(wer_scores) / len(wer_scores)  # Average WER
# Sample test data (input text and reference output)
test_data = [
    ("Hey, can u pls gimme", "hey can you gimme a little"),
    ("Hey there", "Hey there, I"),
    ("Whats the", "Whats the deal?"),
]

# Separate inputs and references
inputs = [item[0] for item in test_data]
references = [item[1] for item in test_data]

# Generate predictions
predictions = [generate_text(input_text) for input_text in inputs]

# Calculate each metric
accuracy = calculate_accuracy(predictions, references)
bleu_score = calculate_bleu(predictions, references)
perplexity = calculate_perplexity(inputs)  # Calculate on inputs only
wer_score = calculate_wer(predictions, references)

# Print the results
print(f"Accuracy: {accuracy:.2f}%")
print(f"BLEU Score: {bleu_score:.2f}")
print(f"Perplexity: {perplexity:.2f}")
print(f"Word Error Rate (WER): {wer_score:.2f}")
import matplotlib.pyplot as plt

# Define metric names and their corresponding values
metrics = ['Accuracy (%)', 'BLEU Score', 'Perplexity', 'WER (%)']
values = [accuracy, bleu_score * 100, perplexity, wer_score]  # BLEU is scaled to percentage for visual consistency

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])

# Add labels and title
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Metrics for Gen Z Typing Assistant')

# Display values above each bar
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.2f}", ha='center', fontweight='bold')

plt.show()

#BERT

from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()  # Set model to evaluation mode
def predict_masked_word(text):
    # Tokenize input and find the [MASK] token index
    inputs = tokenizer(text, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the top predicted token for the masked position
    mask_logits = logits[0, mask_token_index, :]
    top_token_id = torch.argmax(mask_logits, dim=1)
    # The following line is changed to return a string instead of a list
    predicted_word = tokenizer.decode(top_token_id)

    return predicted_word
    # Custom input text with [MASK] token
text = "[MASK] we?"

# Get the top prediction for the masked token
predicted_word = predict_masked_word(text)
predicted_text = text.replace("[MASK]", predicted_word)

# Display the result
print("Input Text:", text)
print("Predicted Text:", predicted_text)
from transformers import BertTokenizer, BertForMaskedLM
import torch

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()  # Set to evaluation mode
import nltk
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer

# Define a function to predict the masked word
def predict_masked_word(text, top_k=1):
    inputs = tokenizer(text, return_tensors="pt")
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the top predicted tokens for the masked position
    mask_logits = logits[0, mask_token_index, :]
    top_k_ids = torch.topk(mask_logits, top_k).indices
    predicted_words = [tokenizer.decode([token_id]) for token_id in top_k_ids[0]]

    return predicted_words  # Return a list of top-k predictions
# Example test data with masked sentences and expected target words
test_data = [
    {"text": "Hey, can you [MASK] me some help?", "target": "give"},
    {"text": "This is a [MASK] example.", "target": "simple"},
    {"text": "Let's [MASK] this problem together.", "target": "solve"},{"text": "What do u [MASK]?", "target": "mean"},
    {"text": "Would u wanna [MASK]?", "target": "come"},
    {"text": "whats the [MASK]?", "target": "matter"},{"text": "less go get [MASK].", "target": "him"},
    {"text": "say what [MASK].", "target": "happened"},{"text": "[MASK] we?", "target": "shall"}
]

# Initialize counters for metrics
correct_predictions = 0
top_k_correct_predictions = 0
total_masks = len(test_data)
bleu_scores = []
perplexities = []
wer_scores = []

# Evaluate each example in the test data
for item in test_data:
    text, target = item["text"], item["target"]

    # Get the top prediction for the masked word
    predicted_words = predict_masked_word(text, top_k=3)  # Top-K predictions
    top_prediction = predicted_words[0]  # Top prediction for exact match accuracy

    # Calculate Exact Match Accuracy
    if top_prediction == target:
        correct_predictions += 1

    # Calculate Top-K Accuracy
    if target in predicted_words:
        top_k_correct_predictions += 1

    # Calculate BLEU Score
    reference = [target.split()]  # BLEU expects references as list of lists
    candidate = top_prediction.split()
    bleu_scores.append(sentence_bleu(reference, candidate))

    # Calculate Perplexity
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        perplexity = torch.exp(outputs.loss)
    perplexities.append(perplexity.item())

    # Calculate Word Error Rate (WER)
    target_sentence = text.replace("[MASK]", target)
    predicted_sentence = text.replace("[MASK]", top_prediction)
    wer_scores.append(wer(target_sentence, predicted_sentence))

# Calculate final averages for metrics
accuracy = (correct_predictions / total_masks) * 100  # Exact match accuracy in percentage
average_bleu = sum(bleu_scores) / len(bleu_scores)  # Average BLEU score
average_perplexity = sum(perplexities) / len(perplexities)  # Average perplexity
average_wer = sum(wer_scores) / len(wer_scores)  # Average WER

# Print results
print(f"Accuracy: {accuracy:.2f}%")
print(f"BLEU Score: {average_bleu:.2f}")
print(f"Perplexity: {average_perplexity:.2f}")
print(f"Word Error Rate (WER): {average_wer:.2f}")
