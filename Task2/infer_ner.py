import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained NER model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained NER model directory")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text for inference")
    return parser.parse_args()

def main():
    args = parse_args()

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # Tokenization
    tokens = tokenizer(args.text.split(),
                       is_split_into_words=True,
                       return_tensors="pt",
                       truncation=True,
                       padding="max_length",
                       max_length=128)

    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**tokens)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Recover words and their tags
    word_ids = tokens["input_ids"][0].cpu().numpy()
    pred_ids = predictions[0].cpu().numpy()

    # Load the list of tags
    label_list = model.config.id2label

    decoded_tokens = tokenizer.convert_ids_to_tokens(word_ids)
    results = []
    for token, label_id in zip(decoded_tokens, pred_ids):
        label = label_list[label_id]
        if token not in ["[CLS]", "[SEP]", "[PAD]"]:
            results.append((token, label))
    
    print("Extracted entities:")
    for token, label in results:
        print(f"{token} -> {label}")

if __name__ == "__main__":
    main()
