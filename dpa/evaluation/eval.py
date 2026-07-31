import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def evaluate_text_accuracy(model, tokenizer, dataset, device):
    # Calculates the Clean Data Accuracy (CDA) for a LLM
    model.eval()
    correct_predictions = 0
    total_questions = len(dataset)

    choice_tokens = [" A", " B", " C", " D"]
    choice_ids = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in choice_tokens]

    print("--- Starting Clean Baseline Evaluations ---")

    with torch.no_grad():
        for item in tqdm(dataset, desc="Evaluating MMLU/Clean Data"):

            prompt = (
                f"Question : {item['question']}\n"
                f"A) {item['choice_A']}\n"
                f"B) {item['choice_B']}\n"
                f"C) {item['choice_C']}\n"
                f"D) {item['choice_D']}\n"
                f"Answer:"
            )

            input = tokenizer(prompt, return_tensor="pt").to(device)

            outputs = model(**inputs)

            next_token_logits = outputs.logits[0, -1, :]

            choice_scores = [next_token_logits[token_id].item() for token_id in choice_ids]

            predicted_index = choice_scores.index(max(choice_scores))
            predicted_letter = ["A", "B", "C", "D"][predicted_index]

            if predicted_letter == item['answer']:
                correct_predictions += 1

    cda = (correct_predictions / total_questions) * 100
    return cda

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = ""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    clean_test_data = []

    baseline_cda = evaluate_clean_multiple_choice(model, tokenizer, clean_test_data, DEVICE)
    print(f"\n[Baseline Results]")
    print(f"Clean Data Accuracy (CDA): {baseline_cda:.2f}%")