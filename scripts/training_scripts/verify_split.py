from datasets import load_dataset, DatasetDict

def test_split_guidelines():
    DATASET_ID = "yahma/alpaca-cleaned"
    print(f"Loading raw dataset: {DATASET_ID}...")
    raw_dataset = load_dataset(DATASET_ID, split="train")
    total_rows = len(raw_dataset)
    print(f"Total starting rows: {total_rows}\n")

    # Execute the splits
    train_testvalid = raw_dataset.train_test_split(test_size=0.20, seed=42)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.50, seed=42)

    dataset = DatasetDict({
        "train": train_testvalid["train"],
        "validation": test_valid["train"],
        "test": test_valid["test"]
    })

    # Verify the math
    print("=== Split Verification ===")
    for split_name, data in dataset.items():
        split_length = len(data)
        actual_percentage = (split_length / total_rows) * 100
        print(f"[{split_name.capitalize()}] Rows: {split_length} | Actual %: {actual_percentage:.2f}%")

if __name__ == "__main__":
    test_split_guidelines()