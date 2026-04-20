"""
Optional LoRA Fine-Tuning Script for SanchitAI.

Generates Q&A training pairs from personal_data.json and knowledge base,
then fine-tunes a small model (Phi-3 mini or Llama 3.2 3B) using LoRA via Unsloth.

Requirements (install separately, GPU recommended):
    pip install unsloth transformers datasets peft trl accelerate bitsandbytes

Usage:
    python scripts/fine_tune.py --model phi3 --output ./models/sanchitai-ft
    python scripts/fine_tune.py --model llama3.2 --output ./models/sanchitai-ft

After fine-tuning:
    1. Export to GGUF: the script saves a GGUF file for Ollama
    2. Import into Ollama: ollama create sanchitai -f ./models/sanchitai-ft/Modelfile
    3. Update config.py: OLLAMA_MODEL = "sanchitai"

Note: Without a GPU, this can take many hours. For most use cases, RAG alone
(without fine-tuning) already provides excellent personalized responses.
"""

import json
import logging
import sys
import argparse
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ----- Training Data Generation -----

def generate_qa_pairs_from_personal_data() -> List[Dict]:
    """Generate Q&A training pairs from personal_data.json."""
    path = config.PERSONAL_DATA_PATH
    with open(path) as f:
        data = json.load(f)

    pairs = []

    # Use FAQs directly
    for faq in data.get("frequently_asked_questions", []):
        if faq.get("question") and faq.get("answer"):
            pairs.append({"question": faq["question"], "answer": faq["answer"]})

    # Identity and basic facts
    identity = data.get("identity", {})
    ai_info = data.get("ai_assistant", {})
    if ai_info.get("purpose"):
        pairs.append({
            "question": "Who are you?",
            "answer": ai_info["purpose"]
        })
    if ai_info.get("created_by"):
        pairs.append({
            "question": "Who built you / who created you?",
            "answer": f"I was built by {ai_info['created_by']}. {ai_info.get('purpose', '')}"
        })

    # Education
    for edu in data.get("education", []):
        pairs.append({
            "question": f"Where did Sanchit study for his {edu.get('degree', 'degree')}?",
            "answer": f"Sanchit pursued his {edu.get('degree')} at {edu.get('institution')} ({edu.get('period', '')}). {', '.join(edu.get('highlights', []))}"
        })

    # Experience
    for exp in data.get("experience", []):
        pairs.append({
            "question": f"Tell me about Sanchit's role at {exp.get('organization', '')}",
            "answer": f"Sanchit worked as {exp.get('title')} at {exp.get('organization')} from {exp.get('period')}. Key work: {'. '.join(exp.get('highlights', []))}"
        })

    # Projects
    for proj in data.get("projects", []):
        pairs.append({
            "question": f"What is {proj.get('name', '')}?",
            "answer": f"{proj.get('name')} — {proj.get('description', '')} Significance: {proj.get('significance', '')} Technologies: {', '.join(proj.get('technologies', []))}"
        })

    # Opinions
    opinions = data.get("opinions", {})
    for key, val in opinions.items():
        if val and not val.startswith("ADD_"):
            pairs.append({
                "question": f"What is Sanchit's opinion on {key.replace('_', ' ')}?",
                "answer": val
            })

    # Awards
    awards = data.get("awards_and_recognition", [])
    if awards:
        pairs.append({
            "question": "What awards and recognition has Sanchit received?",
            "answer": "Sanchit has received: " + "; ".join(awards)
        })

    # Philosophy
    quotes = data.get("philosophy_and_quotes", [])
    if quotes:
        pairs.append({
            "question": "What is Sanchit's philosophy or life motto?",
            "answer": "Sanchit believes: " + " ".join(quotes)
        })

    logger.info(f"Generated {len(pairs)} Q&A training pairs")
    return pairs


def format_for_training(pairs: List[Dict]) -> List[Dict]:
    """Format Q&A pairs as instruction-following examples."""
    system = (
        "You are SanchitAI, a personal AI assistant built by Sanchit Minocha. "
        "Answer questions about Sanchit's career, research, projects, and personality warmly and accurately."
    )
    formatted = []
    for pair in pairs:
        formatted.append({
            "instruction": pair["question"],
            "input": "",
            "output": pair["answer"],
            "text": (
                f"<|system|>\n{system}\n"
                f"<|user|>\n{pair['question']}\n"
                f"<|assistant|>\n{pair['answer']}"
            )
        })
    return formatted


def save_training_data(pairs: List[Dict], output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    formatted = format_for_training(pairs)

    jsonl_path = output_path / "train.jsonl"
    with open(jsonl_path, "w") as f:
        for item in formatted:
            f.write(json.dumps(item) + "\n")

    json_path = output_path / "train.json"
    with open(json_path, "w") as f:
        json.dump(formatted, f, indent=2)

    logger.info(f"Saved {len(formatted)} training examples to {output_path}")
    return jsonl_path


# ----- Fine-Tuning with Unsloth -----

MODEL_MAP = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama3.2": "unsloth/Llama-3.2-3B-Instruct",
    "mistral": "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
}


def fine_tune(model_key: str, output_dir: Path, training_data_path: Path):
    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        logger.error(
            "Fine-tuning dependencies not installed.\n"
            "Install with: pip install unsloth transformers datasets peft trl accelerate bitsandbytes"
        )
        sys.exit(1)

    model_name = MODEL_MAP.get(model_key, model_key)
    logger.info(f"Loading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset("json", data_files=str(training_data_path), split="train")
    logger.info(f"Training dataset: {len(dataset)} examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            output_dir=str(output_dir / "checkpoints"),
        ),
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save model
    logger.info(f"Saving fine-tuned model to {output_dir}...")
    model.save_pretrained(str(output_dir / "lora_model"))
    tokenizer.save_pretrained(str(output_dir / "lora_model"))

    # Export to GGUF for Ollama
    logger.info("Exporting to GGUF for Ollama...")
    try:
        model.save_pretrained_gguf(str(output_dir / "gguf"), tokenizer, quantization_method="q4_k_m")

        # Write Ollama Modelfile
        modelfile = output_dir / "Modelfile"
        modelfile.write_text(
            f'FROM {output_dir}/gguf/model-q4_k_m.gguf\n'
            f'SYSTEM "You are SanchitAI, a personal assistant built by Sanchit Minocha..."\n'
            f'PARAMETER temperature 0.7\n'
            f'PARAMETER num_predict 1024\n'
        )
        logger.info(f"Modelfile written to {modelfile}")
        logger.info(f"\nTo use in Ollama:\n  ollama create sanchitai -f {modelfile}")
        logger.info("Then update config.py: OLLAMA_MODEL = 'sanchitai'")
    except Exception as e:
        logger.warning(f"GGUF export failed: {e}. Using HuggingFace format instead.")

    logger.info("Fine-tuning complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SanchitAI on personal data")
    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        default="phi3",
        help="Base model to fine-tune (default: phi3)"
    )
    parser.add_argument(
        "--output",
        default="./models/sanchitai-ft",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only generate training data, don't run fine-tuning"
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    data_dir = output_dir / "training_data"

    # Generate training data
    pairs = generate_qa_pairs_from_personal_data()
    training_data_path = save_training_data(pairs, data_dir)

    if args.data_only:
        logger.info(f"Training data saved to {data_dir}. Skipping fine-tuning.")
        return

    fine_tune(args.model, output_dir, training_data_path)


if __name__ == "__main__":
    main()
