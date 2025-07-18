from utils import init_wandb, get_device, load_artifact_path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_VERSION = 'v2'

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        init_wandb()
        device = get_device()
        fine_tuned_path = load_artifact_path('base_lora_weights_6', MODEL_VERSION)

        # Load the base Qwen model
        qwen_name = "Qwen/Qwen3-0.6B-Base"
        qwen_model = AutoModelForCausalLM.from_pretrained(qwen_name)
        qwen_model.to(device)
        qwen_model.eval()

        # Load the fine-tuned Lora model
        base_model = AutoModelForCausalLM.from_pretrained(qwen_name)
        fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_path)
        fine_tuned_model.to(device)
        fine_tuned_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(qwen_name)

        return qwen_model, fine_tuned_model, tokenizer, device

    except Exception as e:
        st.error(f"Error loading models: {e}")
        raise  # re-raise to stop execution if needed

def generate_summary(prompt, qwen_model, fine_tuned_model, tokenizer, device):
    # Tokenize the input prompt
    prompt = "Summarize: " + prompt + "\n Summary:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=550, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate summary using the qwen model
    with torch.no_grad():
        qwen_ids = qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
        )
        
    # Generate summary using the fine-tuned model
    with torch.no_grad():
        fine_tuned_ids = fine_tuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
        )

    # Get only the newly generated tokens (after the input)
    input_length = input_ids.shape[1]
    fine_tuned_new_tokens = fine_tuned_ids[0][input_length:]
    qwen_new_tokens = qwen_ids[0][input_length:]

    # Decode only the new tokens
    fine_tuned_summary = tokenizer.decode(fine_tuned_new_tokens, skip_special_tokens=True)
    qwen_summary = tokenizer.decode(qwen_new_tokens, skip_special_tokens=True)
    return fine_tuned_summary, qwen_summary

    
def main():
    qwen_model, fine_tuned_model, tokenizer, device = load_models()
    
    st.title("Qwen TLDR Summarizer")

    prompt = st.text_area("Enter text to summarize:", height=300)
    
    if st.button("Generate Summary"):
        if prompt:
            fine_tuned_summary, qwen_summary = generate_summary(prompt, qwen_model, fine_tuned_model, tokenizer, device)
            st.subheader("Fine-tuned Model Summary:")
            st.write(fine_tuned_summary)
            st.subheader("Qwen Model Summary:")
            st.write(qwen_summary)
        else:
            st.error("Please enter some text to summarize.")

if __name__ == "__main__":
    main()