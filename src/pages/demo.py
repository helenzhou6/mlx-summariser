from utils import init_wandb, get_device, load_artifact_path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_VERSION = 'v2'
PROMPT_TEXT = "Not sure if this goes here but I don't know where else to ask. My public high school has an event called \"project day\" every 6 weeks. It's a pass or fail grade, and it does count for a credit. I'm a senior, and this Friday we have to either volunteer for a retirement home, or volunteer for the Salvation Army. Students had an option for which organisation to assist with. However, this information was distributed via English classes, which I'm not a part of. Instead, I received a letter saying I'm signed up to volunteer for the Salvation Army by default as the retirement home had enough volunteers. I don't support the Salvation Army at all. They're one of the worst organisations out there in my opinion. I emailed my teacher asking to switch me for this exact reason, but she never responded. Can my school force me to volunteer or else they'll give me a failing grade? Who do I talk to so I don't have to volunteer for the Salvation Army?"

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

def generate_summary(qwen_model, fine_tuned_model, tokenizer, device):
    # Tokenize the input prompt
    prompt = "Summarize: " + PROMPT_TEXT + "\n Summary:"
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

    st.subheader("Input prompt:")
    prompt = st.text(PROMPT_TEXT)
    
    if st.button("Generate Summary"):
        if prompt:
            fine_tuned_summary, qwen_summary = generate_summary(qwen_model, fine_tuned_model, tokenizer, device)
            st.subheader("Human Written Summary:")
            st.text("My high school is forcing me to volunteer for the Salvation Army, an organisation I do not want to help. Can they do this? How do I avoid this?")
            st.subheader("Qwen Model Summary:")
            st.write(qwen_summary)
            st.subheader("Fine-tuned Model Summary:")
            st.write(fine_tuned_summary)
            
        else:
            st.error("Please enter some text to summarize.")

if __name__ == "__main__":
    main()