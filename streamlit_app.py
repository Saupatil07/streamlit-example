#import altair as alt
#import numpy as np
#import pandas as pd
import streamlit as st
# from unsloth import FastLanguageModel
from transformers import TextStreamer,AutoTokenizer
from peft import AutoPeftModelForCausalLM


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
 # Enable native 2x faster inference
@st.cache(allow_output_mutation=True)
def get_model():
    # model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    # max_seq_length = max_seq_length,
    # dtype = dtype,
    # load_in_4bit = load_in_4bit,
    # )
    model = AutoPeftModelForCausalLM.from_pretrained(
    "Ellight/gemma-2b-bnb-4bit", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("Ellight/gemma-2b-bnb-4bit")
    return tokenizer,model


tokenizer,model = get_model()



st.set_page_config(
    page_title="Your own aiChat!"
)

# Create a header element
st.header("Your own aiChat!")
user_input = st.text_area('Enter Text to Analyze')
button = st.button("Analyze")

if user_input and button :
    #FastLanguageModel.for_inference(model)

    prompt = """
    ### Instruction:
    {}

    ### Response:
    {}"""

    inputs = tokenizer(
    [
        prompt.format(
            user_input, # instruction
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt")

    text_streamer = TextStreamer(tokenizer)
    outputs = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512)

    st.write(tokenizer.batch_decode(outputs))