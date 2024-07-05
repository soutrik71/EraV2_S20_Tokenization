import gradio as gr
import os
from tokenizer.basic_bpe import BasicTokenizer

print("Loading the model...")
model_path = os.path.join(os.getcwd(), "tokenizer_model")
model_path = os.path.join(model_path, "hindi_sentiments_basic.model")

basic_tokenizer = BasicTokenizer()
basic_tokenizer.load(model_path)


def test_tokenizer(text):
    ids = basic_tokenizer.encode(text)
    decoded = basic_tokenizer.decode(ids)
    mapping = [(str(i), basic_tokenizer.decode([i])) for i in ids]

    return ids, decoded, mapping


with gr.Blocks() as demo:
    gr.HTML("<h1 align = 'center'> Token Generation for Hindi Dataset </h1>")

    with gr.Row():
        with gr.Column():
            inputs = [
                gr.TextArea(
                    label="Enter initial text to generate tokens in Hindi", lines=10
                )
            ]
            generate_btn = gr.Button(value="Generate Text")
        with gr.Column():
            enc = gr.Textbox(label="Encoded Tokens")
            txt = gr.Textbox(label="Decoded Text from tokens")
            map = gr.Textbox(label="Mapping of the tokens and respective texts")
            outputs = [enc, txt, map]

    generate_btn.click(fn=test_tokenizer, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(share=True)
