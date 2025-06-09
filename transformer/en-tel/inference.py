import torch
from pathlib import Path
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
from dataset import causal_mask
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Adds transformer/ to sys.path
from model import build_transformer

def load_tokenizers(config):
    tokenizer_src_path = "/home/prashanth/Desktop/papers/transformer/en-tel/tokenizers/tokenizer_en.json"
    tokenizer_tgt_path = "/home/prashanth/Desktop/papers/transformer/en-tel/tokenizers/tokenizer_te.json"
    print(f"Loading tokenizers from {tokenizer_src_path} and {tokenizer_tgt_path}")
    
    tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
    tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
    return tokenizer_src, tokenizer_tgt

def translate_sentence(model, tokenizer_src, tokenizer_tgt, sentence, config, device):
    model.eval()
    tokens = tokenizer_src.encode(sentence).ids
    encoder_input = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(2).to(device)
    decoder_input = torch.tensor([[tokenizer_tgt.token_to_id("[SOS]")]], dtype=torch.long).to(device)

    eos_id = tokenizer_tgt.token_to_id("[EOS]")
    prev_token = None
    repeat_count = 0

    for _ in range(config['seq_len']):
        decoder_mask = causal_mask(decoder_input.size(1)).type(torch.bool).to(device)
        with torch.no_grad():
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
        next_token = output[0, -1].argmax(-1).item()

        if next_token == eos_id:
            break
        if next_token == prev_token:
            repeat_count += 1
        else:
            repeat_count = 0
        if repeat_count > 10:
            print("⚠️ Repeating token detected too often, breaking early.")
            break
        prev_token = next_token
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)

    return tokenizer_tgt.decode(decoder_input[0].tolist(), skip_special_tokens=True)

# ✅ ENTRY POINT
if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_src, tokenizer_tgt = load_tokenizers(config)

    model = build_transformer(
        d_model=config['d_model'],
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        n_heads=8,
        scr_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
    ).to(device)

    weights_path = get_weights_file_path(config, 5)
    print(f"Loading weights from {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])

    while True:
        sentence = input("Enter an English sentence (or 'exit' to quit):\n> ")
        if sentence.lower() == "exit":
            break
        translation = translate_sentence(model, tokenizer_src, tokenizer_tgt, sentence, config, device)
        print(f"🔁 Translated: {translation}")
