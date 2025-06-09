import torch
from pathlib import Path
from tokenizers import Tokenizer
from config import get_config, get_weights_file_path
from dataset import causal_mask
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Adds transformer/ to sys.path
from model import build_transformer

def load_tokenizers(config):
    tokenizer_src_path = config['tokenizer_path'].format(config['source_lang'])
    tokenizer_tgt_path = config['tokenizer_path'].format(config['target_lang'])

    print(f"🔍 Loading tokenizers:\n - Source: {tokenizer_src_path}\n - Target: {tokenizer_tgt_path}")
    tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
    tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

    return tokenizer_src, tokenizer_tgt


def test_tokenizer(tokenizer_src, tokenizer_tgt, test_sentence="Hello, how are you?"):
    print(f"\n📝 Testing Tokenization: '{test_sentence}'")

    src_encoded = tokenizer_src.encode(test_sentence)
    print(f"🔡 Source tokens: {src_encoded.tokens}")
    print(f"🔢 Source IDs: {src_encoded.ids}")

    # Try decoding back
    decoded = tokenizer_src.decode(src_encoded.ids)
    print(f"🔁 Decoded back: {decoded}")

    # Check special token IDs
    for token in ["[UNK]", "[PAD]", "[EOS]", "[SOS]"]:
        print(f"🧪 Token ID for '{token}': {tokenizer_src.token_to_id(token)}")


def test_encoder_output(config, tokenizer_src, device):
    sentence = "Hello, how are you?"
    token_ids = tokenizer_src.encode(sentence).ids
    print(f"📥 Token IDs: {token_ids}")

    encoder_input = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(2).to(device)

    model = build_transformer(
        d_model=config['d_model'],
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_src.get_vocab_size(),  # dummy
        n_heads=8,
        scr_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
    ).to(device)

    model.eval()
    with torch.no_grad():
        embeddings = model.src_embedding(encoder_input)  # ✅ convert Long -> Float
        encoder_output = model.encoder(embeddings, encoder_mask)

    print(f"🏗️ Encoder Input Shape: {encoder_input.shape}")
    print(f"📈 Encoder Output Shape: {encoder_output.shape}")
    print(f"📊 First vector:\n{encoder_output[0, 0, :10]}")

if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_src = Tokenizer.from_file(config['tokenizer_path'].format(config['source_lang']))
    test_encoder_output(config, tokenizer_src, device)
