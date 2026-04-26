import torch
import torch.nn as nn
import torch.nn.functional as F
from model import byte_tokenise

def decode_tokens(indices: torch.Tensor | list):
    if isinstance(indices, torch.Tensor):
        indices = indices.flatten().tolist()

    # Filter special tokens and padding
    filtered_bytes = bytes([i for i in indices if 0 < i < 256])
    return filtered_bytes.decode(encoding='utf-8', errors='ignore')

@torch.no_grad()
def greedy_generate(model: nn.Module, words: list, device, max_len: int=64):
    model.eval()

    input_ids = byte_tokenise(words).to(device) # (B, max_length_in_batch)
    enc_out = model.encoder(input_ids)         # (B, L, d_model)
    batch_size = enc_out.shape[0]

    # Create tensor of shape (B, 1) containing the BOS token (256)
    generated_seqs = torch.full((batch_size, 1), 256, dtype=torch.long, device=device)

    # Each element set to True if the sequence has finished (hit EOS)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        dec_out = model.decoder(generated_seqs, enc_out)
        logits = model.lm_head(dec_out) # (B, current_step, V)
        next_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True) # (B, 1)

        # Add the generated tokens to the sequences
        generated_seqs = torch.cat([generated_seqs, next_tokens], dim=1)

        # Check if the last token is EOS and change finished to True if it is 
        finished = torch.logical_or(finished, next_tokens.squeeze(-1) == 257)

        # If every word has finished, break
        if finished.all():
            break

    results = []
    for i in range(batch_size):
        results.append(decode_tokens(generated_seqs[i]))

    return results

@torch.no_grad()
def beam_search(model: nn.Module, word: str, device, beam_size: int=4, max_len: int=64):
    model.eval()
    input_ids = byte_tokenise(word).to(device)

    # Compute encoder output
    encoder_out = model.encoder(input_ids)

    # Initial state (log_probability, sequence_tensor)
    bos_idx = 256
    candidates = [(0.0, torch.tensor([[bos_idx]], device=device))]
    finished_candidates = []

    for _ in range(max_len):

        # Phase 1: Beam expansion
        expanded_candidates = []    
        for score, current_prefix in candidates:

            # Compute log probs
            decoder_out = model.decoder(current_prefix, encoder_out)
            logits = model.lm_head(decoder_out) # (B, L, V)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1) # (B, V)

            # Select top K candidates
            top_probs, top_indices = torch.topk(log_probs, k=beam_size, dim=-1) # Both of shape (B, beam_size)

            for i in range(beam_size):
                new_score = score + top_probs[0, i].item() # Take top score from i-th beam
                new_token = top_indices[0, i].view(1, 1) # (1, 1)
                new_prefix = torch.cat([current_prefix, new_token], dim=1) 

                if top_indices[0, i].item() == 257:
                    new_score = new_score / (len(new_prefix[0]) ** 0.7)
                    finished_candidates.append((new_score, new_prefix))
                else:
                    expanded_candidates.append((new_score, new_prefix))

        # Phase 2: Beam pruning
        expanded_candidates.sort(key=lambda x: x[0], reverse=True) # Sort in descending order
        candidates = expanded_candidates[:beam_size] # Cut off low scorers

        # If we have beam_size finished candidates, break early
        if len(finished_candidates) >= beam_size:
            break 

    # Normalise current candidates and compare to finished ones
    all_results = finished_candidates
    for score, prefix in candidates:
        norm_score = score / (len(prefix[0]) ** 0.7)
        all_results.append((norm_score, prefix))

    # From the best candidates, get the very best
    all_results.sort(key=lambda x: x[0], reverse=True)
    return decode_tokens(all_results[0][1])