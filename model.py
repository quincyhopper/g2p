import torch
import torch.nn as nn
import math

class LexicalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()

        self.emb = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of tokenised input ids of shape (B, max_length_in_batch, V)

        Returns:
            Tensor of embedded input_ids of shape (B, L, D), where B is batch size, L is model max length, and D is d_model.
        """
        return self.emb(x.long())
    
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int, padding_idx: int = 0):
        """
        Args:
            embedding_dim: hidden size
            max_len: max sequence length supported (often 512 + 2 specials)
            padding_idx: index with which to pad sequences shorter than max_len
        """
        super().__init__()
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.position_embeddings = nn.Embedding(max_len, embedding_dim, padding_idx=0)

        # We'll map pad positions to index 0 in the position table.
        # (i.e., row 0 stays zeroed and is never updated)
        with torch.no_grad():
            self.position_embeddings.weight[0].zero_()

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embeddings, shape (B, L, D)
            input_ids: token ids, shape (B, L)

        Returns:
            x + learned position embeddings, same shape as x
        """
        bsz, seq_len, d_model = x.shape
        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_len={self.max_len}")

        device = x.device

        mask = (input_ids != self.padding_idx).to(torch.long)
        # position_ids = (all_position_ids * mask).to(device=device)

        # This only works with padding at the end. In order to take into
        # account sequence-initial padding, we can do
        position_ids = torch.cumsum(mask, dim=1) * mask

        pos_emb = self.position_embeddings(position_ids)  # (B, T, D)
        # Additive encoding
        return x + pos_emb
    
class MLP(nn.Module):
    def __init__(self, input_dim: int, mode: str):
        super().__init__()
        """
        Args:
            input_dim (int): equal to d_model.
            mode (str): relu or swiglu.
        """
        self.mode = mode

        if mode == 'relu':
            self.hidden_dim = 4 * input_dim
            self.fc1 = nn.Linear(input_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, input_dim)
            self.act = nn.ReLU()
        elif mode == 'swiglu':
            self.hidden_dim = int((2/3) * 4 * input_dim)
            self.fc_gate = nn.Linear(input_dim, self.hidden_dim)
            self.fc_up = nn.Linear(input_dim, self.hidden_dim)
            self.fc_down = nn.Linear(self.hidden_dim, input_dim)
            self.act = nn.SiLU()
        else:
            raise ValueError(f"MLP mode must be 'relu' or 'swiglu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == 'relu':
            return self.fc2(self.act(self.fc1(x)))
        elif self.mode == 'swiglu':
            gate = self.act(self.fc_gate(x))
            up = self.fc_up(x)
            return self.fc_down(gate * up)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_heads: int):
        """
        Args:
            input_dim (int): the input embedding dim. Same as d_model in this case.
            d_model (int): size of projected embeddings.
            num_heads (int): the number of heads for multi-head attention. Used to compute d_head.
        """
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.query_proj = nn.Linear(input_dim, d_model)
        self.key_proj = nn.Linear(input_dim, d_model)
        self.value_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal_masking: bool, context: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            x: tensor of shape (B, L, D) representing a batch of embedded inputs.
            causal_masking: if True, causal masking is used to compute attention. Otherwise, bidirectional attention is used.
            context: (Optional) encoder output (B, L, D). Used for cross-attention.

        Returns:
            Tensor of shape (B, L, D)
        """

        # If no context (encoder), just use x for K and V
        context = context if context is not None else x
        
        # Compute projections
        Q = self.query_proj(x)
        K = self.key_proj(context)
        V = self.value_proj(context)

        # Reshape to (batch, num_heads, seq_len, d_head)
        batch_size, seq_len_q, _ = x.shape
        seq_len_kv = context.shape[1]
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # Compute scores 
        # Q: (batch, num_heads, seq_len, d_head); K: (batch, num_heads, d_head, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # If causal masking, create triangle mask 
        if causal_masking:
            scores += torch.triu(
                torch.full_like(
                    scores,
                    float('-inf'),
                    device=x.device
                ),
                diagonal=1
            )

        # Compute attention
        probs = torch.softmax(scores, dim=-1)
        result = probs @ V # (batch, num_heads, seq_len, d_head)

        # Swap num_heads and seq_len to prepare to merge num_heads and d_head
        result = result.transpose(1, 2).contiguous() # Contiguous to prepare for .view()

        # Flatten last two dimensions (batch, seq_len, d_model)
        result = result.view(batch_size, seq_len_q, self.d_model)

        return self.output_proj(result)
    
class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_mode: str, dropout_p=0.1, is_decoder: bool=False):
        super().__init__()

        self.is_decoder = is_decoder
        self.dropout = nn.Dropout(dropout_p)

        # Self-attention
        self.ln1 = nn.LayerNorm(normalized_shape=d_model)
        self.attn = MultiHeadAttentionBlock(d_model, d_model, num_heads)
        
        # (Optional) Cross-attention
        if is_decoder:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttentionBlock(d_model, d_model, num_heads)

        # MLP
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_mode)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor=None):
        """
        Args:
            x: if encoder, this is the embeddings of the word tokens. If decoder, this is the embeddings of the IPA sequence.
            encoder_output: output of the encoder.

        Returns:
            Tensor of shape (B, L, D)
        """
        
        # Self attention
        residual = x
        x = self.ln1(x)
        x = self.attn(x, causal_masking=self.is_decoder)
        x = self.dropout(x)
        x = x + residual

        # (If decoder) Cross-attention
        if self.is_decoder and encoder_output is not None:
            residual = x
            x = self.ln_cross(x)
            x = self.cross_attn(x, causal_masking=False, context=encoder_output)
            x = self.dropout(x)
            x = x + residual

        # MLP
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual

        return x
    
class TransformerStack(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_heads, mlp_mode, dropout_p, num_layers, is_decoder=False):
        super().__init__()

        self.is_decoder = is_decoder

        # Input processing
        self.embedding = LexicalEmbedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = LearnedPositionalEmbedding(d_model, max_len, padding_idx=0)

        # Transformer
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, mlp_mode, dropout_p, is_decoder)
            for _ in range(num_layers)
        ])

        # TODO: do we need dropout in the embedding layer?

    def forward(self, input_ids: list[list[int]], encoder_output: torch.Tensor=None ):

        x = self.embedding(input_ids)
        x = self.pos_embedding(x, input_ids)

        if self.is_decoder and encoder_output is not None:
            for layer in self.layers:
                x = layer(x, encoder_output)
        else:
            for layer in self.layers:
                x = layer(x)

        return x
    
class LanguageModellingHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x: final decoder representation, tensor of shape (B, L, D)
        
        Returns:
            Tensor of shape (B, L, V).
        """
        return self.projection(x)
    
class Seq2Seq(nn.Module):
    def __init__(self, d_model, num_heads, mlp_mode, num_layers, dropout_p):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads. Got {d_model}/{num_heads}={d_model/num_heads:.2f}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_mode = mlp_mode
        self.dropout_p = dropout_p

        self.encoder = TransformerStack(
            vocab_size=258, 
            d_model=d_model, 
            max_len=64, 
            num_heads=num_heads,
            mlp_mode=mlp_mode,
            dropout_p=dropout_p, 
            num_layers=num_layers
            )

        self.decoder = TransformerStack(
            vocab_size=258, 
            d_model=d_model, 
            max_len=64, 
            num_heads=num_heads,
            mlp_mode=mlp_mode,
            dropout_p=dropout_p, 
            num_layers=num_layers,
            is_decoder=True
            )
        
        self.lm_head = LanguageModellingHead(
            d_model=d_model,
            vocab_size=258
        )

    def forward(self, encoder_inputs, decoder_inputs):
        """
        Args:
            encoder_inputs: batch of byte tokenised words.
            decoder_inputs: batch of byte toknised IPAs, shifted by 1.
        """

        enc_out = self.encoder(encoder_inputs)
        dec_out = self.decoder(decoder_inputs, enc_out)
        logits = self.lm_head(dec_out)

        return logits