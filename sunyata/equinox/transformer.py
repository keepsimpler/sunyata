from typing import Callable, List
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random
import optax

import equinox as eqx
from equinox import static_field

from sunyata.equinox.layer import LayerNorm, Linear, LinearWithMask, Embedding


@dataclass
class TransformerCfg:
    vocab_size: int  # vocabulary size
    seq_len: int  # token(word) sequence length
    embed_size: int  # size of embedded feature vector
    
    ff_size: int = None  # feedforward size
    heads: int = None  # heads number
    layers_num: int = None  # number of transformer layers
    pe_coef: float = None  # positional encoding coefficient
    
    # layer normalization coefficient
    elementwise_affine: bool = True
    use_bias: bool = False
    
    # initialization function of parameters
    embedding_init_func: Callable = jax.nn.initializers.xavier_normal()
    project_out_init_func: Callable = jax.nn.initializers.xavier_normal()
    
    seed: int = 1  # PRNG seed    

    def __post_init__(self):
        self.pe_coef = 1 / self.embed_size ** 0.5  # Positional Encodings coefficient 1 / sqrt(embed_size)

        
class MultiHeadAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    
    scale: float
    mask: bool
    heads: int
    head_size: int
    
    def __init__(self, heads: int, embed_size: int, key: random.PRNGKey, mask: bool=True):
        super().__init__()
        query_key, key_key, value_key, output_key = random.split(key, 4)
        
        self.query = Linear(query_key, embed_size, embed_size)
        self.key = Linear(key_key, embed_size, embed_size)
        self.value = Linear(value_key, embed_size, embed_size)
        
        self.output = Linear(output_key, embed_size, embed_size)
        
        head_size = embed_size // heads
        self.scale = 1 / head_size ** 0.5
        self.mask = mask
        self.heads = heads
        self.head_size = head_size
        
    def __call__(self, x):
        # x shape (batch_size, seq_len, embed_size)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        query = self.query(x)
        query = query.reshape(*query.shape[:-1], self.heads, self.head_size)
        key = self.key(x)
        key = key.reshape(*key.shape[:-1], self.heads, self.head_size)
        value = self.value(x)
        value = value.reshape(*value.shape[:-1], self.heads, self.head_size)

        scores = jnp.einsum('b i h d, b j h d -> b i j h', query, key)
        scores = scores * self.scale
        if self.mask:
            mask_matrix = (jnp.tril(jnp.ones((seq_len, seq_len), bool)) == 0) * float('-inf')
            mask_matrix = jnp.expand_dims(mask_matrix, axis=-1)
            scores = scores + mask_matrix
            
        attn = jax.nn.softmax(scores, axis=1)
        x = jnp.einsum('b i j h, b j h d -> b i h d', attn, value)
        
        x = x.reshape(batch_size, seq_len, -1)
        x = self.output(x)
        return x
        

class FeedForward(eqx.Module):
    linear1: Linear
    linear2: Linear
    activation: Callable
        
    def __init__(self, embed_size: int, ff_size: int, key: random.PRNGKey, activation=jax.nn.relu):
        super().__init__()
        linear1_key, linear2_key = random.split(key, 2)
        self.linear1 = Linear(linear1_key, embed_size, ff_size)
        self.linear2 = Linear(linear2_key, ff_size, embed_size)
        self.activation = activation
        
    def __call__(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TransformerLayer(eqx.Module):
    self_attn: MultiHeadAttention
    feed_forward: FeedForward
    norm_self_attn: LayerNorm
    norm_ff: LayerNorm
    
    def __init__(self, embed_size: int, self_attn: MultiHeadAttention, feed_forward: FeedForward,
                 elementwise_affine=True, use_bias=True):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        
        self.norm_self_attn = LayerNorm((embed_size,), elementwise_affine=elementwise_affine, use_bias=use_bias)
        self.norm_ff = LayerNorm((embed_size,), elementwise_affine=elementwise_affine, use_bias=use_bias)
        
    def __call__(self, x):
        z = self.norm_self_attn(x)
        self_attn = self.self_attn(z)
        x = x + self_attn
        
        z = self.norm_ff(x)
        ff = self.feed_forward(z)
        x = x + ff
        
        return x


class EmbeddingWithLearnedPositionalEncoding(eqx.Module):
    embedding: Embedding
    positional_encoding: jnp.ndarray
    
    pe_coef: float = static_field()
    
    def __init__(self, embedding:Embedding, seq_len: int, embed_size: int):
        super().__init__()
        self.embedding = embedding
        self.positional_encoding = jnp.zeros((seq_len, embed_size))
        self.pe_coef = 1 / embed_size ** 0.5
        
    def __call__(self, x):
        x = self.embedding(x)
        x = x * self.pe_coef + self.positional_encoding
        return x


class Transformer(eqx.Module):
    embedding_with_positional_encoding: EmbeddingWithLearnedPositionalEncoding
    layers: List[TransformerLayer]
    out_norm: LayerNorm
    project_out: Linear
    
    def __init__(self, cfg:TransformerCfg):
        super().__init__()
        key = random.PRNGKey(cfg.seed)
        embedding_key, project_out_key, key = random.split(key, 3)
        
        embedding = Embedding(embedding_key, cfg.vocab_size, cfg.embed_size, cfg.embedding_init_func)
        self.embedding_with_positional_encoding = EmbeddingWithLearnedPositionalEncoding(embedding, cfg.seq_len, cfg.embed_size)
        self.out_norm = LayerNorm((cfg.embed_size,), elementwise_affine=cfg.elementwise_affine, use_bias=cfg.use_bias)
        self.project_out = Linear(project_out_key, cfg.embed_size, cfg.vocab_size, weight_init_func=cfg.project_out_init_func)
        
        layers = []
        for i in range(cfg.layers_num):
            attn_key, ffn_key, key = random.split(key, 3)
            attn = MultiHeadAttention(cfg.heads, cfg.embed_size, attn_key)
            ffn = FeedForward(cfg.embed_size, cfg.ff_size, ffn_key)
            transformer_layer = TransformerLayer(cfg.embed_size, attn, ffn, 
                                                 elementwise_affine=cfg.elementwise_affine, use_bias=cfg.use_bias)
            layers.append(transformer_layer)
        self.layers = layers
        
    def __call__(self, x):
        x = self.embedding_with_positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        x = self.project_out(x)
        return x


if __name__ == '__main__':
    batch_size, vocab_size, seq_len, embed_size = 3, 256, 16, 16
    inputs = jax.random.randint(jax.random.PRNGKey(1), (batch_size, seq_len), minval=0, maxval=vocab_size-1)

    cfg = TransformerCfg(vocab_size=vocab_size, seq_len=seq_len, embed_size=seq_len, ff_size=seq_len*4, heads=8, layers_num=4)
    
    attn = MultiHeadAttention(cfg.heads, cfg.embed_size, random.PRNGKey(cfg.seed))
    feedforward = FeedForward(cfg.embed_size, cfg.ff_size, random.PRNGKey(cfg.seed))
    transformer_layer = TransformerLayer(cfg.embed_size, attn, feedforward, use_bias=False)

    transformer = Transformer(cfg)
    output = transformer(inputs)
    print(output.shape)