import keras
from keras import layers, ops, Sequential, Model, Layer

@keras.saving.register_keras_serializable()
class TransformerSelfAttention(Layer):
    '''
    Transformer layer with self-attention: input/output shape is (B, T, embed_dim)
    '''
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim        
        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        # Multi-head self-attention
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-forward network
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),    # First dense layer
            layers.Dense(embed_dim)                     # Output dense layer
        ])
        
    def call(self, inputs, mask=None):
        # Layer normalization before multi-head attention
        normed_inputs = self.layernorm1(inputs)        
        # Self-attention
        attn_output = self.mha(normed_inputs, normed_inputs, normed_inputs, attention_mask=mask)
        # Residual connection
        out1 = inputs + attn_output        
        # Layer normalization before the feed-forward network
        normed_out1 = self.layernorm2(out1)
        # Feed-forward network
        ffn_output = self.ffn(normed_out1)
        # Residual connection
        return self.layernorm3(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
            }
        )
        return config

@keras.saving.register_keras_serializable()
class TransformerCrossAttention(Layer):
    '''
    Transformer layer with cross-attention: input/output shape is (B, T, embed_dim)
    '''
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim        
        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)                
        # Multi-head self-attention
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-forward network
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),  # First dense layer
            layers.Dense(embed_dim)  # Output dense layer
        ])
        
    def call(self, inputs, context, mask=None):
        # Layer normalization before multi-head attention
        normed_inputs = self.layernorm1(inputs)        
        # Cross-attention: queries from normed inputs, keys and values from context 
        attn_output = self.mha(query=normed_inputs, value=context, key=context, attention_mask=mask)
        # Residual connection
        out1 = inputs + attn_output        
        # Layer normalization before the feed-forward network
        normed_out1 = self.layernorm2(out1)
        # Feed-forward network
        ffn_output = self.ffn(normed_out1)
        # Residual connection
        return self.layernorm3(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
            }
        )
        return config


def model1(input_shape, n_layers, embed_dim, n_heads, ff_dim):

    inputs = layers.Input(shape=input_shape) # -> (B, T, F)    
    _, T, F = inputs.shape
        
    x = layers.Dense(embed_dim)(inputs)  # -> (B, T, embed_dim)      
    
    embed_x = layers.Embedding(T, embed_dim)(ops.arange(T)) # (T, embed_dim) 
    x = x + embed_x
                
    for _ in range(n_layers): 
        x = TransformerSelfAttention(embed_dim, n_heads, ff_dim)(x)

    outputs = layers.Dense(F)(x)    # (B, T, embed_dim) -> (B, T, F)
                
    return Model(inputs, outputs)


def model2(input_shape, n_layers, embed_dim, n_heads, ff_dim):

    inputs = layers.Input(shape=input_shape)     # -> (B, T, F)
    transp = ops.transpose(inputs, axes=(0,2,1)) # -> (B, F, T)        

    _, T, F = inputs.shape

    # input projection
    x = layers.Dense(embed_dim)(inputs)  # -> (B, T, embed_dim)      
    y = layers.Dense(embed_dim)(transp)  # -> (B, F, embed_dim)

    # positional embeddings
    embed_x = layers.Embedding(T, embed_dim)(ops.arange(T)) # (T, embed_dim) 
    embed_y = layers.Embedding(F, embed_dim)(ops.arange(F)) # (F, embed_dim) 
    x = x + embed_x
    y = y + embed_y
                
    # decoder stack
    x2, y2 = x, y
    for _ in range(n_layers): 
        x1 = TransformerSelfAttention(embed_dim, n_heads, ff_dim)(x2)
        y1 = TransformerSelfAttention(embed_dim, n_heads, ff_dim)(y2)
                        
        x2 = TransformerCrossAttention(embed_dim, n_heads, ff_dim)(x1, y1)
        y2 = TransformerCrossAttention(embed_dim, n_heads, ff_dim)(y1, x1)

    # linear head
    x = layers.Dense(F)(x2) # (B, T, embed_dim) -> (B, T, F)
    y = layers.Dense(T)(y2) # (B, F, embed_dim) -> (B, F, T)

    y = ops.transpose(y, axes=(0,2,1)) # -> (B, T, F)

    # ouput projection
    outputs = layers.Dense(F)(x+y)  # (B, F)
        
    return Model(inputs, outputs)
