# Intuition for a Diamond shaped embedding.

The KV cache acts as a sort of represenation for the sequence. The size of the representation is a function of the embedding size x the number of layers x the sequence length.

Our intuition is that the represenation in the layers near the input and the output, will be less complex, will change faster and are relevant for a shorter time. And that the representations in the middle layers will be more complex, will change more slowly and will be relevant over a longer period of time.

Current transformer architectures use a fixed embedding size for all layers and retain KV  cache at all levels for the entire sequence.  This results in a very large representation size, which will be very expressive, but we suspect it is not optimally efficient.

Our research is to find ways to vary the size of the embeddings and desample the sequence in time to improve computational efficiency.

One solution is to view the language model as a sliding, nested, encoder-decoder transformer.
The outer layers feed downsampled representations into the inner layers and then the output of the inner layers will be upsampled and tranformed into a token sequence.

An analogy is the way that a CNN auto encoder increases the feature count as it decreases the spatial resolution during encoding and then reduces the feature count as it increases the spatial resolution during the decoder phase.  We want to do something similar with the language model, increase the embedding size as we downsample the sequence in time during encoding and decrease the embedding size as we upsample during decoding.