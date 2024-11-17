# Vision-GPT 👁️ Coding a Multi-Modal vision model from scratch 🚀


Vision-GPT is an attempt to create a model that's not just a chatterbox, but also a keen observer. It's like giving your AI both a silver tongue and eagle eyes, reproduced from scratch from the PaliGemma paper by Google Deepmind!

![Vision-GPT Logo](static\image.png)

1. **Clone this repo** (because, duh!)
   ```bash
   git clone https://github.com/PrateekJannu/Vision-GPT.git
   cd Vision-GPT
   ```

2. **Summon the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the mighty paligemma-3b-pt-224 model**
   ```bash
    git lfs install
    git clone https://huggingface.co/google/paligemma-3b-pt-224
   ```

## 🎩 Let the Show Begin!

Now that you've gathered all the magical components, it's time to bring Vision-GPT to life!

```bash
python inference.py --model_path "paligemma-3b-pt-224" --prompt "what do you see in this image?" --image_file_path "image.png" --max_tokens_to_generate 100 --temperature 0.8 --top_p 0.9 --do_sample False --only_cpu True
```

Watch in awe as Vision-GPT describes your image with the eloquence of Shakespeare and the observational skills of Sherlock Holmes!


# Things you can expect to learn


## 1. What is Contrastive Learning in Multi-Modal and why is it necessary?

Contrastive learning in multi-modal settings, particularly involving text and images, is a technique used to learn aligned representations across different modalities. The goal is to maximize the similarity between corresponding text-image pairs while minimizing similarity between unrelated pairs[1][2].

This approach is necessary for several key reasons:

1. **Joint Representation Learning**: It enables learning a shared semantic space where both text and image embeddings can be directly compared, facilitating cross-modal retrieval and understanding[1].

2. **Alignment of Modalities**: By pulling together representations of related text-image pairs and pushing apart unrelated ones, it creates alignments between the visual and textual domains[2].

3. **Transfer Learning**: The learned representations can be used for downstream tasks like zero-shot classification, where a model can classify images into categories it hasn't explicitly seen during training[3].

4. **Data Efficiency**: Contrastive learning can leverage large amounts of unlabeled multi-modal data, reducing the need for expensive labeled datasets[2].

5. **Robustness**: By learning to distinguish between related and unrelated pairs across modalities, models become more robust to noise and variations in input[3].

The necessity stems from the challenge of bridging the semantic gap between different modalities. Text and images, for instance, have very different raw representations. Contrastive learning provides a framework to learn meaningful, comparable representations across these disparate domains[1][2][3].

## 2. Why we use cross entropy in Contrastive Learning?

Cross entropy is commonly used in contrastive learning for several important reasons:

1. **Probabilistic Interpretation**: Cross entropy provides a natural way to interpret the contrastive task as a classification problem. It measures the difference between the predicted probability distribution and the true distribution[4].

2. **Gradient Properties**: The gradients of cross entropy loss are well-behaved, especially when combined with softmax normalization. This leads to stable and efficient training[4].

3. **Flexibility**: Cross entropy can be easily adapted to handle multiple positive pairs or weighted samples, which is often necessary in contrastive learning setups[5].

4. **Theoretical Grounding**: Cross entropy has strong connections to information theory and maximum likelihood estimation, providing a solid theoretical foundation for its use[4].

5. **Effectiveness in Practice**: Empirically, cross entropy has been shown to work well in various contrastive learning frameworks, consistently producing state-of-the-art results[5].

In the context of multi-modal contrastive learning, cross entropy is typically applied to the softmax-normalized similarity scores between text and image embeddings. This formulation encourages the model to assign high probability to matching text-image pairs while assigning low probabilities to non-matching pairs[5].

## 3. What is the problem with CLIP and what is SigLIP and how does it deal with the numerical stability issue of CLIP?

CLIP (Contrastive Language-Image Pre-training) is a powerful multi-modal model, but it faces a significant numerical stability issue:

**Problem with CLIP**: 
CLIP uses a softmax-based contrastive loss which can become numerically unstable with large batch sizes or high-dimensional embeddings. This is because the exponential operations in softmax can lead to overflow or underflow, especially when dealing with large logits[6].

**SigLIP (Sigmoid-based LIP) Solution**:
SigLIP addresses this stability issue by replacing the softmax operation with a sigmoid function:

1. **Sigmoid Function**: Instead of using softmax to normalize the entire similarity matrix, SigLIP applies a sigmoid function to each individual similarity score[6].

2. **Binary Classification**: This transforms the problem into a series of binary classifications, where each text-image pair is classified as either matching (1) or non-matching (0)[6].

3. **Numerical Stability**: The sigmoid function is more numerically stable than softmax, as it operates on individual values rather than normalizing across a potentially large set of values[6].

4. **Scalability**: This approach scales better to larger batch sizes and higher-dimensional embeddings, as each similarity score is treated independently[6].

5. **Performance**: Despite this simplification, SigLIP has been shown to achieve comparable or better performance than CLIP on various benchmarks[6].

By addressing the numerical stability issue, SigLIP allows for more reliable training and potentially better scalability to larger models and datasets[6].

## 4. How to make softmax numerically stable and how to choose constant in the exp(a) so that it does not lead to infinity?

Making softmax numerically stable is crucial for reliable model training. Here are some techniques to achieve this:

1. **Subtract Maximum Value**: 
   The key trick is to subtract the maximum value from all elements before applying the exponential function:

   $$ \text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}} $$

   This ensures that the largest exponent is always 0, preventing overflow[7].

2. **Log-Sum-Exp Trick**: 
   For very large values, compute the logarithm of the softmax:

   $$ \log(\text{softmax}(x_i)) = x_i - \max(x) - \log(\sum_j e^{x_j - \max(x)}) $$

   This further improves numerical stability[7].

3. **Choosing the Constant**:
   When adding a constant $$a$$ as in $$exp(x + a)$$, choose $$a$$ such that:
   - It's large enough to prevent underflow for small $$x$$ values.
   - It's small enough to prevent overflow for large $$x$$ values.
   - Typically, $$a = -\max(x)$$ is a good choice, as it ensures the largest exponent is 1[7].

4. **Use Double Precision**: 
   If possible, use double precision floating-point arithmetic for intermediate calculations[8].

5. **Clipping**: 
   Implement a maximum absolute value for the inputs to exp() to prevent extreme values[8].

By applying these techniques, particularly the "subtract maximum" approach, you can significantly improve the numerical stability of softmax computations without changing the mathematical properties of the function[7][8].

## 5. How SigLIP which is a sigmoid based LIP makes it a binary classification task of having the diagonals which are the text embeddings corresponding to the image embeddings 1 and the rest 0

SigLIP (Sigmoid-based Language-Image Pre-training) reformulates the contrastive learning problem as a binary classification task, which offers several advantages:

1. **Sigmoid Activation**: 
   SigLIP applies a sigmoid function to each similarity score between text and image embeddings:
   
   $$ p(i,j) = \sigma(s(i,j) / \tau) = \frac{1}{1 + e^{-s(i,j) / \tau}} $$

   where $$s(i,j)$$ is the similarity score and $$\tau$$ is a temperature parameter[6].

2. **Binary Labels**: 
   - Diagonal elements (matching text-image pairs) are labeled as 1.
   - Off-diagonal elements (non-matching pairs) are labeled as 0[6].

3. **Loss Function**: 
   The binary cross-entropy loss is used for each pair:
   
   $$ L = -\sum_{i,j} [y_{i,j} \log(p(i,j)) + (1-y_{i,j}) \log(1-p(i,j))] $$

   where $$y_{i,j}$$ is the binary label (1 for matching pairs, 0 otherwise)[6].

4. **Independent Pair Processing**: 
   Each text-image pair is treated independently, allowing for better scalability and stability compared to softmax-based approaches[6].

5. **Balanced Learning**: 
   This formulation naturally balances the learning between positive and negative pairs, as each pair contributes equally to the loss[6].

6. **Flexibility**: 
   The sigmoid-based approach allows for easy incorporation of hard negative mining or weighted sampling strategies[6].

By framing the problem as binary classification, SigLIP simplifies the learning objective while maintaining the core principle of contrastive learning. This approach not only addresses the numerical stability issues of softmax-based methods but also provides a clear, intuitive interpretation of the learning process[6].

Citations:
[1] https://ojs.aaai.org/index.php/AAAI/article/download/25819/25591
[2] https://arxiv.org/abs/2211.03646
[3] https://arxiv.org/abs/2402.14551
[4] https://www.reddit.com/r/MachineLearning/comments/q1y9pm/d_why_would_this_loss_function_for_contrastive/
[5] https://paperswithcode.com/method/supervised-contrastive-loss
[6] https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec?gi=08bde7ecda86
[7] https://www.v7labs.com/blog/contrastive-learning-guide
[8] https://lilianweng.github.io/posts/2021-05-31-contrastive/


## 6. Why use contrastive learning rather than only image embeddings?

Contrastive learning is used in addition to image embeddings because it helps create representations that are good for both images and text. The key benefits are:

1. **Alignment of modalities**: Contrastive learning encourages the model to create embeddings where similar images and text descriptions are close together in the embedding space[1]. This alignment between modalities is crucial for tasks like image-text matching or cross-modal retrieval.

2. **Improved generalization**: By learning to distinguish between similar and dissimilar pairs, contrastive learning helps the model capture more nuanced features that generalize better to unseen data[2].

3. **Robust representations**: Contrastive learning often leads to more robust and transferable representations compared to supervised learning on image embeddings alone[3].

4. **Self-supervised learning**: Contrastive approaches allow for self-supervised learning on large amounts of unlabeled image-text pairs, which can be particularly valuable when labeled data is scarce[4].

## 7. Vision Transformers: Convolution, flattening, and positional encoding

Vision Transformers (ViT) process images differently from textual transformers:

1. **Patch embedding**: Images are divided into fixed-size patches, which are then linearly projected to create initial embeddings[5].

2. **Flattening**: These patch embeddings are flattened into a sequence of vectors[5].

3. **Learnable positional encoding**: Unlike textual transformers that often use fixed sinusoidal encodings, ViTs typically use learnable positional encodings[5][6].

4. **Dimensional knowledge**: The learnable positional encodings allow the model to capture spatial relationships between patches, providing access to dimensional knowledge of the image[6].

## 8. Why Vision Transformers use learnable positional encodings instead of sinusoidal functions

Vision Transformers use learnable positional encodings rather than fixed sinusoidal functions for several reasons:

1. **Flexibility**: Learnable encodings can adapt to the specific patterns and spatial relationships present in image data, which may differ from those in text[7].

2. **Performance**: Empirical studies have shown that learnable positional encodings often perform better than fixed encodings for vision tasks[8].

3. **Input size adaptation**: Learnable encodings can more easily adapt to different input sizes or resolutions during fine-tuning.

4. **Task-specific optimization**: The encodings can be optimized for specific vision tasks, potentially capturing more relevant spatial information[7].

## 9. Why LayerNorm is used in linear layers due to covariate shift

LayerNorm is used in linear layers to address covariate shift for several reasons:

1. **Stability**: LayerNorm helps stabilize the distribution of activations across different layers, reducing the internal covariate shift.

2. **Faster convergence**: By normalizing inputs to each layer, LayerNorm can lead to faster training convergence.

3. **Invariance to scaling**: LayerNorm makes the model more robust to changes in the scale of layer inputs.

4. **Improved gradient flow**: Normalization can help improve gradient flow through deep networks, addressing vanishing/exploding gradient problems.

## 10. Why BatchNorm increases dependency on batch size and forces larger batch sizes

BatchNorm increases dependency on batch size for several reasons:

1. **Statistics calculation**: BatchNorm computes mean and variance statistics over the current batch, making these statistics less reliable for small batch sizes.

2. **Inconsistent behavior**: With small batches, the statistics can vary significantly between batches, leading to inconsistent behavior during training and inference.

3. **Regularization effect**: BatchNorm has an implicit regularization effect that diminishes with smaller batch sizes, potentially leading to overfitting.

4. **Gradient noise**: Smaller batch sizes can introduce more noise into the gradient updates when using BatchNorm, potentially destabilizing training.

To mitigate these issues, larger batch sizes are often used with BatchNorm to ensure more stable and representative statistics.

Citations:
[1] https://arxiv.org/abs/2102.10882
[2] https://www.reddit.com/r/MachineLearning/comments/lrkok7/d_what_is_the_positional_encoding_used_in_vision/
[3] https://www.pinecone.io/learn/series/image-search/vision-transformers/
[4] https://towardsdatascience.com/position-embeddings-for-vision-transformers-explained-a6f9add341d5?gi=4154e0a620bb
[5] https://nn.labml.ai/transformers/vit/index.html
[6] https://stackoverflow.com/questions/73113261/the-essence-of-learnable-positional-embedding-does-embedding-improve-outcomes-b
[7] https://www.sciencedirect.com/science/article/abs/pii/S1047320322001845
[8] https://arxiv.org/pdf/2102.10882.pdf


## 11. Why MLP in Vision Transformer?

Multilayer Perceptrons (MLPs) play a crucial role in Vision Transformers for several reasons:

1. **Independent transformation**: MLPs allow for independent transformation of each embedding, providing the model with more flexibility to learn complex representations[1].

2. **Increased degrees of freedom**: By applying MLPs to each token independently, the model gains additional parameters and degrees of freedom, enabling it to capture more nuanced features[1].

3. **Non-linearity**: MLPs introduce non-linearity into the model, which is essential for learning complex, non-linear relationships in the data[1][2].

4. **Feature interaction**: MLPs help in mixing information across different spatial locations and channels, facilitating better feature interaction[1].

5. **Computational efficiency**: MLPs can be implemented efficiently on modern hardware, making them a practical choice for large-scale vision models[1].

## 12. Non-Linearity and Gradient Flow

Non-linear activation functions play a crucial role in defining the flow of gradients through neural networks:

1. **ReLU (Rectified Linear Unit)**:
   - Allows positive gradients to flow unchanged
   - Blocks negative gradients (sets them to zero)
   - Helps mitigate the vanishing gradient problem for positive inputs
   - Can suffer from "dying ReLU" problem for consistently negative inputs[3]

2. **GELU (Gaussian Error Linear Unit)**:
   - Provides a smooth, non-monotonic activation function
   - Allows for more nuanced gradient flow, especially for inputs close to zero
   - Approximates a smooth version of ReLU, potentially capturing more complex patterns
   - Has shown improved performance in transformer architectures[1][4]

Both functions influence how gradients propagate through the network, affecting training dynamics and the model's ability to learn complex patterns.

## 13. Heuristic Nature of Non-Linear Functions

The development and selection of non-linear activation functions often rely on heuristics and empirical evidence:

1. **Task-specific performance**: Different activation functions may perform better on specific tasks or datasets[1][4].

2. **Model architecture considerations**: The choice of activation function can depend on the overall network architecture and depth[1].

3. **Gradient flow properties**: Functions are often designed to address specific issues like vanishing or exploding gradients[3].

4. **Computational efficiency**: Some functions may be preferred due to their ease of implementation or computational efficiency[1].

5. **Empirical testing**: Many activation functions are discovered through experimentation and refined based on observed performance[4].

No single activation function works optimally for all models or tasks, highlighting the importance of experimentation and task-specific optimization.

## 14. Linear Transformations in Attention Mechanisms

Linear transformations are applied to queries, keys, and values in attention mechanisms for several reasons:

1. **Dimensionality adjustment**: Linear layers allow for adjusting the dimensionality of the input to match the desired attention head size[1].

2. **Learned projections**: These transformations learn to project the input into spaces that are more suitable for computing attention[1].

3. **Parameter efficiency**: Linear layers provide a computationally efficient way to transform the inputs while maintaining the same dimension[1].

4. **Feature extraction**: The linear transformations can be seen as feature extractors, learning to emphasize relevant aspects of the input for attention computation[1].

5. **Consistency**: Maintaining the same dimension after transformation allows for consistent processing across different layers and components of the model[1].

## 15. Multi-Head Attention vs. Single-Head Attention

Multi-head attention offers several advantages over single-head attention:

1. **Diverse representations**: Each head can focus on different aspects of the input, capturing various types of relationships[1].

2. **Parallel processing**: Multiple heads can be computed in parallel, potentially improving computational efficiency[1].

3. **Increased model capacity**: Multi-head attention increases the model's capacity to learn complex patterns without significantly increasing the number of parameters[1].

4. **Context-dependent interpretation**: Different heads can capture different contextual meanings of words, which is particularly useful for words with multiple meanings or usages across languages[1].

5. **Improved feature interaction**: By allowing attention to be computed in multiple representation subspaces, multi-head attention facilitates more complex feature interactions[1].

While a single head could theoretically learn to perform these functions, the use of multiple heads provides a more structured and efficient way to capture diverse relationships in the data.

Citations:
[1] https://www.baeldung.com/cs/gelu-activation-function
[2] https://data-intelligence.hashnode.dev/comprehensive-guide-to-the-relu-activation-function-in-neural-networks-definition-role-and-type-explained
[3] https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[4] https://www.reddit.com/r/MachineLearning/comments/eh80jp/d_gelu_better_than_relu/
[5] https://www.v7labs.com/blog/neural-networks-activation-functions
[6] https://arxiv.org/pdf/2402.02593.pdf
[7] https://github.com/Shawon5030/Deep-Leaning
[8] https://arxiv.org/pdf/2405.20768.pdf


## 16. What is the softmax function formula and why is it used specifically and how does it help the model and are there any other functions that have been used similarly?

The softmax function formula is:

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

Where z is the input vector and K is the number of classes[2].

The softmax function is used specifically to convert a vector of real numbers into a probability distribution. It's commonly used as the final activation function in neural networks for multi-class classification problems[1][2].

The softmax function helps the model in several ways:

1. It normalizes the output to a probability distribution, making it easier to interpret the results[1].
2. It enhances model generalization by encouraging the model to be confident about its most probable prediction while reducing confidence in incorrect predictions[1].
3. It works well with gradient descent optimization methods, as it is differentiable[1].
4. It provides numerical stability in computations, especially when combined with techniques like log-softmax[1].

While softmax is the most commonly used function for this purpose, there are other similar functions that have been used:

1. Sigmoid function: Used for binary classification problems.
2. Hardmax function: A non-differentiable version of softmax that assigns 1 to the maximum value and 0 to others.
3. Sparsemax: A sparse alternative to softmax that can output exact zero probabilities[3].

## 17. Why do we need to mix the attention scores after computing the attention score?

Mixing attention scores after computation is typically done to combine information from different attention heads or layers. This process, often called multi-head attention, allows the model to:

1. Capture different aspects of the input: Each attention head can focus on different parts of the input, allowing the model to capture various relationships and patterns.

2. Increase model capacity: By using multiple attention heads, the model can learn more complex representations.

3. Improve performance: Combining information from multiple attention heads often leads to better overall performance.

4. Enhance robustness: Multiple attention heads can provide redundancy, making the model more robust to noise or errors in individual heads.

The mixing is usually done through concatenation followed by a linear transformation, allowing the model to learn how to best combine the information from different heads.

## 18. What are the use of adding the image embeddings to the text embeddings to the instructions before feeding it into the decoder?

Adding image embeddings to text embeddings before feeding them into the decoder serves several purposes:

1. Multimodal fusion: It allows the model to combine information from both visual and textual modalities, enabling it to understand and generate content based on both image and text inputs.

2. Context enrichment: The image embeddings provide additional context to the text, potentially improving the relevance and accuracy of the generated output.

3. Cross-modal understanding: By combining the embeddings, the model can learn to associate textual descriptions with visual features, enhancing its ability to understand and describe images.

4. Conditional generation: This technique enables the model to generate text that is conditioned on both the input text and the image, allowing for more specific and relevant outputs.

5. Feature transfer: It allows visual features to influence the text generation process, potentially leading to more descriptive and visually-aware outputs.

This approach is commonly used in multimodal models like CLIP (Contrastive Language-Image Pre-training) and various image captioning and visual question answering systems.

## 19. Why is it called conditional Generation? Because our generated tokens are on the condition of the image input

Conditional generation is indeed called so because the output is generated based on, or conditioned by, some input. In the case of image-to-text models, the generated text is conditioned on the input image. This term accurately describes the process for several reasons:

1. Input dependency: The generated output is directly influenced by and dependent on the input condition (in this case, the image).

2. Contextual relevance: The model produces output that is contextually relevant to the input condition, rather than generating text independently.

3. Controlled output: The input condition allows for some control over the generated output, guiding it towards relevance to the input.

4. Multimodal integration: It describes the process of integrating information from one modality (images) to influence the generation in another modality (text).

5. Task specificity: The term "conditional generation" distinguishes this task from unconditional text generation, where text is produced without a specific input condition.

This approach is used in various applications, including image captioning, visual question answering, and text-to-image generation models like DALL-E, where the generated image is conditioned on the input text.

## 20. What is weight tie(ing) technique and how is it useful and what are the advantages and disadvantages?

Weight tying is a technique used in neural networks, particularly in language models, where certain layers share the same weights. Most commonly, it refers to sharing weights between the input embedding layer and the output softmax layer in a language model.

Advantages of weight tying:

1. Reduced model size: By sharing weights, the number of parameters in the model is significantly reduced, leading to smaller model sizes.

2. Improved generalization: Weight tying can help prevent overfitting by reducing the model's capacity to memorize the training data.

3. Faster training: With fewer parameters to update, training can be faster.

4. Improved performance: In many cases, weight tying has been shown to improve model performance, particularly in language modeling tasks.

5. Semantic consistency: It enforces a consistency between how words are represented at the input and output of the model.

Disadvantages of weight tying:

1. Reduced model capacity: The reduction in parameters can potentially limit the model's ability to learn complex patterns.

2. Not always applicable: Weight tying is not suitable for all types of models or tasks, particularly where input and output representations need to be different.

3. Potential performance ceiling: In some cases, weight tying might limit the maximum performance achievable by the model.

4. Increased complexity in implementation: Implementing weight tying can add complexity to the model architecture and training process.

Weight tying has been particularly successful in natural language processing tasks, showing improvements in perplexity and overall performance in language models while significantly reducing the number of parameters.

Citations:
[1] https://botpenguin.com/glossary/softmax-function
[2] https://en.wikipedia.org/wiki/Softmax_function
[3] https://www.engati.com/glossary/softmax-function
[4] https://www.pinecone.io/learn/softmax-activation/
[5] https://community.deeplearning.ai/t/why-use-softmax-instead-of-a-linear-transform-that-sums-to-1/9107
[6] https://stackoverflow.com/questions/17187507/why-use-softmax-as-opposed-to-standard-normalization


## 21. Why do you have different num of heads for query and different for key and values?

Having a different number of heads for queries versus keys and values is a technique called multi-query attention. This approach uses multiple query heads but only one key and value head, which can improve efficiency while maintaining model quality.

The main benefits of multi-query attention are:

1. Reduced memory usage: Fewer key and value heads means less memory is required to store them.
2. Faster inference: With fewer key and value computations, the model can generate outputs more quickly.
3. Comparable performance: Despite the reduction in heads, multi-query attention often achieves similar results to full multi-head attention.

This technique is particularly useful for large language models, where efficiency gains can be significant[1][2].

## 22. What exactly is RMSNorm?

RMSNorm, or Root Mean Square Layer Normalization, is a simplified version of Layer Normalization that focuses solely on re-scaling invariance. The key features of RMSNorm are:

1. It normalizes the summed inputs to a neuron using only the root mean square (RMS) statistic.
2. Unlike LayerNorm, it does not perform mean centering.
3. It provides re-scaling invariance but not re-centering invariance.

The RMSNorm formula is:

$$ \bar{a}_i = \frac{a_i}{\text{RMS}(a)} \cdot g_i $$

where $$\text{RMS}(a) = \sqrt{\frac{1}{n}\sum_{i=1}^n a_i^2}$$

RMSNorm is computationally simpler than LayerNorm, making it more efficient while often achieving comparable performance[1][3].

## 23. What is Rope Theta and how is it useful?

RoPE (Rotary Position Embedding) Theta is a hyperparameter used in the Rotary Position Embedding technique. RoPE is a method for encoding positional information in transformer models. The key aspects of RoPE Theta are:

1. It determines the base wavelength for the rotary embeddings.
2. It affects how quickly the positional information changes across different positions in the sequence.
3. A smaller Theta value results in slower changes, potentially allowing the model to handle longer sequences.

RoPE Theta is useful because:

1. It allows for flexible adjustment of the model's ability to handle different sequence lengths.
2. It can be tuned to optimize the trade-off between short-range and long-range dependencies in the input.
3. Proper selection of Theta can improve the model's performance on tasks requiring understanding of positional relationships[5].

## 24. What is attention bias?

Attention bias in transformer models refers to additional terms added to the attention scores before the softmax operation. Key points about attention bias include:

1. Purpose: It modifies the attention distribution, influencing which parts of the input sequence the model focuses on.
2. Implementation: It's typically added as a matrix or tensor to the raw attention scores.
3. Types: Can include positional bias, causal bias (for autoregressive models), or task-specific biases.

Attention bias is useful for:

1. Incorporating positional information without separate positional embeddings.
2. Enforcing causal attention in language models (preventing attention to future tokens).
3. Guiding the model's attention based on prior knowledge or task requirements.

By carefully designing attention biases, model performance and behavior can be significantly influenced[2][4].

## 25. What is the problem that the KV cache is solving in transformers?

The KV (Key-Value) cache in transformers addresses the computational inefficiency in autoregressive generation. Its main features and benefits are:

1. Problem solved: Repeated computation of keys and values for previously processed tokens.
2. Mechanism: Stores computed key and value tensors from previous time steps.
3. Benefits:
   - Reduces redundant computations during inference.
   - Significantly speeds up autoregressive generation.
   - Lowers memory bandwidth requirements.

The KV cache is particularly important because:

1. It makes real-time text generation more feasible, especially for longer sequences.
2. It allows for more efficient use of computational resources during inference.
3. It enables faster response times in interactive applications of language models.

By caching and reusing previously computed keys and values, the KV cache dramatically improves the efficiency of transformer models during sequential generation tasks[1][4].

Citations:
[1] https://paperswithcode.com/method/rmsnorm
[2] https://dl.acm.org/doi/pdf/10.5555/3454287.3455397
[3] https://openreview.net/references/pdf?id=S1qBAf6rr
[4] https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/c/rmsnorm.html
[5] https://proceedings.neurips.cc/paper/2019/file/1e8a19426224ca89e83cef47f1e7f53b-Reviews.html
[6] https://www.reddit.com/r/MachineLearning/comments/1apb3th/d_why_does_it_matter_that_rmsnorm_is_faster_than/
[7] https://www.youtube.com/watch?v=r3O76I79YkE


## 26. How KV cache buffer works and how it helps with computation and memory?

The KV (Key-Value) cache buffer is a critical component in large language models that helps optimize inference speed and memory usage. Here's how it works:

The KV cache stores the key and value vectors generated for each token during processing. These vectors are based on the token's embedding and the model's weights[1]. As the model processes a sequence of tokens, it accumulates these key-value pairs in the cache.

The cache serves two main purposes:

1. **Computation optimization**: By storing previously computed key-value pairs, the model avoids recalculating them for each new token generation. This significantly reduces the computational load, especially for longer sequences[2].

2. **Memory efficiency**: Although the KV cache itself requires memory, it ultimately leads to more efficient memory usage. Without it, the model would need to recompute and store much larger intermediate results for each token[1].

The size of the KV cache grows linearly with the number of tokens processed. For example, in a 13 billion parameter model, the cache for a single token may require around 800 KB of space. For a sequence of 2048 tokens, this can add up to approximately 1.6 GB[1].

While the KV cache does consume memory, it ultimately enables faster inference and more efficient processing of long sequences, making it a crucial optimization technique for large language models.

## 27. What is prefilling in KV cache in the same forward pass?

Prefilling in KV cache refers to the initial phase of processing where the model computes and stores key-value pairs for the input prompt or context in a single forward pass. This process involves:

1. **Batch processing**: The model processes the entire input sequence (prompt) at once, generating key-value pairs for each token[4].

2. **Cache initialization**: These key-value pairs are stored in the KV cache, preparing it for subsequent token generation[4].

3. **Efficiency gain**: By prefilling the cache in a single forward pass, the model avoids the need to recompute these values for each new token generation, significantly speeding up the inference process[4].

4. **Memory preparation**: This step prepares the necessary memory structures for efficient access during the token generation phase[4].

Prefilling is particularly beneficial for long input sequences, as it allows the model to efficiently process and store the context information needed for generating coherent and contextually relevant outputs.

## 28. Only during prefilling phase we output multiple encodings and discard as we want the latest, explain?

During the prefilling phase, the model processes the entire input sequence and generates encodings (hidden states) for each token. However, only the final encoding is typically retained for the subsequent token generation phase. This process can be explained as follows:

1. **Multiple encodings**: As the model processes each token in the input sequence, it generates an encoding (hidden state) for that token[4].

2. **Discarding intermediate results**: While these intermediate encodings are necessary for computing the final state, they are not all needed for future token generation[4].

3. **Retaining the latest**: The model keeps only the final encoding, which represents the most up-to-date context information for the entire input sequence[4].

4. **Memory efficiency**: By discarding intermediate encodings, the model saves memory, as only the most relevant information (the final encoding) is retained for future use[4].

5. **Preparation for generation**: This final encoding serves as the starting point for generating new tokens, ensuring that the model has the most comprehensive context available[4].

This approach balances the need for processing the entire input sequence with the goal of maintaining an efficient memory footprint for subsequent token generation.

## 29. Explain the phases of pre-filling and token generation in KV cache?

The KV cache mechanism in large language models involves two main phases: pre-filling and token generation. Here's an explanation of each:

**Pre-filling Phase:**

1. **Input processing**: The model takes the entire input prompt or context and processes it in a single forward pass[4].
2. **KV pair generation**: For each token in the input, the model computes key-value pairs based on the token's embedding and model weights[1].
3. **Cache initialization**: These KV pairs are stored in the cache, creating a foundation for subsequent token generation[4].
4. **Final encoding**: The model produces a final encoding that represents the entire input context[4].

**Token Generation Phase:**

1. **Context utilization**: The model uses the final encoding from the pre-filling phase as the starting point[4].
2. **New token prediction**: Based on the cached KV pairs and the current context, the model predicts the next token[2].
3. **Cache update**: The KV pair for the newly generated token is computed and added to the cache[2].
4. **Iterative process**: Steps 2 and 3 are repeated for each new token generated, with the cache continuously expanding[2].

This two-phase approach allows for efficient processing of long sequences by leveraging pre-computed information stored in the KV cache, significantly reducing computational overhead during token generation.

## 30. Why is it that we are not masking the image embeddings and the text embeddings in the Vision model but we mask the placeholder text

In vision-language models, the treatment of image embeddings, text embeddings, and placeholder text differs due to the nature of these inputs and the model's architecture. Here's why masking is applied differently:

1. **Image embeddings**: These are not masked because they represent spatial features of the image that should be fully accessible to the model at all times. Masking them would limit the model's ability to understand the visual content[5].

2. **Text embeddings**: For regular text input, these are typically not masked in vision-language models as they provide context that should be available throughout the processing[5].

3. **Placeholder text**: This is masked to prevent the model from "cheating" by looking ahead at future tokens during training or inference. The mask ensures that the model can only attend to previous tokens when predicting the next one, maintaining the autoregressive nature of text generation[5].

The different treatment of these embeddings allows the model to leverage the full visual context while maintaining the appropriate constraints on text generation. This approach enables the model to effectively combine visual and textual information for tasks like image captioning or visual question answering.

Citations:
[1] https://training.continuumlabs.ai/inference/why-is-inference-important/key-value-cache
[2] https://cyrilzakka.github.io/llm-playbook/nested/kv-cache.html
[3] https://www.reddit.com/r/LocalLLaMA/comments/1czo625/how_much_memory_is_used_to_construct_the_kv_cache/
[4] https://arxiv.org/html/2410.03065v1
[5] https://huggingface.co/blog/kv-cache-quantization
[6] https://www.microsoft.com/en-us/research/blog/llm-profiling-guides-kv-cache-optimization/
[7] https://discuss.huggingface.co/t/generate-using-k-v-cache-is-faster-but-no-difference-to-memory-usage/31272
[8] https://arxiv.org/html/2405.14366v2

## 31. What is grouped query attention and multi attention and how is it different from vanilla multi head attention?

Grouped query attention (GQA) and multi-query attention (MQA) are variants of the standard multi-head attention mechanism that aim to reduce computational and memory costs while maintaining model performance.

In vanilla multi-head attention:
- Each attention head has its own set of query (Q), key (K), and value (V) projections
- This requires storing separate K and V tensors for each head

GQA and MQA modify this structure:

- MQA: Uses a single set of K and V projections shared across all heads, with separate Q projections for each head
- GQA: Groups heads together to share K and V projections, with separate Q projections per group

The key differences are:
1. Reduced parameter count and memory usage for K and V 
2. Decreased computational cost for attention calculations
3. Potential for increased inference speed, especially for autoregressive generation

GQA can be seen as a middle ground between MQA and vanilla multi-head attention, offering a trade-off between efficiency and expressiveness[1][2].

## 32. Why was Multi Attention proposed, to save memory bandwidth of GPU which is not up to the mark of the Tensor FLOPS of a GPU check Datasheets?

Multi-query attention (MQA) was proposed primarily to address the memory bandwidth limitations of GPUs, which often cannot keep up with the theoretical computational capabilities (FLOPS) of modern GPUs.

Key reasons for proposing MQA:

1. Memory bandwidth bottleneck: GPUs often have much higher computational capacity (FLOPS) than memory bandwidth. For example, the NVIDIA A100 GPU can sustain 19.5 TeraFLOPS of computation but only has approximately 2 terabytes/second memory bandwidth[2].

2. Arithmetic intensity: To fully utilize the GPU's computational capacity, we would need to ingest much more data than the memory bandwidth allows. For the A100, sustaining 19.5 TFLOPs would require ingesting at least 156 TB/s of data, far exceeding its 2 TB/s bandwidth[2].

3. Reducing memory transfers: MQA reduces the number of bytes read from memory per arithmetic operation in attention computation, increasing arithmetic intensity and leading to faster, more efficient attention computation[2].

4. KV-cache optimization: MQA reduces the size of the KV-cache, allowing for larger batch sizes and further improving efficiency[2].

By addressing these issues, MQA helps to better utilize the available GPU resources and improve overall performance, especially for inference tasks.

## 33. Why we do gradient checkpointing as it is faster to redo the computations rather than loading them from memory?

Gradient checkpointing is a technique used to reduce memory usage during neural network training by trading off computation for memory. The reason it's effective is that recomputing certain values can be faster than loading them from memory, especially on modern GPUs.

Key points:

1. Memory bandwidth limitation: GPUs are often bottlenecked by memory bandwidth rather than computational capacity. Loading large amounts of data from memory can be slower than recomputing values[1][2].

2. Computation vs. memory trade-off: Modern GPUs have high computational power (FLOPS) relative to their memory bandwidth. This makes recomputation relatively cheap compared to memory access[4].

3. Reduced memory footprint: By not storing all intermediate activations, gradient checkpointing significantly reduces the memory required during the backward pass[1].

4. Selective recomputation: Only certain layers or operations are checkpointed, balancing the trade-off between memory savings and computational overhead[1].

5. Scalability: Gradient checkpointing allows training of larger models or using larger batch sizes that would otherwise not fit in GPU memory[1].

By leveraging the high computational capacity of GPUs and reducing memory pressure, gradient checkpointing enables training of larger models and can sometimes lead to faster overall training times, despite the added computations.

## 34. The total computations that you do in attention is bottlenecked not by computations but by data transfer?

Yes, the attention mechanism in transformer models is often bottlenecked by data transfer rather than computations. This is due to the memory-intensive nature of attention operations and the limitations of GPU memory bandwidth.

Key points:

1. Memory bandwidth limitation: Modern GPUs have much higher computational capacity (FLOPS) than memory bandwidth. For example, the NVIDIA A100 GPU can perform 19.5 TeraFLOPS but has only about 2 terabytes/second memory bandwidth[2].

2. Attention operations: Attention involves several matrix multiplications and other operations that require loading large amounts of data (queries, keys, values) from memory[1][4].

3. Memory-bound computations: The matrix multiplications in attention, especially with small batch sizes or sequence lengths, are often memory-bandwidth-bound rather than compute-bound[1].

4. Data movement costs: Moving data between GPU memory and computation units can be more time-consuming than the actual computations, especially for operations with low arithmetic intensity[2][4].

5. KV-cache challenges: For autoregressive generation, the key-value cache grows with sequence length, further increasing memory pressure and data transfer requirements[2].

These factors make the attention mechanism in transformers primarily bottlenecked by data transfer rather than computational capacity. This realization has led to various optimizations like FlashAttention, multi-query attention, and other techniques aimed at reducing memory bandwidth requirements[1][2][4].

## 35. One way is to reduce the dimension or the number of head for the keys and values, they do not need to copy the key and values again for different query heads reduces data transfer which speeds up the computation

You're correct. This approach is indeed one of the key strategies used to optimize attention mechanisms, particularly in variants like multi-query attention (MQA) and grouped-query attention (GQA). Here's a more detailed explanation:

1. Reduced key and value dimensions:
   - By reducing the dimension or number of heads for keys and values, less memory is required to store these tensors[1][2].
   - This directly reduces the amount of data that needs to be transferred between GPU memory and computation units.

2. Shared keys and values:
   - In MQA and GQA, keys and values are shared across multiple query heads or groups[1][2].
   - This eliminates the need to copy key and value tensors for each query head, significantly reducing memory bandwidth requirements.

3. Decreased KV-cache size:
   - For autoregressive generation, the reduced size of keys and values leads to a smaller KV-cache[2].
   - This allows for larger batch sizes or longer sequences to fit in memory.

4. Improved arithmetic intensity:
   - By reducing data transfer while maintaining similar computation, the arithmetic intensity of the attention operation increases[2].
   - This better utilizes the GPU's computational capabilities relative to its memory bandwidth.

5. Faster computation:
   - The reduced data transfer directly translates to faster attention computation, especially during inference[1][2].
   - This is particularly beneficial for autoregressive generation tasks.

These optimizations, exemplified by techniques like MQA and GQA, effectively address the memory bandwidth bottleneck in attention mechanisms, leading to improved efficiency and performance, especially for large language models and long-context applications[1][2][4].

Citations:
[1] https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices
[2] https://fireworks.ai/blog/multi-query-attention-is-all-you-need
[3] https://www.reddit.com/r/deeplearning/comments/coobpx/gpu_tensor_cores_vs_flops_vs_memory_bandwidth/
[4] https://tridao.me/blog/2024/flash3/
[5] https://www.digitalocean.com/community/tutorials/gpu-memory-bandwidth
[6] https://www.determined.ai/blog/tp
[7] https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
[8] https://blog.paperspace.com/understanding-memory-bandwidth-benchmarks/


## 36. Reducing the number of heads in keys and values

Reducing the number of heads in keys and values does slightly decrease the expressive power and quality of the model, but not significantly. Grouped Query Attention (GQA) aims to balance efficiency and performance by reducing computational costs while maintaining most of the model's capabilities[1][2].

The reduction in quality is generally minimal because:

1. The queries still maintain full expressiveness with separate heads.
2. The shared key and value heads can still capture important patterns.
3. The model can adapt during training to compensate for the reduced heads.

While there is a small trade-off in model capacity, the benefits in terms of efficiency and reduced memory usage often outweigh the minor decrease in performance for many applications[1].

## 37. Grouped Query Attention and KV cache scaling

Grouped Query Attention (GQA) significantly reduces the size of the KV cache, which becomes a critical factor when scaling language models to handle longer sequences or deploy them on devices with limited memory[1][2].

The KV cache scaling problem arises because:

1. Traditional attention mechanisms store separate key and value tensors for each attention head.
2. As models grow larger and handle longer sequences, the memory required for the KV cache increases dramatically.
3. This can lead to out-of-memory errors or limit the model's ability to process long inputs.

GQA addresses this by:

1. Reducing the number of key and value heads, thus decreasing the size of the KV cache.
2. Allowing models to handle longer sequences or run on devices with less memory.
3. Enabling more efficient inference and deployment of large language models.

While GQA helps with scaling, it's important to note that very large models or extremely long sequences may still face memory constraints, requiring additional optimization techniques[2].

## 38. Positional encoding

Positional encoding is a technique used in Transformer models to provide information about the position of tokens in a sequence[3][4]. It's essential because the self-attention mechanism in Transformers doesn't inherently consider the order of input elements.

Key aspects of positional encoding include:

1. It adds unique position-dependent signals to each token's embedding.
2. It allows the model to understand and utilize the sequential order of inputs.
3. The encoding can be learned or fixed using predefined functions.

A common approach is to use sine and cosine functions of different frequencies:

$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

Where `pos` is the position and `i` is the dimension[3].

## 39. Rotation matrix

A rotation matrix is a matrix that represents a rotation in Euclidean space. In the context of positional encodings, rotation matrices are used to transform embeddings based on their positions in the sequence[5].

Key properties of rotation matrices include:

1. They preserve the length of vectors and the angles between them.
2. They can be parameterized by a single angle in 2D or by Euler angles in 3D.
3. In 2D, a rotation matrix has the form:

$$R(\theta) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}$$

Rotation matrices are particularly useful in rotary positional encoding, where they help encode positional information in a way that's easily incorporated into the attention mechanism[5].

## 40. Rotary positional encoding

Rotary Positional Encoding (RoPE) is an alternative to traditional additive positional encodings. It applies position-dependent rotation to the query and key vectors in the attention mechanism[5].

Key features of RoPE include:

1. It doesn't directly add positional encodings to token embeddings.
2. It uses a rotation matrix to transform query and key vectors based on their positions.
3. The dot product between rotated vectors becomes a function of both their content and relative positions.

The rotation is applied as follows:

$$q' = R(\theta)q, k' = R(\theta)k$$

Where R(θ) is a rotation matrix, and θ is a function of the position.

This approach has several advantages:

1. It naturally captures relative positional information.
2. It allows for easy extrapolation to longer sequences.
3. It preserves the dot product's norm, which is beneficial for stability in training.

RoPE has shown promising results in various natural language processing tasks, offering an elegant way to incorporate positional information into Transformer models[5].

Citations:
[1] https://www.geeksforgeeks.org/positional-encoding-in-transformers/
[2] https://www.sciencedirect.com/topics/computer-science/positional-encoding
[3] https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
[4] https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
[5] https://www.youtube.com/watch?v=ZMxVe-HK174
[6] https://www.machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

## 41. Does rotary encoding have decaying effect and what is it and how it helps prove that as distance increases the score decrease?

Rotary encoding, also known as RoPE (Rotary Position Embedding), does have a decaying effect, which is a key feature that helps demonstrate how the relevance of tokens decreases as the distance between them increases.

The decaying effect in rotary encoding refers to the phenomenon where the dot product between two position-encoded vectors decreases as the distance between their positions increases. This property is inherent to the mathematical formulation of RoPE.

Here's how it works:

1. RoPE applies a rotation to token embeddings based on their position in the sequence.
2. As the distance between two tokens increases, the angle of rotation between their embeddings also increases.
3. The dot product between two rotated vectors naturally decreases as the angle between them increases.

This decaying effect helps prove that as distance increases, the score (or relevance) between tokens decreases because:

1. In attention mechanisms, the relevance between tokens is often computed using dot products.
2. The decreasing dot product due to RoPE directly translates to a decreasing attention score.
3. This aligns with the intuition that tokens farther apart in a sequence are generally less relevant to each other.

The decaying effect of RoPE is particularly useful in language models as it allows the model to implicitly capture relative positional information without the need for explicit position embeddings[1][2].

## 42. What is top_p sampling in inference and how and why is it used if at all and how to implement it?

Top_p sampling, also known as nucleus sampling, is a text generation strategy used during inference in language models. It aims to balance between diversity and quality in the generated text.

How it works:

1. The model computes the probability distribution over all tokens in the vocabulary for the next token.
2. Tokens are sorted by probability in descending order.
3. The cumulative probability is calculated, adding up probabilities until it reaches or exceeds the specified top_p value.
4. Only the tokens within this cumulative probability are considered for sampling.

Why it's used:

1. It allows for more dynamic and context-dependent token selection compared to fixed top-k sampling.
2. It helps maintain output diversity while avoiding highly improbable tokens.
3. It can adapt to different scenarios where the probability distribution may be more or less concentrated.

Implementation:

```python
def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits
```

Top_p sampling is widely used in modern language models to generate more natural and diverse text while maintaining coherence[3][4].

## 43. What is temperature in inference and how and why is it used if at all and how to implement it? Introducing noise in the score but restricted to the top_p score and smoothing the probability curve if visualized

Temperature is a hyperparameter used during inference in language models to control the randomness of token selection. It modifies the probability distribution of the next token prediction.

How it works:

1. The logits (raw scores) produced by the model are divided by the temperature value.
2. A lower temperature (< 1) makes the distribution sharper, favoring high-probability tokens.
3. A higher temperature (> 1) makes the distribution more uniform, increasing the chances of selecting lower-probability tokens.

Why it's used:

1. To control the trade-off between diversity and determinism in generated text.
2. Lower temperatures produce more focused and coherent outputs.
3. Higher temperatures encourage more creative and diverse outputs.

Implementation:

```python
def apply_temperature(logits, temperature):
    return logits / temperature

def sample_with_temperature(logits, temperature=1.0):
    logits = apply_temperature(logits, temperature)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

When used in combination with top_p sampling, temperature is applied before the top_p cutoff:

1. First, apply temperature to modify the logits.
2. Then, perform top_p sampling on the temperature-adjusted logits.

This combination allows for fine-tuned control over both the overall randomness (temperature) and the focus on more likely tokens (top_p).

Visualizing the effect:

- Lower temperature: The probability curve becomes steeper, with higher peaks for the most probable tokens.
- Higher temperature: The probability curve becomes flatter, distributing probability more evenly across tokens.

Temperature is a crucial parameter for controlling the creativity and coherence of generated text in many applications of language models[4][5].

Citations:
[1] https://huyenchip.com/2024/01/16/sampling.html
[2] https://docs.ai-solutions.ext.hpe.com/products/gen-ai/latest/get-started/glossary/model-params-top-p/
[3] https://community.openai.com/t/a-better-explanation-of-top-p/2426
[4] https://becomingahacker.org/understanding-key-ai-language-model-parameters-top-p-temperature-num-beams-and-do-sample-9874bf3c89ae?gi=d27378ac63a6
[5] https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/
[6] https://promptengineering.org/prompt-engineering-with-temperature-and-top-p/
[7] https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542
[8] https://docs.aws.amazon.com/bedrock/latest/userguide/inference-parameters.html






















## 🎉 Final Words

Remember, with great power comes great responsibility. Use Vision-GPT wisely, and may your code always compile on the first try!

Happy coding, and may the vision be with you! 🧙‍♀️🔮

Citations:
[0] https://github.com/hkproj/pytorch-paligemma
[1] https://arxiv.org/abs/2403.09027
[2] https://arxiv.org/abs/2305.04790
[3] https://www.v7labs.com/blog/chatgpt-with-vision-guide
[4] https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest
[5] https://www.bentoml.com/blog/multimodal-ai-a-guide-to-open-source-vision-language-models
[6] https://roboflow.com/model-feature/multimodal-vision
[7] https://devblogs.microsoft.com/ise/multimodal-rag-with-vision/
[8] https://community.openai.com/t/how-to-do-few-shot-prompting-with-images-in-gpt-4-vision-api-structure-can-someone-provide-a-code-to-do-so/691101