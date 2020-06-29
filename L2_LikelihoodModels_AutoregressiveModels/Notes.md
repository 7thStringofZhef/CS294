## Motivation
Problems to solve:
* Data synthesis: text, video, speech
* Compression
* Anomaly detection: e.g., identify whether images are likely to be misclassified

## Likelihood models
* Estimate the probability of the data (p_data) from samples {x_1,..., x_n} ~ p_data
    * Learn a distribution that allows computation of p(x) and sampling x ~ p
    * For our models today, p(x) is cheaper than sampling x

* What do we want?
  * Estimate distributions of _complex, high-dimensional_ data
  * Computational and statistical efficiency
  * Both expressive/generalizable and fast/sample-efficient

* Histograms
    * Take discrete random samples \in {1..100]
    * Plot as a histogram
    * Inference: Look up in table
    * Sampling: 
        * Compute cumulative probability function F_i = p_0 + ... + p_i for all i \in {1...k}
        * Sample random # from 0-1. 
        * Return smallest i such that u \le F_i
    * **Weaknesses**
        * Counting fails with too many bins. Even MNIST images have 784 pixels (2^784 probabilities). No training set will cover this space
        * Each image only influences 1 parameter, will not generalize
        * Even for 1 variable, suffers poor training -> test transfer
    * Could fit a smoother distribution to training set rather than jagged histogram
* Function approximators
    * Use a function parameterized by \theta (p_theta) to approximate p_data
    * Many choices for this
    * **Search problem**: Find \theta that minimizes loss over samples
* Maximum Likelihood Objective
    * Intuitively: Choose a theta that maximizes likelihood of observing samples in training
    * Minimize sum of -log probability(x) for all x
    * Alternative formulation: Minimize KL divergence between empirical data distribution and model
    * How to minimize?
        * Local search via SGD

* Designing a model
    * Use neural network here for expressiveness
    * Output between 0 and 1 for image probability
        * Ensure sums to 1
    * Bayesian Networks:
        * Directed acyclic graph over variables in data, modeled as conditional distributions
        * Probability of an observed point is conditioned on other variables, using assumptions tor reduce search space
        * Trend is to condition on everything, but make it representable by representing not as a probability table but a parameterized function
    * Autoregressive model: Basic chain rule model.
        * We use priors like convolutional layers in our network architecture
        * Two approaches to parameterizing with neural networks
            1) RNN: 
                * Say x is a sequence of characters, probability of a string is sum of log p(x_i)
                * RNN estimates probability of next variable given previous using hidden state
                * Good for text, not so much images like MNIST (looking at one pixel after another)
                * If you also give it the pixel coordinates (x,y), does better
            2) Masking models:
                * Parallelized computation for all conditionals. Share parameters across time
                * Masked MLP(MADE) (or ConvNet or self-attention)
                * MADE
                    * Typical approach was to convert noised images to unnoised.
                    * But given an actual image, couldn't output a probability
                    * Want to normalize output. Need to condition outputs
                    * Use masks such that each output only sees values from certain inputs
                    * Principle: 
                        * MADE is a chain rule model (x_1...x_6)
                        * Outputting p(x_1), p(x_2|x_1), p(x_3|x_1, x_2), etc
                        * Can have arbitrary number of hidden units, so long as signals can only reach outputs they're supposed to
                        * This is essentially a mask applied to erase invalid paths.
                        * Could change mask, give different orderings, even with same parameters
                    * MADE works well on MNIST. Too many orderings might make network too unexpressive. Larger networks tend to be able to use more orderings
                    * Different orderings can let you avoid missing data by conditioning on inputs you actually have
                    * Each individual output is normalized, and so the full output is a proper joint distribution
                    * Training: Maximize log probability for each x^(i) where x^(i) = x_1, x_2, etc
                    * Sampling: Sample using output p(x_1), feed back to input, use output x_2, feed back, use x_3, etc. Slow process. 10 minutes to generate 1 MNIST sample
                * Masked Convolutions
                    * Masked temporal (1D) convolutions: Use same parameterized filters, but mask 
                    * In MADE: Receptive field of x_k is x_1...k-1
                    * In 1D: Receptive field depends on filter configuration
                    * WaveNet: For sampling, keep conditioning on previous output. Filters are dilated convolution to look further back
                        * Uses GRU blocks and skip connections for greater expressivity
                    * Sucks at MNIST, much better when given (x,y)
                    * 2D version (PixelCNN and variations)
                        * Seems silly to flatten images into 1D.
                        * Mask the filters directly (e.g., 3x3 filter with 4 1s means you only see those)
                        * Sample pixel after pixel, keep feeding back in
                        * Problem: Blind spot forms, can't see pixels that come after you. Bottom left of image has especially large blind spot
                            * Architecture choice: Set up a 2*3 receptive field, but 1 pixel above, so not seeing self. But we introduced a blind spot to the left
                            * Then add a separate 1D filter on just that row to clear that spot up
                            * 2*3 comes before horizontal, so it can receive that output
                        * Gated PixelCNN: Use Gated ResNet block to learn weights for this split feature set
                        * PixelCNN++: Output was a 256-way softmax, but that's silly. We know nearby pixel values tend to co-occur
                            *  Use difference between two sigmoids, mixture of multiple logistic distributions
                            *  Use skip connections across convolutional layers to introduce further-back dependencies
                    * Masked Attention 
                        * [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need)
                        * Attention:
                            * We have some vector query
                            * Take inner product of query with keys in network, higher inner product is better match
                            * Attention is weighted value of match, subtract out max for stability
                        * Processing word, could have many meanings (queries), queries find other words, which shape our interpretation (multiple hypotheses)
                        * Self-attention
                            * Bring in all other queries, mask what we don't want
                            * Why is this better than MADE?
                                * Parameter sharing across all spots. Highly connected, but low parameter
                                * PixelSNAIL: Conv + self-attention. Has a smaller blind spot than the other PixelCNN things (which lose precision over length of chain)
                    * Other ideas
                        * Class-conditional CNN: Condition on one-hot label, which converts to bias on filters
                        * Hierarchical AR models with Auxiliary Decoders: Start by generating low-dim image, use as input to go higher, then higher, etc
                        * Grayscale PixelCNN
                * So
                    * Training is very simple, expressive, can capture our priors
                    * Sampling sucks, we have to keep feeding outputs back in to generate one image
                        * Coarse generation (Multiscale PixelCNN): Generate some subset of pixels, then generate sub-images on each (not conditioning on pixels outside of sub-image), can parallelize
                        * Less expressive, but much faster. Good if we're clever
                        * Scaling AR video models
                        * Caching: Save activations from previous layers, reuse activations from previous passes(fast-pixel-cnn). No loss
                * What about representation learning? For RL to work more quickly
                    * RNN gives easy latent representation. What is the equivalent for these?
                    * Difficult: Random input at each pixel (softmax)
                    * Fisher score:
                        * Gradient of log probability at data point
                        * Intuitively: Multi-modal distribution. Mode my data point is in should be mode with higher log probability
                        * Use this as latent space. 
                        * In practice, better than interpolation.
                        
                    
                                
    
    
    