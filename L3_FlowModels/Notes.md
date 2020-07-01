## Goal
* Fit a density model p_theta(x) with continuous x \in R_n
* Good fit, be able to sample, get a latent representation, get prob
* Main differences are continuous and focus on latent representation

## Notes
* What is a probability density model?
    * P(x\in [a, b]) is probability of landing in interval a -> b, integral of curve
* Given data, how do we fit?
    * Choose theta to minimize -log probabilities of observed data (similar)
    * Could fit a mixture of gaussians (theta is set of means and variances plus mixture weights)
        * Bad for high-dimensional data. 
        * E.g., a gaussian center in each image is a fuzzy image, not a totally different image
* Challenges
    * Can't just put a softmax at the end (infinite possible outputs), so how to normalize?
        * Output would drive to positive infinity
    * How to sample? Latent representation
* 1-D case
    * Instead of outputting p_theta(x), output scalar z = f_theta(x)
    * If z is set to come from **normal distribution**, it's **normalizing flow**, z ~ N(0,1)
    * We're avoiding normalization issue by outputting this z with predefined bounds
    * How to relate z and p_theta?
        * f_theta turns x into z
        * f_theta must be differentiable and invertible to convert (always goes up or down in 1-D, steepness determines density)
    * Training flows
        * Can sub z into objective, so now:
        * max sum(log p_theta(x^i)) = max sum(log p_z(x^i) + log|dz/dx(x^i)|)
        * Can optimize with SGD
    * What do you pick as distribution for z? What about neural network for f_theta?
    * Sampling
        * Sample z from gaussian z ~ p_Z(z). Z is our latent space
        * x = invert(f_theta(z))
    * Examples: Could try to map z to uniform, to Beta(5,5), to Gaussian
    * Typically map z to uniform or gaussian. We're learning the model f_theta, not so much p_z
    * Practical considerations
        * Ideal functions (invertible, differentiable)
            * CDF (Gaussian/logistic mixtures, anything 0-1)
            * Neural networks
                * If each layer is a flow, composition is a flow
                * ReLU isn't invertible
                * Sigmoid is good. tanh is ok (but -1,1 instead of 0,1)
    * How general are flows?
        * CDF measures probability mass up to x. We can sample from uniform distribution over z, get x with invertible
        * We can flow to anything, and inverse flow is flow. Any smooth p(x) into any smooth p(z)
* 2-D flows
    * If 1-D is x_1 -> z_1 -> f_theta(x_1)
    * 2-D is x_2 -> z_2 -> f_theta(x_1, x_2)
    * Training will therefore be sequential
* N-D (x and z in same R_d)
    * AR Flows
        * Autoregressive (fast training, slow sampling) and inverse autoregressive (slow training, fast sampling)
        * Autoregressive involve conditioning on past inputs. Sampling is invertible mapping z to x
        * p_theta(x) = p(f_theta(x))|det(df/dx)|
        * Can swap flows from x->z to z->x because of invertible f
            * z_1 = inverse(f(x_1)), z_2 = inverse(f(x_2; z_1)), etc
            * x_1 = f(z_1), x_2 = f(z_2;z_1)
            * In original implementation, since training was all x, and we have all x, then it's fast. Sampling needs z, so slow
            * Here, sampling only depends on z, so that's easy, while training needs sequential
            * Pick depending on needs
        * Parallel WaveNet, IAF-VAE use inverse
        * Naively, both are as deep as # variables. Can use RNN/masking as last lecture
    * Can we change many variables x_j -> z_j at the same time?
        * Imagine dz and dx as volumes; for simplicity, say a diagonal matrix (dz_1/dx_1, dz_2/dx_2, etc along diagonal)
        * Can rescale along each coordinate this way. Product of the 3 entries is same as determinant (in this simple case)
        * **New requirement**: Jacobian determinant must be easy to compute
        * So now we can do flows without AR. Can compose in network
    * Affine flows: f(x) = A^-1(x-b)
        * Sampling: x = Az + b, z ~ N(0,1)
        * Training: Jacobian of f is A^-1, not generally efficient
    * Elementwise flows: 
        * Each element flows independently, easy to compute Jacobian, but independence assumption loses expressivity
    * NICE/[RealNVP](https://arxiv.org/pdf/1605.08803.pdf):
        * Split variables in half (i.e., x_1:d/2, x_d/2+1:end)
        * First half: z_1:d/2 = x_1:d/2
        * Second half: Multiply x by s_theta(x_1:d/2), add t_theta(x_1:d/2) offset
            * We already know x_1:d/2 from first half pass
            * both s and t can be networks
            * In NVP, s_theta is "scale"
        * Essentially, data-parameterized element-wise flows
        * Can we easily compute determinant?
            * Jacobian dz/dx looks like [I, 0]/[dz_d/2:d /dx_1:d/2, diag s_theta(x_1:d/2)]
            * Determinant of a matrix with 0 above or below diagonal (**triangular**) is just product of diagonal, so just Prod(s_theta(x_1:d/2))
            * Because inverse does not require inverse of s or t, they can be arbitrarily complex
            * In RealNVP, alternates ordering of dimensions in each layer so that every input has a chance to be changed
            * NICE was RealNVP without the "scale" term
        * Example:
            * Input a 28*28*3 x image, get out 28*28*3 z
            * Sample new z to get x
            * What do first and second half mean in an image? Arbitrary, could do across channels, locations, etc. It does matter, though
            * In RealNVP
                * Checkerboard pattern by location (alternating pixels belonging to each half)
                * As convolutional filters are applied, # channels increases, size decreases. Then condition alternately on first half and second half of channels 
        * Could we break it up in different ways than half?
            * 1 vs everything else (AR). Probably is half-half to maximize information at every stage
    * Choice of coupling transformation: What invertible transformation f should we use?
        * x_i = f_theta(z_i; parent(x_i))
        * Affine is most commonly used (NICE, RealNVP): x_i = z_i * a_theta(parent(x_i)) + b_theta(parent(x_i))
        * More complex and general transformations are possible
            * [Flow++](https://arxiv.org/pdf/1902.00275.pdf): Mixture of gaussians/logistics
                * Also adds self-attention
            * Other stuff for further research
            * [Glow](https://openai.com/blog/glow/): Simplifies and enhances RealNVP architecture. Each step of flow has
                1) Actnorm: Serves same purpose as batch normalization (transforms input using scale and bias for each channel), but for batches of 1
                2) Invertible 1x1 convolution: 
                    * In RealNVP: Permute inputs by reversing ordering across channel. Split into A and B. Feed in A, use to transform B, combine the two.
                    * This is strict A -> B -> A -> B, etc. Not necessarily optimal permutation
                    * In GLOW: Permutation is equivalent to 1x1 convolution, replace fixed permutation with learned convolutions
                3) Affine coupling layer a la RealNVP
            * [FFJORD](https://arxiv.org/pdf/1810.01367.pdf): Continuous time dynamics, using ODEs
* Dequantization
    * Imagine you have training data, train flow model on this discrete distribution. Global optimum will be to have sharp peaks. What to do?
        * Common problem, RGB images with 255 values will have some crazy peaks
    * Could parameterize flow not to generate peaks
    * Could do **dequantization**
        * An underlying data model of a 255-value image is more like rough density over each of the 3 channels
        * Probability assigned to specific value should really be integrated around region
        * Could uniformly perturb my data (with noise from -0.5 to 0.5) before input to get what I want (add noise)
* Future directions
    * Flow models have fewest papers
    * Goals: Fast sampling, inference, training, good samples and compression
    * How do we design flows? Architecture design
            
        
        
        
         
    
        
        
     