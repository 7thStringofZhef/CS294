# What is Deep Unsupervised Learning?
* "Capturing patterns in data in a label-free way"
* Two categories:
  1) _Generative_: Make data like the data, recreate distribution
  2) _Self-supervised_: Extract patterns that let us use the data to solve a task
* Why?
  * Brains must use unsupervised learning, have more parameters than data (Hinton)
  * LeCun: Need an enormous amount of data to make general AI, unsupervised is only way to get it
  * Compression: Finding all patterns is equivalent to determining the shortest representation of data (low **Kolmogorov complexity**)
  * **Solomonoff Induction**: Given all possible computer programs that could generate the data, prefer the shortest
  * **AIXI**: Theoretical RL agent that considers reward over every computable environment, weights them by belief state 
  * Applications
    * Generate new data
    * Conditional synthesis (e.g., generate speech based on some text you wrote)
    * Compression
    * Improve downstream tasks by reducing the data requirement (learned representations are more efficient at leveraging labeled data)
    * Building block for other things
  * Examples
    * MNIST digit generation
    * Face generation (and deep fakes)
    * Image translation (e.g., horse to zebra)
    * Photorealistic image "dreaming"
    * Audio generation (WaveNet), Video generation (DVD-GAN), Text generation (GPT-2/3)
    * Math generation (Char-rnn) (Maybe I can escape LaTeX!)
    * Lossless Compression (Sparse Transformer) and effective lossy compression (WaveOne) 
    * Downstream Tasks
      * Sentiment Detection: LSTM discovered sentiment in text during unsupervised use
      * NLP (BERT), GLUE dataset
      * Vision on the PASCAL dataset, using self-supervised contrastive pre-training
# Active Research Areas
* Density Modeling
* Flows
* VAE
* Unsupervised Learning for RL
      