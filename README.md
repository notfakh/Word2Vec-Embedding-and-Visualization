# Word2Vec-Embedding-and-Visualization
Trains a Word2Vec model on sample sentences to generate word embeddings and visualizes relationships between words using PCA, showing semantic similarities in a 2D plot.

## 📋 Project Overview

This project demonstrates how to train a Word2Vec model to create word embeddings that capture semantic meaning. It uses PCA (Principal Component Analysis) to reduce high-dimensional word vectors to 2D for intuitive visualization, allowing you to see how semantically similar words cluster together.

## 🎯 What is Word2Vec?

Word2Vec is a neural network-based technique that learns distributed representations of words. Words with similar meanings are positioned close to each other in the vector space, enabling powerful semantic operations like:

- **Word Similarity**: Finding similar words
- **Word Analogies**: "king" - "man" + "woman" ≈ "queen"
- **Semantic Relationships**: Understanding context and meaning

## 🔑 Key Features

- ✅ Loads pre-trained Text8 corpus from Gensim
- ✅ Trains custom Word2Vec model on sample sentences
- ✅ Generates word embeddings (50-dimensional vectors)
- ✅ Applies PCA for 2D visualization
- ✅ Creates scatter plot with word labels
- ✅ Demonstrates semantic clustering

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/notfakh/word2vec-visualization.git
cd word2vec-visualization
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

Run the script:
```bash
python word2vec_embeddings.py
```

The script will:
1. Load the Text8 corpus
2. Train a Word2Vec model on custom sentences
3. Generate word embeddings
4. Reduce dimensions using PCA
5. Display a 2D scatter plot visualization

## 📊 Sample Word Groups

The model is trained on semantically related word groups:

| Category | Words |
|----------|-------|
| **Royalty** | king, queen, prince, princess, throne, kingdom |
| **Fruits** | apple, orange, banana, fruit |
| **Pets** | dog, cat, pet, animal |
| **Gender** | man, woman |

## 📈 Expected Output

A scatter plot showing:
- Words positioned based on semantic similarity
- Related words clustered together (e.g., "king" near "queen")
- Gender and category relationships visible in 2D space

### Visualization Interpretation:
- **Close proximity** = Similar meaning/context
- **Distinct clusters** = Different semantic categories
- **Linear relationships** = Analogical relationships (e.g., king-man, queen-woman)

## 🔍 Model Parameters

```python
Word2Vec(
    sentences,           # Training data
    min_count=1,        # Minimum word frequency
    vector_size=50,     # Dimensionality of word vectors
    workers=4           # Number of CPU cores for training
)
```

### Parameter Tuning:
- **vector_size**: Higher = more detailed (try 100, 200, 300)
- **min_count**: Filter rare words (increase for larger datasets)
- **workers**: Utilize more CPU cores for faster training
- **window**: Context window size (default=5)
- **sg**: 0=CBOW, 1=Skip-gram

## 🛠️ Customization

### Add Your Own Words

Modify the `sentences` list:
```python
sentences = [
    ["your", "custom", "words", "here"],
    ["machine", "learning", "deep", "neural"],
    # Add more sentences...
]
```

### Change Visualization Words

Edit the words to visualize:
```python
words = ["king", "queen", "prince", "princess", "throne"]
```

### Adjust Vector Dimensions

Change embedding size:
```python
model = Word2Vec(sentences, vector_size=100)  # Higher dimension
```

## 📚 Dataset Information

**Text8 Corpus:**
- Pre-processed Wikipedia dump
- ~100MB of clean text
- Loaded via Gensim's API
- Used for pre-training context (optional)

**Custom Sentences:**
- 8 sentences with related word groups
- Demonstrates semantic relationships
- Minimal training data for quick experimentation

## 🎨 Visualization Details

### PCA Dimensionality Reduction
- Reduces 50D vectors to 2D
- Preserves maximum variance
- Makes relationships visible to human eye

### Scatter Plot Features
- Each point = one word
- X/Y axes = principal components
- Annotations show word labels
- Spatial distance = semantic similarity

## 💡 Use Cases

- **Educational**: Understanding word embeddings
- **NLP Projects**: Text preprocessing and feature extraction
- **Semantic Analysis**: Finding word relationships
- **Research**: Exploring language structure
- **Chatbots**: Improving natural language understanding

## 🔬 Extending the Project

Ideas for enhancement:

1. **Load Pre-trained Models**
   - Use Google's Word2Vec (3M words)
   - Load GloVe embeddings
   
2. **Word Arithmetic**
   ```python
   result = model.wv.most_similar(
       positive=['king', 'woman'],
       negative=['man']
   )
   # Should return "queen"
   ```

3. **Similarity Calculations**
   ```python
   similarity = model.wv.similarity('king', 'queen')
   ```

4. **3D Visualization**
   - Use PCA with n_components=3
   - Create interactive 3D plots with Plotly

5. **t-SNE Instead of PCA**
   - Better for non-linear relationships
   - More accurate clustering

## 🤝 Contributing

Contributions are welcome! Enhancement ideas:

- Add more diverse training sentences
- Implement t-SNE visualization
- Create interactive plots with Plotly
- Add word similarity calculations
- Include word analogy examples
- Compare different embedding sizes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Fakhrul Sufian**
- GitHub: [@notfakh](https://github.com/notfakh)
- LinkedIn: [Fakhrul Sufian](https://www.linkedin.com/in/fakhrul-sufian-b51454363/)
- Email: fkhrlnasry@gmail.com

## 🙏 Acknowledgments

- Gensim library for Word2Vec implementation
- Tomas Mikolov et al. for Word2Vec algorithm
- Scikit-learn for PCA implementation
- Text8 corpus contributors

## 📚 References

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Word2Vec Tutorial](https://rare-technologies.com/word2vec-tutorial/)

## 📧 Contact

For questions or suggestions:
- Open an issue in this repository
- Email: fkhrlnasry@gmail.com
- Connect on LinkedIn

---

⭐ If this project helped you understand word embeddings, please give it a star!

## 🎓 Learning Outcomes

After exploring this project, you'll understand:
- How Word2Vec creates word embeddings
- Semantic relationships in vector space
- Dimensionality reduction with PCA
- Visualizing high-dimensional data
- Training custom NLP models

**Perfect for:** NLP beginners, students, and anyone curious about word embeddings!
