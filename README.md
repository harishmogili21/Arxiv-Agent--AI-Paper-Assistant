# 📚 Arxiv Agent - AI-Powered Paper Assistant

An intelligent AI agent that helps you interact with Arxiv papers using Google's Gemini API, advanced embeddings, and Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **🔍 Smart Paper Search**: Search for papers on Arxiv using natural language queries
- **🧠 Hybrid RAG System**: Combines paper knowledge with general AI knowledge for complete answers
- **📊 Advanced Vector Search**: Enhanced semantic similarity with multi-signal reranking
- **🎯 Smart Search Button**: Automatically finds relevant papers for your questions
- **�  Multiple Interfaces**: Beautiful web UI (Streamlit) and powerful CLI
- **�  Intelligent Paper Management**: View, organize, and analyze your paper collection
- **🎨 High-Contrast UI**: Improved readability with professional design
- **🔧 Easy Setup**: Simple configuration with environment variables
- **💡 Search Suggestions**: Get intelligent suggestions for better paper discovery

## ✨ Latest Enhancements

### 🧠 Hybrid Response System
- **Complete Answers**: Always provides comprehensive responses, even when papers don't contain specific information
- **General Knowledge Fallback**: Uses AI knowledge to answer fundamental questions
- **Paper Integration**: Seamlessly combines paper insights with general knowledge

### 🔍 Advanced Search & Ranking
- **Multi-Signal Reranking**: Combines semantic similarity, keyword matching, title relevance, and abstract relevance
- **Smart Chunking**: Sentence-aware text splitting for better semantic coherence
- **Intelligent Suggestions**: Automatic search term suggestions for better paper discovery

### 🎨 Enhanced User Experience
- **High-Contrast UI**: Professional design with excellent readability
- **Smart Search Button**: One-click paper discovery and analysis
- **Better Chat Interface**: Clear message distinction and improved formatting
- **Real-time Feedback**: Detailed logging and progress indicators

## 🛠️ Technology Stack

- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Gemini Embedding Model (`embedding-001`)
- **Vector Search**: FAISS for efficient similarity search with advanced reranking
- **RAG**: Enhanced Retrieval-Augmented Generation with hybrid knowledge
- **Web UI**: Streamlit for beautiful, interactive interface
- **Paper API**: Arxiv API for paper retrieval

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- Internet connection for Arxiv access

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd arxiv-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Google API key:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## 🚀 Quick Start

### Web Interface (Recommended)

Start the Streamlit web application:

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

### Command Line Interface

Start interactive mode:

```bash
python cli.py --interactive
```

Or use specific commands:

```bash
# Search for papers
python cli.py --search "machine learning"

# Load papers and ask a question
python cli.py --load "transformer models" --question "What are the main findings?"

# Just ask a question (if papers are already loaded)
python cli.py --question "How does this compare to other approaches?"
```

## 📖 Usage Guide

### 1. Web Interface

1. **Search Papers**: Use the sidebar to search for papers on Arxiv
2. **Load Knowledge Base**: Click "Load Papers into Knowledge Base" to prepare for chat
3. **Chat**: Ask questions about the loaded papers in the chat tab
4. **Browse**: View paper details and access PDFs in the papers tab

### 2. Command Line Interface

#### Interactive Mode Commands:
- `search <query>` - Search for papers
- `load <query>` - Load papers into knowledge base
- `papers` - Show currently loaded papers
- `clear` - Clear the knowledge base
- `status` - Show agent status
- `help` - Show help message
- `quit` - Exit the application

#### Direct Commands:
```bash
# Search for papers
python cli.py --search "deep learning"

# Load papers and chat
python cli.py --load "computer vision" --question "What are the latest advances?"

# Interactive mode
python cli.py --interactive
```

## 🔍 Example Usage

### Example 1: Researching a Topic

```bash
# Start interactive mode
python cli.py --interactive

# Search for papers on transformers
search transformer models

# Load papers into knowledge base
load transformer models

# Ask questions
What are the main advantages of transformer architecture?
How do transformers compare to RNNs?
What are the latest improvements in transformer models?
```

### Example 2: Quick Analysis

```bash
# Load papers and get immediate analysis
python cli.py --load "machine learning" --question "What are the current trends in ML?"
```

### Example 3: Web Interface Workflow

1. Open the web interface
2. Search for "reinforcement learning"
3. Load papers into knowledge base
4. Ask: "What are the main challenges in RL?"
5. Follow up: "How do these papers address exploration vs exploitation?"

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Arxiv API     │    │   Gemini API    │    │   FAISS Index   │
│                 │    │                 │    │                 │
│ • Paper Search  │───▶│ • LLM (Gemini)  │    │ • Vector Search │
│ • Metadata      │    │ • Embeddings    │    │ • Similarity    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Arxiv Agent Core                            │
│                                                                 │
│ • Paper Retrieval                                              │
│ • Content Extraction                                           │
│ • Text Chunking                                                │
│ • RAG Pipeline                                                 │
│ • Response Generation                                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   CLI           │
│   (Streamlit)   │    │   (Terminal)    │
└─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required)
- `ARXIV_EMAIL`: Your email for Arxiv API (optional)

### Customization

You can modify the following parameters in `arxiv_agent.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `top_k`: Number of search results (default: 5)
- `max_results`: Maximum papers to search (default: 10)

## 📊 Performance Tips

1. **Search Queries**: Use specific, descriptive search terms
2. **Paper Loading**: Load 5-10 papers for optimal performance
3. **Question Specificity**: Ask specific questions for better responses
4. **Knowledge Base**: Clear the knowledge base when switching topics
5. **API Limits**: Be mindful of Gemini API rate limits

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   ❌ Failed to initialize Arxiv Agent: Google API key is required
   ```
   **Solution**: Set the `GOOGLE_API_KEY` environment variable

2. **No Papers Found**:
   ```
   ❌ No papers found. Try a different search query.
   ```
   **Solution**: Try different search terms or check your internet connection

3. **Embedding Errors**:
   ```
   ❌ Error getting embeddings
   ```
   **Solution**: Check your API key and internet connection

4. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'google.generativeai'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

### Debug Mode

Enable debug logging by modifying the logging level in `arxiv_agent.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for LLM and embedding capabilities
- Arxiv for providing access to research papers
- FAISS for efficient vector search
- Streamlit for the web interface framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section
2. Review the logs for error messages
3. Open an issue on GitHub
4. Check the documentation

---

**Happy researching! 🎉** 