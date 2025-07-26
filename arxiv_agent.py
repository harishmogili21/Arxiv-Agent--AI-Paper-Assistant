import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
import arxiv
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Data class to store paper information"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published_date: str
    categories: List[str]
    pdf_url: str
    content: Optional[str] = None

@dataclass
class SearchResult:
    """Data class for search results"""
    paper: Paper
    similarity_score: float
    relevant_chunks: List[str]

class ArxivAgent:
    """Arxiv Agent with RAG capabilities using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Arxiv Agent"""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize FAISS index for vector search
        self.dimension = 768  # Gemini embedding dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.papers: List[Paper] = []
        self.chunks: List[str] = []
        self.chunk_to_paper: List[int] = []  # Maps chunk index to paper index
        
        logger.info("Arxiv Agent initialized successfully")
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search for papers on Arxiv"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in search.results():
                paper = Paper(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    arxiv_id=result.entry_id.split('/')[-1],
                    published_date=result.published.strftime("%Y-%m-%d"),
                    categories=result.categories,
                    pdf_url=result.pdf_url
                )
                papers.append(paper)
                logger.info(f"Found paper: {paper.title}")
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            return []
    
    def extract_paper_content(self, paper: Paper) -> str:
        """Extract full text content from paper (simplified version)"""
        try:
            # Enhanced content extraction with more structured information
            content = f"""
PAPER INFORMATION:
Title: {paper.title}
Authors: {', '.join(paper.authors)}
Arxiv ID: {paper.arxiv_id}
Published Date: {paper.published_date}
Categories: {', '.join(paper.categories)}
PDF URL: {paper.pdf_url}

ABSTRACT:
{paper.abstract}

RESEARCH CONTEXT:
This paper was published in the categories: {', '.join(paper.categories)}
The research was conducted by: {', '.join(paper.authors)}
Publication date: {paper.published_date}
            """.strip()
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {paper.arxiv_id}: {e}")
            return f"Title: {paper.title}\nAbstract: {paper.abstract}"
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """Split text into overlapping chunks with sentence boundary awareness"""
        chunks = []
        
        # Split by sentences first for better semantic coherence
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # Add sentence to current chunk
            test_chunk = current_chunk + sentence + ". "
            
            # If chunk is getting too large, finalize it
            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = sentence + ". "
            else:
                current_chunk = test_chunk
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Fallback to character-based chunking if sentence-based didn't work well
        if not chunks or len(chunks) == 1 and len(text) > chunk_size * 2:
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                chunks.append(chunk)
                start = end - overlap
                if start >= len(text):
                    break
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using Gemini embedding model"""
        try:
            embeddings = []
            for text in texts:
                # Use the correct embedding API
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = result['embedding']
                embeddings.append(embedding)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return np.array([])
    
    def build_knowledge_base(self, papers: List[Paper]) -> None:
        """Build knowledge base from papers"""
        try:
            logger.info(f"Building knowledge base from {len(papers)} papers")
            
            for i, paper in enumerate(papers):
                # Extract content
                content = self.extract_paper_content(paper)
                paper.content = content
                
                # Chunk the content
                chunks = self.chunk_text(content)
                
                # Store chunks and mapping
                for chunk in chunks:
                    self.chunks.append(chunk)
                    self.chunk_to_paper.append(i)
            
            # Get embeddings for all chunks
            if self.chunks:
                embeddings = self.get_embeddings(self.chunks)
                if len(embeddings) > 0:
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(embeddings)
                    self.index.add(embeddings)
                    logger.info(f"Added {len(embeddings)} chunks to knowledge base")
            
            self.papers = papers
            
        except Exception as e:
            logger.error(f"Error building knowledge base: {e}")
    
    def rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Advanced reranking using multiple signals"""
        try:
            # Calculate additional relevance signals
            for result in results:
                # Keyword matching score
                query_words = set(query.lower().split())
                chunk_words = set(result.relevant_chunks[0].lower().split())
                keyword_overlap = len(query_words.intersection(chunk_words)) / len(query_words) if query_words else 0
                
                # Title relevance score
                title_words = set(result.paper.title.lower().split())
                title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
                
                # Abstract relevance score
                abstract_words = set(result.paper.abstract.lower().split())
                abstract_overlap = len(query_words.intersection(abstract_words)) / len(query_words) if query_words else 0
                
                # Combined relevance score
                combined_score = (
                    result.similarity_score * 0.6 +  # Semantic similarity (primary)
                    keyword_overlap * 0.2 +          # Keyword matching
                    title_overlap * 0.15 +           # Title relevance
                    abstract_overlap * 0.05          # Abstract relevance
                )
                
                result.similarity_score = combined_score
                
            # Sort by combined score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results

    def search_knowledge_base(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Enhanced search with semantic similarity and reranking"""
        try:
            logger.info(f"Searching knowledge base for query: '{query}'")
            logger.info(f"Knowledge base contains {len(self.chunks)} chunks from {len(self.papers)} papers")
            
            # Get query embedding
            query_embedding = self.get_embeddings([query])
            if len(query_embedding) == 0:
                logger.error("Failed to get query embedding")
                return []
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search index with more candidates for reranking
            search_k = min(top_k * 2, len(self.chunks))  # Get 2x candidates for reranking
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Create initial results
            initial_results = []
            seen_papers = set()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    paper_idx = self.chunk_to_paper[idx]
                    paper = self.papers[paper_idx]
                    chunk = self.chunks[idx]
                    
                    # Group chunks by paper to avoid redundancy
                    if paper_idx not in seen_papers:
                        result = SearchResult(
                            paper=paper,
                            similarity_score=float(score),
                            relevant_chunks=[chunk]
                        )
                        initial_results.append(result)
                        seen_papers.add(paper_idx)
                    else:
                        # Add chunk to existing paper result
                        for existing_result in initial_results:
                            if existing_result.paper.arxiv_id == paper.arxiv_id:
                                existing_result.relevant_chunks.append(chunk)
                                # Update score if this chunk is more relevant
                                if score > existing_result.similarity_score:
                                    existing_result.similarity_score = float(score)
                                break
            
            # Apply advanced reranking
            reranked_results = self.rerank_results(query, initial_results)
            
            # Return top results
            final_results = reranked_results[:top_k]
            
            for result in final_results:
                logger.info(f"Final result - Paper: {result.paper.title}, Score: {result.similarity_score:.3f}, Chunks: {len(result.relevant_chunks)}")
            
            logger.info(f"Returning {len(final_results)} reranked search results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def suggest_better_search_terms(self, query: str) -> List[str]:
        """Suggest better search terms based on the query"""
        suggestions = {
            "supervised learning": ["supervised learning survey", "supervised machine learning", "classification algorithms", "regression analysis"],
            "machine learning": ["machine learning fundamentals", "machine learning survey", "statistical learning theory"],
            "deep learning": ["deep learning survey", "neural networks", "deep neural networks"],
            "neural networks": ["artificial neural networks", "neural network architectures", "multilayer perceptrons"],
            "classification": ["classification algorithms", "pattern classification", "supervised classification"],
            "regression": ["regression analysis", "linear regression", "nonlinear regression"],
            "clustering": ["clustering algorithms", "unsupervised learning", "cluster analysis"],
            "reinforcement learning": ["reinforcement learning survey", "Q-learning", "policy gradient methods"],
            "natural language processing": ["NLP survey", "text processing", "language models"],
            "computer vision": ["computer vision survey", "image processing", "object detection"]
        }
        
        query_lower = query.lower()
        for key, terms in suggestions.items():
            if key in query_lower:
                return terms
        
        return []

    def generate_hybrid_response(self, query: str) -> str:
        """Generate a comprehensive response using general knowledge when papers are insufficient"""
        try:
            prompt = f"""You are an expert AI assistant with comprehensive knowledge of machine learning, artificial intelligence, and computer science. Provide a detailed, accurate answer to the user's question using your general knowledge.

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a comprehensive, accurate definition or explanation
2. Include key concepts, principles, and examples
3. Structure your answer clearly with main points
4. Use technical accuracy while remaining accessible
5. Include practical applications and examples where relevant
6. Mention different types, categories, or approaches if applicable
7. Provide a complete answer that fully addresses the question

ANSWER:"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating hybrid response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"

    def generate_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate response using Gemini with RAG and hybrid knowledge"""
        try:
            # Check if we have good search results
            filtered_results = [r for r in search_results if r.similarity_score > 0.1] if search_results else []
            
            # If no papers or low relevance, provide hybrid response
            if not search_results or not filtered_results:
                logger.info("No relevant papers found, generating hybrid response with general knowledge")
                general_answer = self.generate_hybrid_response(query)
                
                suggestions = self.suggest_better_search_terms(query)
                suggestion_text = ""
                if suggestions:
                    suggestion_text = f"\n\n---\n\n💡 **For research papers on this topic, try searching for:**\n" + "\n".join([f"• {term}" for term in suggestions[:3]])
                
                return f"{general_answer}{suggestion_text}"
            
            # Prepare context from search results with multiple chunks per paper
            context_parts = []
            for i, result in enumerate(filtered_results[:3]):  # Limit to top 3 most relevant papers
                # Use multiple chunks for richer context
                relevant_chunks = result.relevant_chunks[:2]  # Top 2 chunks per paper
                chunks_text = "\n---\n".join(relevant_chunks)
                
                context_parts.append(f"""
=== PAPER {i+1} ===
Title: {result.paper.title}
Authors: {', '.join(result.paper.authors)}
Arxiv ID: {result.paper.arxiv_id}
Published: {result.paper.published_date}
Categories: {', '.join(result.paper.categories)}
Relevance Score: {result.similarity_score:.3f}

RELEVANT CONTENT:
{chunks_text}
                """.strip())
            
            context = "\n\n".join(context_parts)
            
            # Create enhanced prompt for paper-based response
            prompt = f"""You are an expert AI research assistant. Provide a comprehensive answer that combines information from the provided papers with your general knowledge to give the user a complete response.

CONTEXT FROM RELEVANT PAPERS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Start with a clear, comprehensive answer to the user's question using your general knowledge
2. Then incorporate specific information, findings, and insights from the provided papers
3. When referencing papers, cite them specifically (e.g., "According to [Paper Title] by [Authors]...")
4. Compare and synthesize information across papers when relevant
5. Provide concrete examples, methodologies, or results from the papers
6. Structure your answer clearly: general explanation first, then paper-specific insights
7. Make sure to fully answer the user's question, not just describe what the papers contain

ANSWER:"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def chat(self, query: str, search_query: Optional[str] = None) -> str:
        """Main chat interface"""
        try:
            # If no papers loaded, inform user to search first
            if not self.papers:
                if search_query:
                    logger.info(f"Searching for papers with query: {search_query}")
                    papers = self.search_papers(search_query, max_results=5)
                    if papers:
                        self.build_knowledge_base(papers)
                    else:
                        return "I couldn't find any papers matching your search query. Please try a different search term."
                else:
                    return "No papers are currently loaded in the knowledge base. Please search for papers first using the sidebar, then load them into the knowledge base before asking questions."
            
            # Search knowledge base
            search_results = self.search_knowledge_base(query, top_k=5)
            
            # Generate response
            response = self.generate_response(query, search_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"I encountered an error: {str(e)}"
    
    def get_paper_info(self, arxiv_id: str) -> Optional[Paper]:
        """Get detailed information about a specific paper"""
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            result = next(search.results())
            
            paper = Paper(
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                arxiv_id=result.entry_id.split('/')[-1],
                published_date=result.published.strftime("%Y-%m-%d"),
                categories=result.categories,
                pdf_url=result.pdf_url
            )
            
            return paper
            
        except Exception as e:
            logger.error(f"Error getting paper info for {arxiv_id}: {e}")
            return None 