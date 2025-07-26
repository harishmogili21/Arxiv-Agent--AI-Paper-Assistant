#!/usr/bin/env python3
"""
Example script demonstrating Arxiv Agent usage
"""

import os
from dotenv import load_dotenv
from arxiv_agent import ArxivAgent

# Load environment variables
load_dotenv()

def main():
    """Example usage of Arxiv Agent"""
    
    print("🚀 Arxiv Agent Example")
    print("=" * 50)
    
    # Initialize the agent
    try:
        agent = ArxivAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("💡 Make sure you have set the GOOGLE_API_KEY environment variable")
        return
    
    # Example 1: Search for papers
    print("\n🔍 Example 1: Searching for papers")
    print("-" * 30)
    
    search_query = "machine learning"
    print(f"Searching for: '{search_query}'")
    
    papers = agent.search_papers(search_query, max_results=3)
    
    if papers:
        print(f"✅ Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors)}")
            print(f"   ID: {paper.arxiv_id}")
    else:
        print("❌ No papers found")
        return
    
    # Example 2: Build knowledge base
    print("\n📚 Example 2: Building knowledge base")
    print("-" * 30)
    
    print("Building knowledge base from papers...")
    agent.build_knowledge_base(papers)
    print(f"✅ Knowledge base built with {len(agent.chunks)} chunks")
    
    # Example 3: Ask questions
    print("\n💬 Example 3: Asking questions")
    print("-" * 30)
    
    questions = [
        "What are the main topics covered in these papers?",
        "What are the key findings or contributions?",
        "How do these papers relate to each other?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🤔 Question {i}: {question}")
        print("-" * 40)
        
        response = agent.chat(question)
        print(f"🤖 Answer: {response}")
    
    # Example 4: Search knowledge base
    print("\n🔍 Example 4: Searching knowledge base")
    print("-" * 30)
    
    search_query = "neural networks"
    print(f"Searching knowledge base for: '{search_query}'")
    
    results = agent.search_knowledge_base(search_query, top_k=2)
    
    if results:
        print(f"✅ Found {len(results)} relevant results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Paper: {result.paper.title}")
            print(f"   Similarity Score: {result.similarity_score:.3f}")
            print(f"   Relevant Content: {result.relevant_chunks[0][:200]}...")
    else:
        print("❌ No relevant results found")
    
    print("\n🎉 Example completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main() 