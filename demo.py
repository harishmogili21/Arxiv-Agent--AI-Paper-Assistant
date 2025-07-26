#!/usr/bin/env python3
"""
Demo script for Arxiv Agent - AI Paper Assistant
This script demonstrates the key features of the Arxiv Agent.
"""

import os
import time
from arxiv_agent import ArxivAgent

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n🔹 Step {step}: {description}")
    print("-" * 40)

def demo_basic_functionality():
    """Demonstrate basic functionality"""
    print_header("ARXIV AGENT DEMO")
    
    # Initialize agent
    print_step(1, "Initializing Arxiv Agent")
    try:
        agent = ArxivAgent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("💡 Make sure you have set the GOOGLE_API_KEY environment variable")
        return None
    
    return agent

def demo_paper_search(agent):
    """Demonstrate paper search functionality"""
    print_step(2, "Searching for Papers")
    
    query = "machine learning survey"
    print(f"🔍 Searching for: '{query}'")
    
    papers = agent.search_papers(query, max_results=3)
    
    if papers:
        print(f"✅ Found {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n   {i}. {paper.title}")
            print(f"      Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
            print(f"      Published: {paper.published_date}")
    else:
        print("❌ No papers found")
    
    return papers

def demo_knowledge_base(agent, papers):
    """Demonstrate knowledge base building"""
    print_step(3, "Building Knowledge Base")
    
    if not papers:
        print("❌ No papers to load")
        return
    
    print(f"📚 Loading {len(papers)} papers into knowledge base...")
    agent.build_knowledge_base(papers)
    print(f"✅ Knowledge base built with {len(agent.chunks)} chunks")

def demo_chat_functionality(agent):
    """Demonstrate chat functionality"""
    print_step(4, "Testing Chat Functionality")
    
    questions = [
        "What is machine learning?",
        "What are the main types of machine learning?",
        "What are the current challenges in machine learning?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("🤔 Thinking...")
        
        response = agent.chat(question)
        
        # Truncate long responses for demo
        if len(response) > 300:
            response = response[:300] + "..."
        
        print(f"🤖 Response: {response}")
        
        # Small delay for demo effect
        time.sleep(1)

def demo_hybrid_response(agent):
    """Demonstrate hybrid response system"""
    print_step(5, "Testing Hybrid Response System")
    
    # Clear knowledge base to test general knowledge
    agent.papers = []
    agent.chunks = []
    agent.chunk_to_paper = []
    
    question = "What is supervised learning?"
    print(f"\n❓ Question (no papers loaded): {question}")
    print("🤔 Using general knowledge...")
    
    response = agent.chat(question)
    
    # Truncate for demo
    if len(response) > 400:
        response = response[:400] + "..."
    
    print(f"🤖 Response: {response}")

def main():
    """Main demo function"""
    print("🚀 Welcome to the Arxiv Agent Demo!")
    print("This demo will showcase the key features of the AI Paper Assistant.")
    
    # Initialize agent
    agent = demo_basic_functionality()
    if not agent:
        return
    
    # Search for papers
    papers = demo_paper_search(agent)
    
    # Build knowledge base
    demo_knowledge_base(agent, papers)
    
    # Test chat functionality
    demo_chat_functionality(agent)
    
    # Test hybrid response
    demo_hybrid_response(agent)
    
    print_header("DEMO COMPLETE")
    print("🎉 Demo completed successfully!")
    print("\n💡 To try the full application:")
    print("   • Web UI: streamlit run app.py")
    print("   • CLI: python cli.py --interactive")
    print("\n📚 Visit the GitHub repository for more information:")
    print("   https://github.com/harishmogili21/Arxiv-Agent--AI-Paper-Assistant")

if __name__ == "__main__":
    main()