#!/usr/bin/env python3
"""
Command-line interface for Arxiv Agent
"""

import argparse
import sys
import os
from typing import List, Optional
from dotenv import load_dotenv

from arxiv_agent import ArxivAgent, Paper

# Load environment variables
load_dotenv()

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    📚 Arxiv Agent CLI                        ║
    ║              AI-Powered Paper Assistant                      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_paper(paper: Paper, show_abstract: bool = True):
    """Print paper information in a formatted way"""
    print(f"\n{'='*80}")
    print(f"📄 {paper.title}")
    print(f"{'='*80}")
    print(f"👥 Authors: {', '.join(paper.authors)}")
    print(f"🆔 Arxiv ID: {paper.arxiv_id}")
    print(f"📅 Published: {paper.published_date}")
    print(f"🏷️  Categories: {', '.join(paper.categories)}")
    if show_abstract:
        print(f"\n📝 Abstract:")
        print(f"{paper.abstract}")
    print(f"🔗 PDF: {paper.pdf_url}")
    print(f"{'='*80}")

def interactive_mode(agent: ArxivAgent):
    """Run interactive mode"""
    print("\n🎯 Interactive Mode - Type 'help' for commands, 'quit' to exit")
    print("="*60)
    
    while True:
        try:
            command = input("\n🤖 Arxiv Agent > ").strip()
            
            if not command:
                continue
                
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if command.lower() == 'help':
                print_help()
                continue
                
            if command.lower() == 'status':
                print_status(agent)
                continue
                
            if command.lower().startswith('search '):
                query = command[7:].strip()
                if query:
                    search_papers_interactive(agent, query)
                else:
                    print("❌ Please provide a search query")
                continue
                
            if command.lower().startswith('load '):
                query = command[5:].strip()
                if query:
                    load_papers_interactive(agent, query)
                else:
                    print("❌ Please provide a search query to load papers")
                continue
                
            if command.lower() == 'clear':
                clear_knowledge_base(agent)
                continue
                
            if command.lower() == 'papers':
                show_loaded_papers(agent)
                continue
                
            # Default: treat as a chat question
            if agent.papers:
                print("\n💭 Thinking...")
                response = agent.chat(command)
                print(f"\n🤖 Assistant: {response}")
            else:
                print("❌ No papers loaded. Use 'search <query>' to find papers first.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_help():
    """Print help information"""
    help_text = """
📚 Available Commands:
    
    search <query>     - Search for papers on Arxiv
    load <query>       - Search and load papers into knowledge base
    papers             - Show currently loaded papers
    clear              - Clear the knowledge base
    status             - Show agent status
    help               - Show this help message
    quit/exit/q        - Exit the application
    
💬 Chat Mode:
    Just type your question to chat with the loaded papers!
    """
    print(help_text)

def print_status(agent: ArxivAgent):
    """Print agent status"""
    print(f"\n📊 Agent Status:")
    print(f"   Papers loaded: {len(agent.papers)}")
    print(f"   Chunks in knowledge base: {len(agent.chunks)}")
    print(f"   Index size: {agent.index.ntotal if hasattr(agent.index, 'ntotal') else 'N/A'}")

def search_papers_interactive(agent: ArxivAgent, query: str):
    """Search papers interactively"""
    print(f"\n🔍 Searching for: '{query}'")
    try:
        papers = agent.search_papers(query, max_results=5)
        if papers:
            print(f"\n✅ Found {len(papers)} papers:")
            for i, paper in enumerate(papers, 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Authors: {', '.join(paper.authors)}")
                print(f"   ID: {paper.arxiv_id}")
        else:
            print("❌ No papers found")
    except Exception as e:
        print(f"❌ Error searching papers: {e}")

def load_papers_interactive(agent: ArxivAgent, query: str):
    """Load papers interactively"""
    print(f"\n📚 Loading papers for: '{query}'")
    try:
        papers = agent.search_papers(query, max_results=5)
        if papers:
            agent.build_knowledge_base(papers)
            print(f"✅ Loaded {len(papers)} papers into knowledge base")
            print("\n📋 Loaded papers:")
            for i, paper in enumerate(papers, 1):
                print(f"   {i}. {paper.title}")
        else:
            print("❌ No papers found to load")
    except Exception as e:
        print(f"❌ Error loading papers: {e}")

def clear_knowledge_base(agent: ArxivAgent):
    """Clear the knowledge base"""
    agent.papers = []
    agent.chunks = []
    agent.chunk_to_paper = []
    agent.index = agent.index.__class__(agent.dimension)
    print("✅ Knowledge base cleared")

def show_loaded_papers(agent: ArxivAgent):
    """Show currently loaded papers"""
    if agent.papers:
        print(f"\n📚 Currently loaded papers ({len(agent.papers)}):")
        for i, paper in enumerate(agent.papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors)}")
            print(f"   ID: {paper.arxiv_id}")
    else:
        print("📚 No papers currently loaded")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Arxiv Agent - AI-powered paper assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --interactive                    # Start interactive mode
  python cli.py --search "machine learning"      # Search for papers
  python cli.py --load "transformer models"      # Load papers and chat
  python cli.py --question "What are the main findings?" --load "deep learning"
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--search', '-s',
        type=str,
        help='Search for papers'
    )
    
    parser.add_argument(
        '--load', '-l',
        type=str,
        help='Load papers into knowledge base'
    )
    
    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Ask a question about loaded papers'
    )
    
    parser.add_argument(
        '--max-results',
        type=int,
        default=5,
        help='Maximum number of search results (default: 5)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Google API key (or set GOOGLE_API_KEY environment variable)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Initialize agent
    try:
        agent = ArxivAgent(api_key=args.api_key)
        print("✅ Arxiv Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize Arxiv Agent: {e}")
        print("💡 Make sure you have set the GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    # Handle different modes
    if args.interactive:
        interactive_mode(agent)
    elif args.search:
        search_papers_interactive(agent, args.search)
    elif args.load:
        load_papers_interactive(agent, args.load)
        if args.question:
            print(f"\n💭 Question: {args.question}")
            response = agent.chat(args.question)
            print(f"\n🤖 Answer: {response}")
    elif args.question:
        if agent.papers:
            print(f"\n💭 Question: {args.question}")
            response = agent.chat(args.question)
            print(f"\n🤖 Answer: {response}")
        else:
            print("❌ No papers loaded. Use --load to load papers first.")
    else:
        # Default to interactive mode
        interactive_mode(agent)

if __name__ == "__main__":
    main() 