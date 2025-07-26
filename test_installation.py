#!/usr/bin/env python3
"""
Test script to verify Arxiv Agent installation
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        'google.generativeai',
        'arxiv',
        'requests',
        'dotenv',
        'streamlit',
        'pandas',
        'numpy',
        'faiss',
        'bs4',
        'lxml',
        'tiktoken'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_environment():
    """Test environment variables"""
    print("\n🔍 Testing environment variables...")
    
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            print("✅ GOOGLE_API_KEY found")
            return True
        else:
            print("❌ GOOGLE_API_KEY not found")
            print("💡 Set GOOGLE_API_KEY environment variable or add it to .env file")
            return False
            
    except Exception as e:
        print(f"❌ Error testing environment: {e}")
        return False

def test_agent_initialization():
    """Test if the agent can be initialized"""
    print("\n🔍 Testing agent initialization...")
    
    try:
        from arxiv_agent import ArxivAgent
        
        agent = ArxivAgent()
        print("✅ ArxivAgent initialized successfully!")
        
        # Test embedding functionality
        print("🔍 Testing embedding functionality...")
        test_texts = ["This is a test sentence for embeddings."]
        embeddings = agent.get_embeddings(test_texts)
        if len(embeddings) > 0 and embeddings.shape[1] > 0:
            print("✅ Embedding functionality working!")
            return True
        else:
            print("❌ Embedding functionality failed")
            return False
        
    except Exception as e:
        print(f"❌ Failed to initialize ArxivAgent: {e}")
        return False

def test_arxiv_search():
    """Test Arxiv search functionality"""
    print("\n🔍 Testing Arxiv search...")
    
    try:
        import arxiv
        
        search = arxiv.Search(query="machine learning", max_results=1)
        results = list(search.results())
        
        if results:
            print("✅ Arxiv search working!")
            return True
        else:
            print("❌ No search results (might be network issue)")
            return False
            
    except Exception as e:
        print(f"❌ Arxiv search failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Arxiv Agent Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment Variables", test_environment),
        ("Agent Initialization", test_agent_initialization),
        ("Arxiv Search", test_arxiv_search)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Arxiv Agent is ready to use.")
        print("\nNext steps:")
        print("1. Run: streamlit run app.py (for web interface)")
        print("2. Run: python cli.py --interactive (for CLI)")
        print("3. Run: python example.py (for example usage)")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set GOOGLE_API_KEY environment variable")
        print("3. Check internet connection for Arxiv access")

if __name__ == "__main__":
    main() 