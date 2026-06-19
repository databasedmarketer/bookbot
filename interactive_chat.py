#!/usr/bin/env python3
"""
Interactive chat interface for the Book RAG system
"""
from rag_pipeline import BookRAGPipeline

def main():
    print("📚 BookBot - Interactive Mode")
    print("=" * 50)
    
    # Initialize the pipeline
    print("🚀 Initializing system...")
    pipeline = BookRAGPipeline()
    if not pipeline.initialize():
        print("❌ Failed to initialize system")
        return
    
    print("✅ System ready! Ask questions about books.")
    print("Type 'quit' to exit, 'help' for examples\n")
    
    while True:
        try:
            # Get user input
            question = input("💬 You: ").strip()

            # 1. Empty check first
            if not question:
                continue

            # 2. Commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if question.lower() == 'help':
                print("\n📝 Example queries:")
                print("• What books are about marketing?")
                print("• Find books on business communication")
                print("• What are the main themes in the classical marketing book?")
                print("• Recommend books on leadership\n")
                continue

            # 3. Word limit — pipeline also enforces this, but we catch it
            #    early here to give a faster response in the CLI
            MAX_PROMPT_WORDS = 50
            if len(question.split()) > MAX_PROMPT_WORDS:
                print(f"⚠️ Input too long ({len(question.split())} words). Please keep questions under {MAX_PROMPT_WORDS} words.")
                continue

            # 4. Process the query
            print("🔍 Searching...")
            result = pipeline.query(question)

            # Display results
            print(f"\n🤖 Assistant:")
            print(result['answer'])
            print(f"\n⏱️ Processing time: {result['processing_time']:.2f}s")
            print(f"📄 Retrieved {len(result['retrieved_documents'])} documents")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
