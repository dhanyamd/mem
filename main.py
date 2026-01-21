"""
Radix-Titan Neural Memory Engine - Main Entry Point

This is the main entry point for the neural memory system.
Use chatbot.py for interactive chat or run individual components for testing.
"""

def main():
    print("🤖 Radix-Titan Neural Memory Engine")
    print("===================================")
    print()
    print("Available commands:")
    print("• python chatbot.py [user_id]    - Start interactive chat")
    print("• python test_neural_memory.py   - Run component tests")
    print("• python -m neural_memory        - Import test")
    print()
    print("Make sure Redis and Qdrant are running!")
    print("See README_NEURAL_MEMORY.md for setup instructions.")


if __name__ == "__main__":
    main()
