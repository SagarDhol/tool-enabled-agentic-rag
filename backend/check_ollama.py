import ollama
try:
    response = ollama.list()
    print("Local Ollama models:")
    # The response object might be different in newer versions
    for model in getattr(response, 'models', []):
        print(f"- {model.model}")
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print(f"Object type: {type(response)}")
    print(f"Object content: {response}")
