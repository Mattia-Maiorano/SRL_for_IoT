import nltk

if __name__ == "__main__":
    print("Downloading NLTK corpora: wordnet and omw-1.4...")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    print("Download complete.")
