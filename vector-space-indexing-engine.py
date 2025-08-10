import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union
import string


class VectorSearchEngine:
    """
    A simple vector space search engine implementation.
    
    Uses TF-IDF weighting and cosine similarity for document ranking.
    """
    
    def __init__(self):
        self.documents = {}
        self.index = {}
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Clean and tokenize text for processing.
        
        Args:
            text: Raw text string
            
        Returns:
            List of cleaned tokens
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split into words
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Split and filter out empty strings
        tokens = [word.strip() for word in text.split() if word.strip()]
        
        return tokens
    
    def create_concordance(self, text: str) -> Dict[str, int]:
        """
        Create a word frequency dictionary (concordance) from text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        tokens = self.preprocess_text(text)
        return Counter(tokens)
    
    def calculate_tf_idf(self, term_freq: int, doc_length: int, doc_freq: int) -> float:
        """
        Calculate TF-IDF score for a term.
        
        Args:
            term_freq: Term frequency in document
            doc_length: Total terms in document
            doc_freq: Number of documents containing the term
            
        Returns:
            TF-IDF score
        """
        if doc_freq == 0:
            return 0.0
        
        tf = term_freq / doc_length if doc_length > 0 else 0
        idf = math.log(self.total_documents / doc_freq) if doc_freq > 0 else 0
        
        return tf * idf
    
    def vector_magnitude(self, vector: Dict[str, float]) -> float:
        """
        Calculate the magnitude of a vector.
        
        Args:
            vector: Dictionary representing a vector
            
        Returns:
            Vector magnitude
        """
        if not isinstance(vector, dict):
            raise ValueError("Vector must be a dictionary")
        
        return math.sqrt(sum(value ** 2 for value in vector.values()))
    
    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        if not isinstance(vec1, dict) or not isinstance(vec2, dict):
            raise ValueError("Both arguments must be dictionaries")
        
        # Calculate dot product
        dot_product = 0
        for term in vec1:
            if term in vec2:
                dot_product += vec1[term] * vec2[term]
        
        # Calculate magnitudes
        mag1 = self.vector_magnitude(vec1)
        mag2 = self.vector_magnitude(vec2)
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def add_document(self, doc_id: Union[int, str], text: str) -> None:
        """
        Add a document to the search engine index.
        
        Args:
            doc_id: Unique identifier for the document
            text: Document text content
        """
        if not isinstance(text, str):
            raise ValueError("Document text must be a string")
        
        # Store the original document
        self.documents[doc_id] = text
        
        # Create concordance
        concordance = self.create_concordance(text)
        
        # Update document frequencies
        for word in concordance:
            self.document_frequencies[word] += 1
        
        # Store concordance in index
        self.index[doc_id] = concordance
        self.total_documents = len(self.documents)
    
    def create_tfidf_vector(self, concordance: Dict[str, int]) -> Dict[str, float]:
        """
        Convert a concordance to a TF-IDF weighted vector.
        
        Args:
            concordance: Word frequency dictionary
            
        Returns:
            TF-IDF weighted vector
        """
        doc_length = sum(concordance.values())
        tfidf_vector = {}
        
        for term, freq in concordance.items():
            doc_freq = self.document_frequencies.get(term, 0)
            tfidf_vector[term] = self.calculate_tf_idf(freq, doc_length, doc_freq)
        
        return tfidf_vector
    
    def search(self, query: str, max_results: int = 10) -> List[Tuple[float, Union[int, str], str]]:
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of tuples (score, doc_id, preview_text)
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        if not self.documents:
            return []
        
        # Create query vector
        query_concordance = self.create_concordance(query)
        query_vector = {}
        
        # Use simple term frequency for query (no IDF)
        total_query_terms = sum(query_concordance.values())
        for term, freq in query_concordance.items():
            if term in self.document_frequencies:
                query_vector[term] = freq / total_query_terms
        
        # Score all documents
        results = []
        for doc_id, concordance in self.index.items():
            # Create TF-IDF vector for document
            doc_vector = self.create_tfidf_vector(concordance)
            
            # Calculate similarity
            similarity = self.cosine_similarity(query_vector, doc_vector)
            
            if similarity > 0:
                # Get preview text (first 100 characters)
                preview = self.documents[doc_id][:100]
                if len(self.documents[doc_id]) > 100:
                    preview += "..."
                
                results.append((similarity, doc_id, preview))
        
        # Sort by similarity score (descending)
        results.sort(reverse=True, key=lambda x: x[0])
        
        return results[:max_results]
    
    def get_document_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Dictionary containing various statistics
        """
        if not self.documents:
            return {"total_documents": 0, "unique_terms": 0, "avg_doc_length": 0}
        
        total_terms = sum(len(self.create_concordance(doc)) for doc in self.documents.values())
        avg_length = total_terms / len(self.documents)
        
        return {
            "total_documents": self.total_documents,
            "unique_terms": len(self.document_frequencies),
            "avg_doc_length": round(avg_length, 2)
        }


def main():
    """
    Demonstration of the search engine with sample documents.
    """
    # Create search engine instance
    engine = VectorSearchEngine()
    
    # Sample documents (based on the blog posts from the original)
    sample_docs = {
        0: "At Scale You Will Hit Every Performance Issue. I used to think I knew a bit about performance scalability and how to keep things trucking when you hit large amounts of data. Truth is I know diddly squat on the subject since the most I have ever done is read about how its done.",
        
        1: "Richard Stallman to visit Australia. Im not usually one to promote events and the like unless I feel there is a genuine benefit to be had by attending but this is one stands out. Richard M Stallman the guru of Free Software is coming Down Under to hold a talk.",
        
        2: "MySQL Backups Done Easily. One thing that comes up a lot on sites like Stackoverflow and the like is how to backup MySQL databases. The first answer is usually use mysqldump. This is all fine and good till you start to want to dump multiple databases.",
        
        3: "Why You Shouldn't roll your own CAPTCHA. At a TechEd I attended a few years ago I was watching a presentation about Security presented by Rocky Heckman. In it he was talking about security algorithms.",
        
        4: "The Great Benefit of Test Driven Development Nobody Talks About. The feeling of productivity because you are writing lots of code. Think about that for a moment. Ask any developer who wants to develop why they became a developer.",
        
        5: "Setting up GIT to use a Subversion SVN style workflow. Moving from Subversion SVN to GIT can be a little confusing at first. I think the biggest thing I noticed was that GIT doesnt have a specific workflow you have to pick your own.",
        
        6: "Why CAPTCHA Never Use Numbers 0 1 5 7. Interestingly this sort of question pops up a lot in my referring search term stats. Its because each of the above numbers are easy to confuse with a letter."
    }
    
    # Add documents to the search engine
    print("Building search index...")
    for doc_id, text in sample_docs.items():
        engine.add_document(doc_id, text)
    
    # Display statistics
    stats = engine.get_document_stats()
    print(f"\nSearch engine ready!")
    print(f"Documents indexed: {stats['total_documents']}")
    print(f"Unique terms: {stats['unique_terms']}")
    print(f"Average document length: {stats['avg_doc_length']} terms")
    
    # Interactive search loop
    print("\n" + "="*50)
    print("SEARCH ENGINE - Enter your queries below")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        try:
            query = input("\nEnter search term: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                print("Please enter a search term.")
                continue
            
            # Perform search
            results = engine.search(query, max_results=5)
            
            if not results:
                print(f"No results found for '{query}'")
            else:
                print(f"\nResults for '{query}':")
                print("-" * 40)
                for i, (score, doc_id, preview) in enumerate(results, 1):
                    print(f"{i}. Score: {score:.4f}")
                    print(f"   Doc ID: {doc_id}")
                    print(f"   Preview: {preview}")
                    print()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()