"""
Vector store for Examina using ChromaDB.
Handles embeddings and semantic search for RAG.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import Config
from models.llm_manager import LLMManager


class VectorStore:
    """Manages vector embeddings and semantic search."""

    def __init__(
        self, persist_dir: Optional[Path] = None, llm_manager: Optional[LLMManager] = None
    ):
        """Initialize vector store.

        Args:
            persist_dir: Directory to persist embeddings
            llm_manager: LLM manager for generating embeddings
        """
        self.persist_dir = persist_dir or Config.CHROMA_PATH
        self.llm = llm_manager or LLMManager()

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False)
        )

    def get_or_create_collection(
        self, course_code: str, collection_type: str = "exercises"
    ) -> chromadb.Collection:
        """Get or create a collection for a course.

        Args:
            course_code: Course code
            collection_type: Type of collection ("exercises", "procedures")

        Returns:
            ChromaDB collection
        """
        collection_name = f"{collection_type}_{course_code}"

        # Get or create collection
        collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"course_code": course_code, "type": collection_type}
        )

        return collection

    def add_exercise(self, course_code: str, exercise_id: str, text: str, metadata: Dict[str, Any]):
        """Add an exercise to the vector store.

        Args:
            course_code: Course code
            exercise_id: Exercise ID
            text: Exercise text
            metadata: Exercise metadata
        """
        collection = self.get_or_create_collection(course_code, "exercises")

        # Generate embedding
        embedding = self.llm.embed(text)
        if not embedding:
            print(f"Warning: Failed to generate embedding for {exercise_id}")
            return

        # Add to collection
        collection.add(
            ids=[exercise_id], embeddings=[embedding], documents=[text], metadatas=[metadata]
        )

    def add_exercises_batch(self, course_code: str, exercises: List[Dict[str, Any]]):
        """Add multiple exercises at once.

        Args:
            course_code: Course code
            exercises: List of exercise dicts with 'id', 'text', and metadata
        """
        if not exercises:
            return

        collection = self.get_or_create_collection(course_code, "exercises")

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for exercise in exercises:
            # Generate embedding
            embedding = self.llm.embed(exercise["text"])
            if not embedding:
                continue

            ids.append(exercise["id"])
            embeddings.append(embedding)
            documents.append(exercise["text"])

            # Prepare metadata (only JSON-serializable values, no None)
            metadata = {
                "topic": exercise.get("topic") or "unknown",
                "knowledge_item_id": exercise.get("knowledge_item_id") or "unknown",
                "difficulty": exercise.get("difficulty") or "unknown",
                "has_images": exercise.get("has_images", False),
                "page_number": exercise.get("page_number", 0),
                "source_pdf": exercise.get("source_pdf") or "unknown",
            }
            metadatas.append(metadata)

        if ids:
            collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def search_similar(
        self,
        course_code: str,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar exercises.

        Args:
            course_code: Course code
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of matching exercises with metadata
        """
        collection = self.get_or_create_collection(course_code, "exercises")

        # Generate query embedding
        query_embedding = self.llm.embed(query)
        if not query_embedding:
            return []

        # Search
        results = collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=filter_metadata
        )

        # Format results
        matches = []
        if results["ids"]:
            for i, ex_id in enumerate(results["ids"][0]):
                matches.append(
                    {
                        "id": ex_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
                    }
                )

        return matches

    def add_knowledge_item(
        self,
        course_code: str,
        knowledge_item_id: str,
        name: str,
        description: str,
        procedure: List[str],
        example_exercises: List[str],
    ):
        """Add a core loop to the vector store.

        Args:
            course_code: Course code
            knowledge_item_id: Core loop ID
            name: Core loop name
            description: Description
            procedure: List of procedure steps
            example_exercises: Exercise IDs that use this core loop
        """
        collection = self.get_or_create_collection(course_code, "procedures")

        # Create document from procedure
        procedure_text = f"{name}\n\n{description}\n\nProcedure:\n" + "\n".join(
            f"{i + 1}. {step}" for i, step in enumerate(procedure)
        )

        # Generate embedding
        embedding = self.llm.embed(procedure_text)
        if not embedding:
            print(f"Warning: Failed to generate embedding for core loop {knowledge_item_id}")
            return

        # Add to collection
        collection.add(
            ids=[knowledge_item_id],
            embeddings=[embedding],
            documents=[procedure_text],
            metadatas=[
                {"name": name, "example_count": len(example_exercises), "course_code": course_code}
            ],
        )

    def search_knowledge_items(
        self, course_code: str, query: str, n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """Search for relevant core loops.

        Args:
            course_code: Course code
            query: Search query
            n_results: Number of results

        Returns:
            List of matching core loops
        """
        collection = self.get_or_create_collection(course_code, "procedures")

        # Generate query embedding
        query_embedding = self.llm.embed(query)
        if not query_embedding:
            return []

        # Search
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

        # Format results
        matches = []
        if results["ids"]:
            for i, loop_id in enumerate(results["ids"][0]):
                matches.append(
                    {
                        "id": loop_id,
                        "procedure_text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i] if "distances" in results else None,
                    }
                )

        return matches

    def get_collection_stats(self, course_code: str) -> Dict[str, Any]:
        """Get statistics about collections for a course.

        Args:
            course_code: Course code

        Returns:
            Dict with stats
        """
        stats = {}

        # Exercise collection
        try:
            ex_collection = self.get_or_create_collection(course_code, "exercises")
            stats["exercises_count"] = ex_collection.count()
        except Exception:
            stats["exercises_count"] = 0

        # Procedure collection
        try:
            proc_collection = self.get_or_create_collection(course_code, "procedures")
            stats["procedures_count"] = proc_collection.count()
        except Exception:
            stats["procedures_count"] = 0

        return stats

    def delete_collection(self, course_code: str, collection_type: str = "exercises"):
        """Delete a collection.

        Args:
            course_code: Course code
            collection_type: Type of collection to delete
        """
        collection_name = f"{collection_type}_{course_code}"
        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
