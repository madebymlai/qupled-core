#!/usr/bin/env python3
"""
Phase 9.5: Detailed Multi-Course Analysis

Provides detailed examples and analysis for each course's exercise types,
proof detection, and theory categorization.
"""

import sqlite3
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


class DetailedAnalyzer:
    """Detailed analysis of Phase 9 features."""

    def __init__(self):
        self.db_path = Path(__file__).parent / "data" / "examina.db"
        self.courses = {
            "B006802": "ADE - Computer Architecture",
            "B006807": "AL - Linear Algebra",
            "B018757": "PC - Concurrent Programming"
        }

    def connect_db(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def analyze_ade_exercises(self):
        """Detailed analysis of ADE exercises."""
        console.print("\n[bold cyan]ADE (Computer Architecture) - Detailed Analysis[/bold cyan]\n")

        conn = self.connect_db()

        # Check for FSM design exercises (procedural)
        console.print("[yellow]1. FSM Design Exercises (Procedural):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty
            FROM exercises
            WHERE course_code = 'B006802'
            AND (text LIKE '%automa%' OR text LIKE '%Moore%' OR text LIKE '%Mealy%')
            LIMIT 3
        """)

        for i, row in enumerate(cursor, 1):
            console.print(f"Example {i}:")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")
            console.print(f"Text snippet: {row['text'][:250]}...")
            self._classify_exercise(row['text'])
            console.print()

        # Check for performance analysis (theory-heavy)
        console.print("[yellow]2. Performance Analysis Exercises (Theory/Hybrid):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty
            FROM exercises
            WHERE course_code = 'B006802'
            AND (text LIKE '%Amdahl%' OR text LIKE '%prestazioni%' OR text LIKE '%performance%'
                 OR text LIKE '%CPI%' OR text LIKE '%throughput%')
            LIMIT 3
        """)

        for i, row in enumerate(cursor, 1):
            console.print(f"Example {i}:")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")
            console.print(f"Text snippet: {row['text'][:250]}...")
            self._classify_exercise(row['text'])
            console.print()

        # Check for Boolean algebra (potential proofs)
        console.print("[yellow]3. Boolean Algebra Exercises (Potential Proofs):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty
            FROM exercises
            WHERE course_code = 'B006802'
            AND (text LIKE '%booleana%' OR text LIKE '%boolean%'
                 OR text LIKE '%dimostrazione%' OR text LIKE '%dimostra%')
            LIMIT 3
        """)

        found = False
        for i, row in enumerate(cursor, 1):
            found = True
            console.print(f"Example {i}:")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")
            console.print(f"Text snippet: {row['text'][:250]}...")
            self._classify_exercise(row['text'])
            console.print()

        if not found:
            console.print("[dim]No Boolean algebra proof exercises found in ADE[/dim]\n")

        conn.close()

    def analyze_al_exercises(self):
        """Detailed analysis of AL (Linear Algebra) exercises."""
        console.print("\n[bold cyan]AL (Linear Algebra) - Detailed Analysis[/bold cyan]\n")

        conn = self.connect_db()

        # Proof exercises
        console.print("[yellow]1. Proof Exercises (Dimostrazioni):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty, topic_id, knowledge_item_id
            FROM exercises
            WHERE course_code = 'B006807'
            AND (text LIKE '%dimostra%' OR text LIKE '%dimostrazione%' OR text LIKE '%proof%')
            LIMIT 4
        """)

        for i, row in enumerate(cursor, 1):
            console.print(f"[bold]Proof Example {i}:[/bold]")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")

            # Extract proof-specific text
            text = row['text']
            if 'dimostra' in text.lower():
                # Find the sentence with "dimostra"
                sentences = text.split('.')
                proof_sentence = next((s for s in sentences if 'dimostra' in s.lower()), '')
                console.print(f"Proof request: {proof_sentence.strip()[:150]}...")

            self._classify_exercise(row['text'])
            self._analyze_proof_characteristics(row['text'])
            console.print()

        # Theory questions (definitions, theorems)
        console.print("[yellow]2. Theory Questions (Definitions, Theorems):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty
            FROM exercises
            WHERE course_code = 'B006807'
            AND (text LIKE '%definizione%' OR text LIKE '%deﬁnizione%'
                 OR text LIKE '%teorema%' OR text LIKE '%enunciare%')
            LIMIT 3
        """)

        for i, row in enumerate(cursor, 1):
            console.print(f"[bold]Theory Example {i}:[/bold]")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")

            # Extract theory question
            text = row['text']
            if 'definizione' in text.lower() or 'deﬁnizione' in text.lower():
                sentences = text.split('.')
                theory_sentence = next((s for s in sentences if 'defin' in s.lower()), '')
                console.print(f"Theory question: {theory_sentence.strip()[:150]}...")

            self._classify_exercise(row['text'])
            console.print()

        # Computational exercises (procedural)
        console.print("[yellow]3. Computational Exercises (Procedural):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty
            FROM exercises
            WHERE course_code = 'B006807'
            AND (text LIKE '%calcola%' OR text LIKE '%trova%' OR text LIKE '%determina%')
            AND text NOT LIKE '%dimostra%'
            LIMIT 3
        """)

        for i, row in enumerate(cursor, 1):
            console.print(f"[bold]Procedural Example {i}:[/bold]")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")
            console.print(f"Text snippet: {row['text'][:200]}...")
            self._classify_exercise(row['text'])
            console.print()

        conn.close()

    def analyze_pc_exercises(self):
        """Detailed analysis of PC (Concurrent Programming) exercises."""
        console.print("\n[bold cyan]PC (Concurrent Programming) - Detailed Analysis[/bold cyan]\n")

        conn = self.connect_db()

        # Proof exercises (safety/liveness properties)
        console.print("[yellow]1. Proof Exercises (Safety/Liveness Properties):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty
            FROM exercises
            WHERE course_code = 'B018757'
            AND (text LIKE '%dimostra%' OR text LIKE '%dimostrazione%'
                 OR text LIKE '%induzione%' OR text LIKE '%proprietà%')
            LIMIT 4
        """)

        for i, row in enumerate(cursor, 1):
            console.print(f"[bold]Proof Example {i}:[/bold]")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")

            # Look for property verification
            text = row['text']
            has_ltl = 'LTL' in text or 'ltl' in text
            has_safety = 'safety' in text.lower() or 'sicurezza' in text.lower()
            has_liveness = 'liveness' in text.lower() or 'vivacità' in text.lower()

            console.print(f"Contains LTL: {has_ltl}")
            console.print(f"Safety property: {has_safety}")
            console.print(f"Liveness property: {has_liveness}")

            self._classify_exercise(row['text'])
            console.print()

        # Theory questions (synchronization concepts)
        console.print("[yellow]2. Theory Questions (Synchronization Concepts):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty
            FROM exercises
            WHERE course_code = 'B018757'
            AND (text LIKE '%definire%' OR text LIKE '%spiegare%'
                 OR text LIKE '%cos''è%' OR text LIKE '%descrivere%')
            LIMIT 3
        """)

        found = False
        for i, row in enumerate(cursor, 1):
            found = True
            console.print(f"[bold]Theory Example {i}:[/bold]")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Text snippet: {row['text'][:250]}...")
            self._classify_exercise(row['text'])
            console.print()

        if not found:
            console.print("[dim]No explicit theory questions found (may be embedded in exercises)[/dim]\n")

        # Practical exercises (monitor design, synchronization)
        console.print("[yellow]3. Practical Exercises (Procedural):[/yellow]\n")
        cursor = conn.execute("""
            SELECT id, text, difficulty, tags
            FROM exercises
            WHERE course_code = 'B018757'
            AND (text LIKE '%monitor%' OR text LIKE '%semaforo%'
                 OR text LIKE '%progetta%' OR text LIKE '%implementa%')
            AND text NOT LIKE '%dimostra%'
            LIMIT 3
        """)

        for i, row in enumerate(cursor, 1):
            console.print(f"[bold]Procedural Example {i}:[/bold]")
            console.print(f"ID: {row['id'][:40]}")
            console.print(f"Difficulty: {row['difficulty']}")
            console.print(f"Tags: {row['tags']}")
            console.print(f"Text snippet: {row['text'][:200]}...")
            self._classify_exercise(row['text'])
            console.print()

        conn.close()

    def _classify_exercise(self, text: str):
        """Classify exercise and show reasoning."""
        text_lower = text.lower()

        # Check for proof keywords
        proof_keywords = ['dimostra', 'dimostrazione', 'proof', 'induzione', 'per assurdo']
        proof_found = [kw for kw in proof_keywords if kw in text_lower]

        # Check for theory keywords
        theory_keywords = ['definizione', 'deﬁnizione', 'teorema', 'enunciare', 'spiegare', 'descrivere']
        theory_found = [kw for kw in theory_keywords if kw in text_lower]

        # Check for procedural keywords
        proc_keywords = ['calcola', 'trova', 'determina', 'progetta', 'implementa', 'costruisci']
        proc_found = [kw for kw in proc_keywords if kw in text_lower]

        console.print("[dim]Classification:[/dim]")
        if proof_found:
            console.print(f"  [magenta]Type: PROOF[/magenta] (keywords: {', '.join(proof_found)})")
        elif theory_found and len(theory_found) >= 2:
            console.print(f"  [yellow]Type: THEORY[/yellow] (keywords: {', '.join(theory_found)})")
        elif theory_found and proc_found:
            console.print(f"  [blue]Type: HYBRID[/blue] (theory: {', '.join(theory_found)}, procedural: {', '.join(proc_found[:2])})")
        elif proc_found:
            console.print(f"  [green]Type: PROCEDURAL[/green] (keywords: {', '.join(proc_found[:3])})")
        else:
            console.print(f"  [dim]Type: UNKNOWN (no clear indicators)[/dim]")

    def _analyze_proof_characteristics(self, text: str):
        """Analyze characteristics of proof exercises."""
        text_lower = text.lower()

        characteristics = []

        if 'induzione' in text_lower or 'induction' in text_lower:
            characteristics.append("Proof by induction")
        if 'assurdo' in text_lower or 'contradiction' in text_lower:
            characteristics.append("Proof by contradiction")
        if 'diretta' in text_lower or 'direct' in text_lower:
            characteristics.append("Direct proof")

        # Check for mathematical structures
        if 'matrice' in text_lower or 'matrix' in text_lower:
            characteristics.append("Matrix properties")
        if 'spazio vettoriale' in text_lower or 'vector space' in text_lower:
            characteristics.append("Vector space properties")
        if 'applicazione lineare' in text_lower or 'linear map' in text_lower:
            characteristics.append("Linear transformation properties")

        if characteristics:
            console.print(f"[dim]Proof characteristics: {', '.join(characteristics)}[/dim]")

    def comparative_summary(self):
        """Generate comparative summary across courses."""
        console.print("\n[bold cyan]Comparative Summary: Exercise Types Across Courses[/bold cyan]\n")

        conn = self.connect_db()

        # Count by type for each course
        summary_data = {}

        for code, name in self.courses.items():
            cursor = conn.execute("""
                SELECT COUNT(*) as total FROM exercises WHERE course_code = ?
            """, (code,))
            total = cursor.fetchone()['total']

            # Proof count
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM exercises
                WHERE course_code = ?
                AND (text LIKE '%dimostra%' OR text LIKE '%dimostrazione%' OR text LIKE '%proof%')
            """, (code,))
            proof = cursor.fetchone()['count']

            # Theory count (definitions, theorems)
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM exercises
                WHERE course_code = ?
                AND (text LIKE '%definizione%' OR text LIKE '%deﬁnizione%'
                     OR text LIKE '%teorema%' OR text LIKE '%enunciare%')
                AND text NOT LIKE '%dimostra%'
            """, (code,))
            theory = cursor.fetchone()['count']

            # Procedural (computational)
            cursor = conn.execute("""
                SELECT COUNT(*) as count FROM exercises
                WHERE course_code = ?
                AND (text LIKE '%calcola%' OR text LIKE '%trova%' OR text LIKE '%determina%'
                     OR text LIKE '%progetta%' OR text LIKE '%implementa%')
                AND text NOT LIKE '%dimostra%'
            """, (code,))
            procedural = cursor.fetchone()['count']

            summary_data[name] = {
                'total': total,
                'proof': proof,
                'theory': theory,
                'procedural': procedural
            }

        # Display summary
        for name, data in summary_data.items():
            console.print(f"[bold]{name}[/bold]")
            console.print(f"  Total exercises: {data['total']}")
            console.print(f"  Proof: {data['proof']} ({data['proof']/data['total']*100:.1f}%)")
            console.print(f"  Theory: {data['theory']} ({data['theory']/data['total']*100:.1f}%)")
            console.print(f"  Procedural: {data['procedural']} ({data['procedural']/data['total']*100:.1f}%)")

            # Characterization
            proof_pct = data['proof'] / data['total']
            if proof_pct > 0.4:
                console.print(f"  [yellow]Characterization: Proof-heavy course[/yellow]")
            elif proof_pct > 0.15:
                console.print(f"  [blue]Characterization: Balanced (theory + practice)[/blue]")
            else:
                console.print(f"  [green]Characterization: Procedurally-focused course[/green]")
            console.print()

        conn.close()

    def run(self):
        """Run detailed analysis."""
        console.print("[bold cyan]Phase 9.5: Detailed Multi-Course Analysis[/bold cyan]")
        console.print("[dim]Comprehensive examination of exercise types across courses[/dim]\n")

        self.analyze_ade_exercises()
        self.analyze_al_exercises()
        self.analyze_pc_exercises()
        self.comparative_summary()

        console.print("\n[bold green]✓ Detailed analysis complete![/bold green]\n")


if __name__ == '__main__':
    analyzer = DetailedAnalyzer()
    analyzer.run()
