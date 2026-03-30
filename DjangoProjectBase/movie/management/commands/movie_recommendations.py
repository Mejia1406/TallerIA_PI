from __future__ import annotations

from django.core.management.base import BaseCommand

from movie.recommendations import recommend_movie_for_prompt


class Command(BaseCommand):
    help = "Recommend the most similar movie to a prompt (uses embeddings stored in the DB)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--prompt",
            type=str,
            required=True,
            help="Text prompt to search (e.g., 'película de la segunda guerra mundial').",
        )

    def handle(self, *args, **options):
        prompt = options["prompt"]
        best_movie, similarity = recommend_movie_for_prompt(prompt)

        if best_movie is None or similarity is None:
            self.stdout.write(self.style.WARNING("No recommendation could be computed."))
            return

        self.stdout.write(
            self.style.SUCCESS(
                f"La película más similar al prompt es: {best_movie.title} con similitud {similarity:.4f}"
            )
        )
