from dataclasses import dataclass

import numpy as np


@dataclass
class MatchCandidate:
    student_id: str
    full_name: str
    score: float
    top_reference_scores: list
    top_reference_images: list


@dataclass
class MatchResult:
    status: str
    student_id: str
    full_name: str
    score: float
    second_score: float
    margin: float
    candidates: list


class FaceGalleryMatcher:
    def __init__(
        self,
        gallery_path,
        top_k=2,
        min_score=0.55,
        min_margin=0.08,
        max_candidates=3,
    ):
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.max_candidates = int(max_candidates)

        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")

        gallery = np.load(
            gallery_path,
            allow_pickle=False,
        )

        self.embeddings = gallery[
            "image_embeddings"
        ].astype(np.float32)

        self.image_student_ids = gallery[
            "image_student_ids"
        ].astype(str).tolist()

        self.image_filenames = gallery[
            "image_filenames"
        ].astype(str).tolist()

        student_ids = gallery[
            "student_ids"
        ].astype(str).tolist()

        student_names = gallery[
            "student_names"
        ].astype(str).tolist()

        self.student_names = dict(
            zip(student_ids, student_names)
        )

        self.indices_by_student = {}

        for index, student_id in enumerate(
            self.image_student_ids
        ):
            self.indices_by_student.setdefault(
                student_id,
                [],
            ).append(index)

        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

        self.embeddings = (
            self.embeddings
            / np.maximum(norms, 1e-12)
        ).astype(np.float32)

    def normalize_query(
        self,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(-1)

        if query.size != self.embeddings.shape[1]:
            raise ValueError(
                f"Expected query dimension "
                f"{self.embeddings.shape[1]}, "
                f"got {query.size}."
            )

        norm = float(np.linalg.norm(query))

        if norm <= 1e-12:
            raise ValueError(
                "Query embedding has zero norm."
            )

        return query / norm

    def rank_candidates(
        self,
        query_embedding: np.ndarray,
        exclude_image_index=None,
    ) -> list:
        query = self.normalize_query(query_embedding)

        candidates = []

        for student_id, original_indices in (
            self.indices_by_student.items()
        ):
            reference_indices = [
                index
                for index in original_indices
                if index != exclude_image_index
            ]

            if not reference_indices:
                continue

            reference_embeddings = self.embeddings[
                reference_indices
            ]

            similarities = (
                reference_embeddings @ query
            )

            ranked_local_indices = np.argsort(
                similarities
            )[::-1]

            selected_count = min(
                self.top_k,
                len(ranked_local_indices),
            )

            selected_local_indices = (
                ranked_local_indices[:selected_count]
            )

            selected_scores = similarities[
                selected_local_indices
            ]

            selected_global_indices = [
                reference_indices[int(local_index)]
                for local_index in selected_local_indices
            ]

            student_score = float(
                selected_scores.mean()
            )

            candidates.append(
                MatchCandidate(
                    student_id=student_id,
                    full_name=self.student_names.get(
                        student_id,
                        "",
                    ),
                    score=student_score,
                    top_reference_scores=[
                        float(score)
                        for score in selected_scores
                    ],
                    top_reference_images=[
                        self.image_filenames[index]
                        for index in selected_global_indices
                    ],
                )
            )

        candidates.sort(
            key=lambda candidate: candidate.score,
            reverse=True,
        )

        return candidates

    def match(
        self,
        query_embedding: np.ndarray,
        exclude_image_index=None,
    ) -> MatchResult:
        candidates = self.rank_candidates(
            query_embedding=query_embedding,
            exclude_image_index=exclude_image_index,
        )

        if not candidates:
            return MatchResult(
                status="UNKNOWN",
                student_id="",
                full_name="",
                score=0.0,
                second_score=0.0,
                margin=0.0,
                candidates=[],
            )

        best = candidates[0]

        second_score = (
            candidates[1].score
            if len(candidates) > 1
            else -1.0
        )

        margin = best.score - second_score

        if best.score < self.min_score:
            status = "UNKNOWN"

        elif margin < self.min_margin:
            status = "AMBIGUOUS"

        else:
            status = "MATCH"

        return MatchResult(
            status=status,
            student_id=best.student_id,
            full_name=best.full_name,
            score=best.score,
            second_score=second_score,
            margin=margin,
            candidates=candidates[
                :self.max_candidates
            ],
        )

