import cv2
import numpy as np

from smart_attendance_ai.gpu_runtime import create_gpu_session


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(norms, 1e-12)

    return (vectors / norms).astype(np.float32)


class ArcFaceRecognizer:
    def __init__(
        self,
        model_path,
        input_size=(112, 112),
        embedding_size=512,
        batch_size=8,
        normalize=True,
        provider="cuda",
        allow_cpu_fallback=False,
        device_id=0,
    ):
        self.input_width = int(
            input_size[0]
        )

        self.input_height = int(
            input_size[1]
        )

        self.embedding_size = int(
            embedding_size
        )

        self.batch_size = int(
            batch_size
        )

        self.normalize = bool(
            normalize
        )

        self.requested_provider = str(
            provider
        ).lower()

        self.allow_cpu_fallback = bool(
            allow_cpu_fallback
        )

        self.device_id = int(
            device_id
        )

        if self.batch_size < 1:
            raise ValueError(
                "Batch size must be at least 1."
            )

        self.session = create_gpu_session(
            model_path,
            provider=self.requested_provider,
            allow_cpu_fallback=(
                self.allow_cpu_fallback
            ),
            device_id=self.device_id,
        )

        self.input_node = (
            self.session.get_inputs()[0]
        )

        self.output_node = (
            self.session.get_outputs()[0]
        )

        self.input_name = (
            self.input_node.name
        )

        self.output_name = (
            self.output_node.name
        )

        
    @property
    def providers(self):
        return self.session.get_providers()

    def preprocess(self, faces: list) -> np.ndarray:
        if not faces:
            raise ValueError("Face batch is empty.")

        validated_faces = []

        for index, face in enumerate(faces):
            if face is None or face.size == 0:
                raise ValueError(f"Face at index {index} is empty.")

            if face.ndim != 3 or face.shape[2] != 3:
                raise ValueError(
                    f"Face at index {index} has invalid shape: "
                    f"{face.shape}"
                )

            if (
                face.shape[1] != self.input_width
                or face.shape[0] != self.input_height
            ):
                face = cv2.resize(
                    face,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            validated_faces.append(face)

        blob = cv2.dnn.blobFromImages(
            validated_faces,
            scalefactor=1.0 / 127.5,
            size=(self.input_width, self.input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )

        return blob.astype(np.float32)

    def warm_up(self):
        dummy_face = np.zeros(
            (self.input_height, self.input_width, 3),
            dtype=np.uint8,
        )

        blob = self.preprocess([dummy_face])

        self.session.run(
            [self.output_name],
            {self.input_name: blob},
        )

    def embed_faces(self, faces: list) -> np.ndarray:
        if not faces:
            return np.empty(
                (0, self.embedding_size),
                dtype=np.float32,
            )

        embedding_batches = []

        for start in range(0, len(faces), self.batch_size):
            end = min(start + self.batch_size, len(faces))
            face_batch = faces[start:end]

            blob = self.preprocess(face_batch)

            output = self.session.run(
                [self.output_name],
                {self.input_name: blob},
            )[0]

            embeddings = np.asarray(
                output,
                dtype=np.float32,
            ).reshape(len(face_batch), -1)

            if embeddings.shape[1] != self.embedding_size:
                raise RuntimeError(
                    f"Expected embedding size {self.embedding_size}, "
                    f"but model returned {embeddings.shape[1]}."
                )

            if not np.isfinite(embeddings).all():
                raise RuntimeError(
                    f"ArcFace returned invalid values for batch "
                    f"{start}:{end}."
                )

            if self.normalize:
                embeddings = l2_normalize(embeddings)

            embedding_batches.append(embeddings)

        return np.concatenate(
            embedding_batches,
            axis=0,
        ).astype(np.float32)