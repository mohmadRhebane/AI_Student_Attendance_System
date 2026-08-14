from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_ROOT = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_ROOT),
    )


from smart_attendance_ai.ai_contracts import (
    AISettings,
)
from smart_attendance_ai.ai_facade import (
    SmartAttendanceAI,
)
from smart_attendance_ai.gallery_version_service import (
    GalleryVersionService,
)


def main() -> None:
    generated_root = (
        PROJECT_ROOT
        / "data"
        / "generated"
    )

    generated_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read the current real Gallery only as
    # test input. We will never modify its pointer.
    source_service = (
        GalleryVersionService(
            root=PROJECT_ROOT
        )
    )

    source_gallery = (
        source_service
        .get_active_gallery_path(
            fallback_path=(
                PROJECT_ROOT
                / "data"
                / "generated"
                / "face_gallery_calibrated.npz"
            )
        )
    )

    print(
        "SOURCE GALLERY:",
        source_gallery,
    )

    with np.load(
        source_gallery,
        allow_pickle=False,
    ) as source:
        source_embeddings = (
            source[
                "image_embeddings"
            ]
            .astype(np.float32)
            .copy()
        )

        source_ids = (
            source[
                "image_student_ids"
            ]
            .astype(str)
            .tolist()
        )

        source_names = (
            source[
                "image_full_names"
            ]
            .astype(str)
            .tolist()
        )

        source_filenames = (
            source[
                "image_filenames"
            ]
            .astype(str)
            .tolist()
        )

        source_student_ids = (
            source[
                "student_ids"
            ]
            .astype(str)
            .tolist()
        )

    records = []

    for index, (
        student_id,
        full_name,
        embedding,
        filename,
    ) in enumerate(
        zip(
            source_ids,
            source_names,
            source_embeddings,
            source_filenames,
        ),
        start=1,
    ):
        records.append(
            {
                "student_id": student_id,
                "full_name": full_name,
                "embedding": (
                    embedding
                ),
                "record_id": index,
                "image_filename": (
                    filename
                ),
                "source": (
                    "bootstrap_rebuild_test"
                ),
            }
        )

    with TemporaryDirectory(
        prefix=(
            "gallery_bootstrap_"
            "rebuild_test_"
        ),
        dir=str(
            generated_root
        ),
    ) as temporary:
        temporary_root = Path(
            temporary
        )

        settings = AISettings(
            project_root=(
                PROJECT_ROOT
            ),
            legacy_gallery_path=(
                temporary_root
                / "missing_legacy.npz"
            ),
            active_gallery_pointer_path=(
                temporary_root
                / "active_gallery.json"
            ),
            gallery_versions_root=(
                temporary_root
                / "versions"
            ),
            enrollment_staging_root=(
                temporary_root
                / "staging"
            ),
        )

        ai = SmartAttendanceAI(
            settings=settings
        )

        try:
            # ---------------------------------
            # 1. Clean deployment health
            # ---------------------------------
            health = ai.get_health()

            assert (
                health[
                    "gallery_ready"
                ]
                is False
            )

            assert (
                health[
                    "active_gallery"
                ]
                is None
            )

            print(
                "CLEAN HEALTH PASSED"
            )

            # ---------------------------------
            # 2. Empty bootstrap
            # ---------------------------------
            bootstrap = (
                ai.bootstrap_empty_gallery(
                    version_id=(
                        "TEST_BOOTSTRAP_EMPTY_V001"
                    ),
                    activate_after_build=True,
                )
            )

            assert (
                bootstrap.operation
                == "BOOTSTRAP_EMPTY"
            )

            assert (
                bootstrap.student_count
                == 0
            )

            assert (
                bootstrap.embedding_count
                == 0
            )

            assert (
                bootstrap.activated
                is True
            )

            with np.load(
                bootstrap.gallery_path,
                allow_pickle=False,
            ) as gallery:
                embeddings = (
                    gallery[
                        "image_embeddings"
                    ]
                )

                templates = (
                    gallery[
                        "student_templates"
                    ]
                )

                student_ids = (
                    gallery[
                        "student_ids"
                    ]
                )

                dimension = int(
                    np.asarray(
                        gallery[
                            "embedding_dimension"
                        ]
                    ).item()
                )

                assert (
                    embeddings.shape
                    == (
                        0,
                        dimension,
                    )
                )

                assert (
                    templates.shape
                    == (
                        0,
                        dimension,
                    )
                )

                assert len(
                    student_ids
                ) == 0

            bootstrap_active = (
                ai.get_active_gallery_path()
            )

            assert (
                bootstrap_active.resolve()
                == bootstrap
                .gallery_path
                .resolve()
            )

            print(
                "EMPTY BOOTSTRAP PASSED"
            )

            # ---------------------------------
            # 3. Full DB-style rebuild
            # ---------------------------------
            rebuild = (
                ai.rebuild_gallery_from_records(
                    records=records,
                    version_id=(
                        "TEST_REBUILD_V001"
                    ),
                    activate_after_build=False,
                )
            )

            assert (
                rebuild.operation
                == "REBUILD_FROM_RECORDS"
            )

            assert (
                rebuild.embedding_count
                == len(
                    records
                )
            )

            assert (
                rebuild.student_count
                == len(
                    set(
                        source_ids
                    )
                )
            )

            # Rebuild must NOT replace active
            # Gallery unless explicitly activated.
            active_after_build = (
                ai.get_active_gallery_path()
            )

            assert (
                active_after_build
                .resolve()
                == bootstrap
                .gallery_path
                .resolve()
            )

            with np.load(
                rebuild.gallery_path,
                allow_pickle=False,
            ) as rebuilt_gallery:
                rebuilt_ids = (
                    rebuilt_gallery[
                        "student_ids"
                    ]
                    .astype(str)
                    .tolist()
                )

                rebuilt_embeddings = (
                    rebuilt_gallery[
                        "image_embeddings"
                    ]
                )

                assert set(
                    rebuilt_ids
                ) == set(
                    source_student_ids
                )

                assert (
                    len(
                        rebuilt_embeddings
                    )
                    == len(
                        source_embeddings
                    )
                )

            print(
                "REBUILD WITHOUT "
                "ACTIVATION PASSED"
            )

            # ---------------------------------
            # 4. Explicit activation
            # ---------------------------------
            ai.activate_gallery_version(
                rebuild.version_id
            )

            final_active = (
                ai.get_active_gallery_path()
            )

            assert (
                final_active.resolve()
                == rebuild
                .gallery_path
                .resolve()
            )

            print(
                "REBUILD ACTIVATION PASSED"
            )

            # ---------------------------------
            # 5. Bootstrap must refuse overwrite
            # ---------------------------------
            refused = False

            try:
                ai.bootstrap_empty_gallery(
                    version_id=(
                        "TEST_ILLEGAL_BOOTSTRAP"
                    )
                )

            except Exception:
                refused = True

            assert refused

            print(
                "BOOTSTRAP SAFETY PASSED"
            )

            print()
            print(
                "GALLERY BOOTSTRAP / "
                "REBUILD TEST PASSED"
            )

        finally:
            ai.shutdown()


if __name__ == "__main__":
    main()