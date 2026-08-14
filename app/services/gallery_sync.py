import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Student, FaceEmbedding


async def build_gallery_from_db(
    session: AsyncSession,
    ai,
    activate_after_build: bool = False,
):
    query = (
        select(
            Student.id,
            Student.first_name,
            Student.last_name,
            FaceEmbedding.embedding,
        )
        .join(FaceEmbedding, FaceEmbedding.student_id == Student.id)
        .where(Student.is_active.is_(True))
    )

    result = await session.execute(query)

    records = [
        {
            "student_id": str(row.id),
            "full_name": f"{row.first_name} {row.last_name}".strip(),
            "embedding": row.embedding,
        }
        for row in result.all()
    ]

    return await asyncio.to_thread(
        ai.rebuild_gallery_from_records,
        records=records,
        activate_after_build=activate_after_build,
    )