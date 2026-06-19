from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import os
from pathlib import Path

from app.database import get_async_db
from app.core.security import get_current_user
from app.core.config import settings
from app.services.image_storage_service import get_image_storage_service
from app.services.s3_service import get_s3_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _local_image_url(storage_id: str) -> str:
    """외부 볼륨에 저장된 이미지를 StaticFiles로 서빙하는 URL."""
    base = settings.LOCAL_STORAGE_BASE_URL.rstrip("/")
    return f"{base}/{storage_id}.png"


def _serialize_generated(img) -> Dict[str, Any]:
    """generated_images 행 → 갤러리 응답(JSON) 직렬화."""
    return {
        "id": img.id,
        "storage_id": img.storage_id,
        "s3_url": _local_image_url(img.storage_id),  # 외부 볼륨(StaticFiles) URL
        "team_id": img.team_id,
        "user_id": img.user_id,
        "prompt": img.prompt,
        "negative_prompt": img.negative_prompt,
        "width": img.width,
        "height": img.height,
        "seed": img.seed,
        "workflow_name": img.workflow_name,
        "model_name": img.model_name,
        "extra_metadata": img.extra_metadata or {},
        "file_size": img.file_size,
        "mime_type": img.mime_type,
        "created_at": img.created_at.isoformat() if img.created_at else None,
        "updated_at": img.updated_at.isoformat() if getattr(img, "updated_at", None) else None,
    }


@router.get("/images", response_model=Dict[str, Any])
async def get_gallery_images(
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(12, ge=1, le=100, description="페이지당 항목 수"),
    team_id: Optional[int] = Query(None, description="팀 ID 필터"),
    db: AsyncSession = Depends(get_async_db),
    current_user: Dict = Depends(get_current_user)
):
    """
    갤러리 이미지 목록 조회 (페이지네이션)

    IMAGE_STORAGE_TYPE=local 이면 외부 볼륨에 저장된 generated_images 를 조회한다.
    그 외(S3)에서는 기존 IMAGE_STORAGE(S3 presigned) 경로를 사용한다.
    """
    try:
        # ── 로컬(외부 볼륨) 모드 ────────────────────────────────
        if settings.IMAGE_STORAGE_TYPE == "local":
            from sqlalchemy import select, func
            from app.models.generated_image import GeneratedImage

            user_id = current_user.get("sub")
            offset = (page - 1) * page_size

            # team_id 가 명시되면 팀 기준, 아니면 본인 생성물 기준
            cond = (
                GeneratedImage.team_id == team_id
                if team_id is not None
                else GeneratedImage.user_id == user_id
            )

            rows = (
                await db.execute(
                    select(GeneratedImage)
                    .where(cond)
                    .order_by(GeneratedImage.created_at.desc())
                    .limit(page_size)
                    .offset(offset)
                )
            ).scalars().all()

            total_count = (
                await db.execute(
                    select(func.count()).select_from(GeneratedImage).where(cond)
                )
            ).scalar() or 0

            return {
                "images": [_serialize_generated(r) for r in rows],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "total_pages": (total_count + page_size - 1) // page_size,
                },
            }

        # ── S3 레거시 모드 ─────────────────────────────────────
        teams = current_user.get("teams", [])
        if not teams:
            return {
                "images": [],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_count": 0,
                    "total_pages": 0
                }
            }

        target_team_id = team_id if team_id else 1

        image_storage_service = get_image_storage_service()
        limit = page_size
        offset = (page - 1) * page_size

        images = await image_storage_service.get_images_by_group(
            group_id=target_team_id,
            db=db,
            limit=limit,
            offset=offset
        )

        from sqlalchemy import select, func
        from app.models.image_storage import ImageStorage

        count_result = await db.execute(
            select(func.count()).select_from(ImageStorage).where(ImageStorage.group_id == target_team_id)
        )
        total_count = count_result.scalar()

        return {
            "images": images,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": (total_count + page_size - 1) // page_size
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"갤러리 이미지 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 목록 조회 실패: {str(e)}")


@router.delete("/images/{storage_id}")
async def delete_gallery_image(
    storage_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: Dict = Depends(get_current_user)
):
    """갤러리 이미지 삭제 (local: 외부 볼륨 파일 + generated_images 행)."""
    try:
        from sqlalchemy import select, delete

        # ── 로컬(외부 볼륨) 모드 ────────────────────────────────
        if settings.IMAGE_STORAGE_TYPE == "local":
            from app.models.generated_image import GeneratedImage

            result = await db.execute(
                select(GeneratedImage).where(GeneratedImage.storage_id == storage_id)
            )
            image = result.scalar_one_or_none()
            if not image:
                raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")

            # 외부 볼륨의 실제 파일 삭제 (저장된 경로 우선, 없으면 표준 경로)
            removed = False
            candidates = []
            if image.s3_url:
                candidates.append(image.s3_url)
            candidates.append(str(Path(settings.LOCAL_STORAGE_PATH) / f"{storage_id}.png"))
            for p in candidates:
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                        removed = True
                        break
                except Exception as fe:
                    logger.warning(f"로컬 이미지 파일 삭제 실패(계속 진행) {p}: {fe}")

            await db.execute(
                delete(GeneratedImage).where(GeneratedImage.storage_id == storage_id)
            )
            await db.commit()
            return {"message": "이미지가 삭제되었습니다.", "file_removed": removed}

        # ── S3 레거시 모드 ─────────────────────────────────────
        from app.models.image_storage import ImageStorage

        result = await db.execute(
            select(ImageStorage).where(ImageStorage.storage_id == storage_id)
        )
        image = result.scalar_one_or_none()

        if not image:
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")

        s3_service = get_s3_service()
        if image.s3_url:
            url_parts = image.s3_url.split('/', 3)
            if len(url_parts) > 3:
                s3_key = url_parts[3]
            else:
                s3_key = f"generate_image/team_{image.group_id}/{storage_id}.png"
        else:
            s3_key = f"generate_image/team_{image.group_id}/{storage_id}.png"

        try:
            s3_service.delete_file(s3_key)
        except Exception as e:
            logger.warning(f"S3 파일 삭제 실패 (계속 진행): {e}")

        await db.execute(
            delete(ImageStorage).where(ImageStorage.storage_id == storage_id)
        )
        await db.commit()

        return {"message": "이미지가 삭제되었습니다."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 삭제 실패: {str(e)}")


@router.get("/images/{storage_id}")
async def get_gallery_image_detail(
    storage_id: str,
    db: AsyncSession = Depends(get_async_db),
    current_user: Dict = Depends(get_current_user)
):
    """갤러리 이미지 상세 조회."""
    try:
        from sqlalchemy import select

        # ── 로컬(외부 볼륨) 모드 ────────────────────────────────
        if settings.IMAGE_STORAGE_TYPE == "local":
            from app.models.generated_image import GeneratedImage

            result = await db.execute(
                select(GeneratedImage).where(GeneratedImage.storage_id == storage_id)
            )
            image = result.scalar_one_or_none()
            if not image:
                raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
            return _serialize_generated(image)

        # ── S3 레거시 모드 ─────────────────────────────────────
        from app.models.image_storage import ImageStorage

        result = await db.execute(
            select(ImageStorage).where(ImageStorage.storage_id == storage_id)
        )
        image = result.scalar_one_or_none()

        if not image:
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")

        s3_service = get_s3_service()
        if image.s3_url:
            url_parts = image.s3_url.split('/', 3)
            if len(url_parts) > 3:
                s3_key = url_parts[3]
            else:
                s3_key = f"generate_image/team_{image.group_id}/{storage_id}.png"
        else:
            s3_key = f"generate_image/team_{image.group_id}/{storage_id}.png"

        presigned_url = s3_service.generate_presigned_url(s3_key)

        return {
            "storage_id": image.storage_id,
            "group_id": image.group_id,
            "created_at": image.created_at.isoformat() if image.created_at else None,
            "s3_url": presigned_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 상세 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 상세 조회 실패: {str(e)}")
