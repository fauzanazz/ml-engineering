from collections import OrderedDict

from webcam_effect.video_filters.background_blur import BackgroundBlurFilter
from webcam_effect.video_filters.background_blur_lite import BackgroundBlurLiteFilter
from webcam_effect.video_filters.background_replace import BackgroundReplaceFilter
from webcam_effect.video_filters.base import FilterAssets, VideoFilter
from webcam_effect.video_filters.beauty_soften import BeautySoftenFilter
from webcam_effect.video_filters.cartoon_face import CartoonFaceFilter
from webcam_effect.video_filters.face_sticker import FaceStickerFilter
from webcam_effect.video_filters.hand_magic_trail import HandMagicTrailFilter
from webcam_effect.video_filters.neon_face_mesh import NeonFaceMeshFilter
from webcam_effect.video_filters.pose_aura import PoseAuraFilter
from webcam_effect.video_filters.virtual_glasses import VirtualGlassesFilter


def build_filters(provider: object, assets: FilterAssets) -> OrderedDict[str, VideoFilter]:
    filters = [
        BackgroundBlurFilter(provider),
        VirtualGlassesFilter(provider, assets.glasses),
        NeonFaceMeshFilter(provider),
        BeautySoftenFilter(provider),
        CartoonFaceFilter(provider),
        HandMagicTrailFilter(provider),
        PoseAuraFilter(provider),
        FaceStickerFilter(provider, assets.sticker),
        BackgroundReplaceFilter(provider, assets.background),
        BackgroundBlurLiteFilter(provider),
    ]
    return OrderedDict((filter_item.spec.key, filter_item) for filter_item in filters)


def filter_names(filters: OrderedDict[str, VideoFilter]) -> list[str]:
    return [filter_item.spec.name for filter_item in filters.values()]
