import pytest
import numpy as np

from bfe import binning, quality_metrics, embeddings, proxies, scoring, stratification, deduplication
from bfe.frame_info import FrameInfo

# ---- Tests for binning.py ----

def test_bin_altitude():
    assert binning.bin_altitude(10, 20) == "high"
    assert binning.bin_altitude(30, 20) == "low"

def test_bin_view():
    assert binning.bin_view(1.0, 1.8) == "oblique"
    assert binning.bin_view(2.0, 1.8) == "nadir"

def test_bin_cover():
    assert binning.bin_cover(0.4, 0.5) == "sparse"
    assert binning.bin_cover(0.6, 0.5) == "dense"

def test_bin_lighting():
    assert binning.bin_lighting(100, 115) == "dark"
    assert binning.bin_lighting(120, 115) == "bright"

# --- Tests for quality_metrics.py ---

@pytest.fixture
def dummy_image():
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

@pytest.fixture
def dummy_bgr_image():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

def test_variance_of_laplacian(dummy_image):
    score = quality_metrics.variance_of_laplacian(dummy_image)
    assert isinstance(score, float)

def test_estimate_contrast(dummy_image):
    score = quality_metrics.estimate_contrast(dummy_image)
    assert isinstance(score, float)

def test_exposure_metrics(dummy_image):
    mean, under, over = quality_metrics.exposure_metrics(dummy_image)
    assert isinstance(mean, float)
    assert isinstance(under, float)
    assert isinstance(over, float)

def test_exposure_score_from_metrics():
    score = quality_metrics.exposure_score_from_metrics(128, 0.1, 0.1)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_estimate_noise_score(dummy_image):
    score = quality_metrics.estimate_noise_score(dummy_image)
    assert isinstance(score, float)
    assert 0 <= score <= 1

# --- Tests for embeddings.py ---

def test_hsv_histogram(dummy_bgr_image):
    hist = embeddings.hsv_histogram(dummy_bgr_image)
    assert isinstance(hist, np.ndarray)
    assert hist.shape == (16+16+16,)

def test_lowres_embedding(dummy_bgr_image):
    emb = embeddings.lowres_embedding(dummy_bgr_image)
    assert isinstance(emb, np.ndarray)

def test_combined_embed(dummy_bgr_image):
    hsv, low, emb = embeddings.combined_embed(dummy_bgr_image)
    assert isinstance(hsv, np.ndarray)
    assert isinstance(low, np.ndarray)
    assert isinstance(emb, np.ndarray)

def test_cosine_similarity():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = np.array([1, 0, 0])
    assert embeddings.cosine_similarity(a, b) == pytest.approx(0)
    assert embeddings.cosine_similarity(a, c) == pytest.approx(1)

# --- Tests for proxies.py ---

def test_altitude_proxy(dummy_image):
    proxy = proxies.altitude_proxy(dummy_image)
    assert isinstance(proxy, float)

def test_view_entropy(dummy_image):
    proxy = proxies.view_entropy(dummy_image)
    assert isinstance(proxy, float)

def test_green_cover_ratio(dummy_bgr_image):
    proxy = proxies.green_cover_ratio(dummy_bgr_image)
    assert isinstance(proxy, float)
    assert 0 <= proxy <= 1

def test_classify_lighting(dummy_image):
    proxy = proxies.classify_lighting(dummy_image)
    assert isinstance(proxy, float)

# --- Tests for scoring.py ---

@pytest.fixture
def dummy_frame_info():
    return FrameInfo(
        idx=0, t_sec=0.0, bgr=np.array([]), gray=np.array([]),
        sharpness=150, contrast=50, exposure_score=0.8, noise_score=0.9,
        hsv_hist=np.random.rand(48), lowres_vec=np.random.rand(2304), embed=np.random.rand(2352)
    )

def test_ml_frame_scorer_init():
    scorer = scoring.MLFrameScorer()
    assert scorer is not None

def test_ml_frame_scorer_score(dummy_frame_info):
    scorer = scoring.MLFrameScorer()
    frames = [dummy_frame_info]
    scorer.score(frames)
    assert frames[0].ml_score > 0
    assert "quality" in frames[0].subscores
    assert "content_novelty" in frames[0].subscores

# --- Tests for stratification.py ---

@pytest.fixture
def dummy_strat_config():
    return {
        "stratification": {
            "axes": {
                "altitude": ["low", "high"],
                "view": ["nadir", "oblique"],
                "cover": ["sparse", "dense"],
                "lighting": ["dark", "bright"],
            }
        }
    }

def test_agro_stratifier_init(dummy_strat_config):
    stratifier = stratification.AgroStratifier(dummy_strat_config)
    assert stratifier is not None
    assert len(stratifier.targets) == 16 # 2*2*2*2

# --- Tests for deduplication.py ---

@pytest.fixture
def dummy_frames_for_dedup():
    frames = []
    for i in range(5):
        fi = FrameInfo(
            idx=i, t_sec=float(i), bgr=np.array([]), gray=np.array([]),
            sharpness=100, contrast=50, exposure_score=0.8, noise_score=0.9,
            hsv_hist=np.random.rand(48), lowres_vec=np.random.rand(2304), embed=np.random.rand(2352),
            ml_score=0.5 + i * 0.1
        )
        frames.append(fi)
    # Add a duplicate
    frames[2].embed = frames[0].embed
    return frames

def test_deduplicate_quality_first(dummy_frames_for_dedup):
    selected = deduplication.deduplicate_quality_first(dummy_frames_for_dedup, cosine_threshold=0.99)
    assert len(selected) == 4

def test_deduplicate_dbscan(dummy_frames_for_dedup):
    selected = deduplication.deduplicate_dbscan(dummy_frames_for_dedup, eps=0.1)
    assert len(selected) <= 5
