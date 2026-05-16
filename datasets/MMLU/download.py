import os
import shutil
from typing import Iterator

import requests
import tarfile
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from KVCOMM.utils.log import logger

# (connect timeout, read timeout between chunks) — large tar needs a generous read timeout
_REQUEST_TIMEOUT = (30, 600)

# Hugging Face hosts the same Hendrycks archive; CDN is often much faster than Berkeley direct.
_HF_MMLU_TAR = (
    "https://huggingface.co/datasets/Stevross/mmlu/resolve/main/data.tar"
)
_BERKELEY_MMLU_TAR = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

_HF_HUB_REPO = "Stevross/mmlu"
_HF_HUB_FILENAME = "data.tar"


def _tar_url_candidates() -> Iterator[str]:
    """URLs to try, in order. Set MMLU_DATA_TAR_URL to force a single mirror."""
    custom = os.environ.get("MMLU_DATA_TAR_URL", "").strip()
    if custom:
        yield custom
        return
    yield _HF_MMLU_TAR
    yield _BERKELEY_MMLU_TAR


def _request_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _requests_session() -> requests.Session:
    """Session with retries for flaky TLS / CDN resets."""
    s = requests.Session()
    retries = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _download_via_hf_hub(tar_path: str) -> bool:
    """Use huggingface_hub (respects HF_ENDPOINT, token, hub cache; more robust than raw GET)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    try:
        cached = hf_hub_download(
            repo_id=_HF_HUB_REPO,
            filename=_HF_HUB_FILENAME,
            repo_type="dataset",
        )
        if os.path.abspath(cached) == os.path.abspath(tar_path):
            return True
        shutil.copy2(cached, tar_path)
        return True
    except Exception as exc:
        logger.warning("huggingface_hub download failed: {}", exc)
        return False


def _stream_download(url: str, tar_path: str) -> None:
    expected_size = None
    session = _requests_session()
    with session.get(
        url,
        allow_redirects=True,
        stream=True,
        timeout=_REQUEST_TIMEOUT,
        headers=_request_headers(),
    ) as r:
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        expected_size = int(cl) if cl is not None else None
        chunk_size = 1024 * 1024
        with open(tar_path, "wb") as f, tqdm(
            desc="data.tar",
            total=expected_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
        ) as pbar:
            for data in r.iter_content(chunk_size=chunk_size):
                if data:
                    f.write(data)
                    pbar.update(len(data))
    if expected_size is not None:
        got = os.path.getsize(tar_path)
        if got != expected_size:
            raise OSError(
                f"Incomplete download: expected {expected_size} bytes, got {got}"
            )


def download():

    this_file_path = os.path.split(__file__)[0]
    tar_path = os.path.join(this_file_path, "data.tar")
    if not os.path.exists(tar_path):
        custom_url = os.environ.get("MMLU_DATA_TAR_URL", "").strip()
        ok = False
        if not custom_url:
            logger.info(
                "Trying huggingface_hub (repo={}, file={}; set HF_ENDPOINT for mirrors)",
                _HF_HUB_REPO,
                _HF_HUB_FILENAME,
            )
            ok = _download_via_hf_hub(tar_path)
        if ok:
            logger.info("Saved to {}", tar_path)
        else:
            last_error: Exception | None = None
            for url in _tar_url_candidates():
                logger.info("Downloading {}", url)
                try:
                    _stream_download(url, tar_path)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if os.path.exists(tar_path):
                        os.unlink(tar_path)
                    logger.warning("Download from {} failed: {}", url, exc)
            if last_error is not None:
                raise last_error
            logger.info("Saved to {}", tar_path)

    data_path = os.path.join(this_file_path, "data")
    if not os.path.exists(data_path):
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(this_file_path)
        except (tarfile.ReadError, tarfile.TarError, EOFError) as exc:
            if os.path.isdir(data_path):
                shutil.rmtree(data_path, ignore_errors=True)
            if os.path.exists(tar_path):
                os.unlink(tar_path)
            logger.error(
                "data.tar is corrupt or truncated ({}). Removed archive and partial "
                "'data/'; run again to re-download.",
                exc,
            )
            raise
        logger.info("Saved to {}", data_path)


if __name__ == "__main__":
    download()
