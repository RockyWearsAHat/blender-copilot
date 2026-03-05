"""Scrape .blend character models from SmutBase / Open3DLab network.

SmutBase and Open3DLab host thousands of free Blender character models
with rich metadata (name, creator, description, tags, universe, software).
These are excellent training data for human/character geometry — high-quality,
community-curated, and all in .blend format.

SmutBase: ~4000+ models (adult content, character-focused)
Open3DLab: ~2000+ models (general purpose 3D resources)

Both sites share the same infrastructure and page layout.

Usage:
    python -m scrapers.smutbase_scraper --output data/raw/smutbase
    python -m scrapers.smutbase_scraper --site open3dlab --output data/raw/open3dlab
"""

import argparse
import json
import logging
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, urlencode

import requests
from bs4 import BeautifulSoup

try:
    import rarfile
    HAS_RARFILE = True
except ImportError:
    HAS_RARFILE = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed — fall back to env vars

from .utils import (
    setup_logging, load_config, ensure_dir, download_file,
    save_metadata, load_progress, save_progress, is_blend_file,
    GlobalHashRegistry,
)

logger = logging.getLogger(__name__)

SITES = {
    "smutbase": {
        "base_url": "https://smutba.se",
        "name": "SmutBase",
        "rss": "https://smutba.se/rss/",
    },
    "open3dlab": {
        "base_url": "https://open3dlab.com",
        "name": "Open3DLab",
        "rss": "https://open3dlab.com/rss/",
    },
}

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def create_session(site_key: str = "smutbase") -> requests.Session:
    """Create a requests session with age-verification cookies.

    If SMUTBASE_EMAIL / SMUTBASE_PASSWORD are set in .env, logs in to
    unlock premium downloads.  Falls back to free-only CDN downloads.
    """
    session = requests.Session()
    session.headers.update(SESSION_HEADERS)

    base_url = SITES[site_key]["base_url"]
    domain = urlparse(base_url).netloc

    session.cookies.set("ageverified", "1", domain=domain)
    session.cookies.set("cookie_consent", "accepted", domain=domain)

    # SmutBase and Open3DLab share login infrastructure — credentials live
    # under OPEN3DLAB_USER / OPEN3DLAB_PASSWORD in .env
    username = os.environ.get("OPEN3DLAB_USER", "").strip()
    password = os.environ.get("OPEN3DLAB_PASSWORD", "").strip()

    if username and password:
        _login_smutbase(session, base_url, username, password)
    else:
        logger.info(
            f"Created anonymous session for {domain} (free CDN downloads only). "
            "Set OPEN3DLAB_USER / OPEN3DLAB_PASSWORD in .env to unlock premium files."
        )

    return session


def _login_smutbase(session: requests.Session, base_url: str,
                    username: str, password: str) -> bool:
    """Log in to a SmutBase/Open3DLab site via Django auth.

    Both sites use /accounts/login/ with csrfmiddlewaretoken.
    The login field is 'username' (not email).
    """
    login_url = f"{base_url}/accounts/login/"
    try:
        resp = session.get(login_url, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        csrf = soup.find("input", {"name": "csrfmiddlewaretoken"})
        csrf_token = csrf["value"] if csrf else ""

        if not csrf_token:
            # Fall back to cookie-based CSRF
            csrf_token = session.cookies.get("csrftoken", "")

        resp = session.post(
            login_url,
            data={
                "csrfmiddlewaretoken": csrf_token,
                "login": username,     # Django allauth uses "login" (not "username")
                "password": password,
                "remember": "on",
                "next": "/",
            },
            headers={"Referer": login_url},
            timeout=30,
            allow_redirects=True,
        )

        # If redirected away from /accounts/login, login succeeded
        if "/accounts/login" not in resp.url:
            logger.info(f"Logged in to {base_url} as '{username}' (premium downloads unlocked)")
            return True
        else:
            logger.warning(f"Login failed for {base_url} (user='{username}') — check credentials")
            return False

    except Exception as e:
        logger.warning(f"Login error for {base_url}: {e}")
        return False


def get_listing_page(session: requests.Session, base_url: str,
                     page: int = 1,
                     software_tag: str = "blender",
                     sort_by: str = "recent") -> list[dict]:
    """Scrape one page of the browse/listing view.

    sort_by: 'recent' (default), 'popular', 'downloads', 'name'
    Returns list of dicts with 'project_id', 'title', 'url'.
    """
    order_map = {
        "recent": "-last_file_date",
        "popular": "-views",
        "downloads": "-downloads",
        "name": "title",
    }
    params = {
        "page": page,
        "order_by": order_map.get(sort_by, "-last_file_date"),
    }
    if software_tag:
        params["software_tag"] = software_tag

    url = f"{base_url}/?{urlencode(params)}"

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch listing page {page}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    projects = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.search(
            r"/project/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/",
            href,
        )
        if match:
            project_id = match.group(1)
            title_tag = link.find(["h3", "h4", "h5", "span"])
            title = title_tag.get_text(strip=True) if title_tag else ""

            if not title:
                title = link.get_text(strip=True)

            if project_id and title and len(title) > 1:
                full_url = urljoin(base_url, href)
                if not any(p["project_id"] == project_id for p in projects):
                    projects.append({
                        "project_id": project_id,
                        "title": title,
                        "url": full_url,
                    })

    return projects


def get_project_details(session: requests.Session, project_url: str,
                        project_id: str,
                        include_premium: bool = False) -> dict | None:
    """Scrape a project page for metadata and download link.

    Returns dict with title, description, creator, tags, download_url, etc.
    """
    try:
        resp = session.get(project_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch project {project_id}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    desc_div = soup.find("div", class_="description") or soup.find(
        "div", class_="project-description"
    )
    if not desc_div:
        desc_sections = soup.find_all("h2", string=re.compile(r"Description", re.I))
        if desc_sections:
            desc_div = desc_sections[0].find_next_sibling()
    description = desc_div.get_text(strip=True) if desc_div else ""

    creator = ""
    creator_links = soup.find_all("a", href=re.compile(r"/user/\d+/"))
    if creator_links:
        creator = creator_links[0].get_text(strip=True)

    tags = {}
    for tag_link in soup.find_all("a", href=True):
        href = tag_link["href"]
        if "query=" in href:
            tag_text = tag_link.get_text(strip=True)
            if tag_text and len(tag_text) > 1 and not tag_text.startswith("..."):
                parent = tag_link.parent
                if parent:
                    parent_text = parent.get_text(strip=True)
                    if "Universe" in parent_text or "Property" in parent_text:
                        tags.setdefault("universe", []).append(tag_text)
                    elif "Character" in parent_text:
                        tags.setdefault("character", []).append(tag_text)
                    elif "Software" in parent_text:
                        tags.setdefault("software", []).append(tag_text)
                    elif "Misc" in parent_text or "General" in parent_text:
                        tags.setdefault("misc", []).append(tag_text)
                    else:
                        tags.setdefault("other", []).append(tag_text)

    downloads = None
    views = None

    # Strategy 0: Parse from the file table (columns: Filename, Downloads, Created, Filesize)
    # Also build file_id → filename map so we can label download entries later.
    file_id_to_filename: dict[str, str] = {}
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("td")[:4]]
        if "downloads" in headers:
            dl_col = headers.index("downloads")
            for row in table.find_all("tr")[2:]:  # skip header rows
                cells = row.find_all("td")
                if len(cells) > dl_col:
                    val = cells[dl_col].get_text(strip=True).replace(",", "")
                    if val.isdigit():
                        downloads = (downloads or 0) + int(val)
                # Extract filename (first cell) and file_id from download link in this row
                row_dl_link = row.find("a", class_="download-link")
                if row_dl_link:
                    href = row_dl_link.get("href", "")
                    fid_m = re.search(r"/download/(\d+)/", href)
                    if fid_m and cells:
                        fname = cells[0].get_text(strip=True)
                        if fname:
                            file_id_to_filename[fid_m.group(1)] = fname

    # Strategy 1: look for text nodes with digit-only content near labels
    if downloads is None:
        for text_node in soup.find_all(string=True):
            text = str(text_node).strip()
            if text.isdigit():
                prev = text_node.find_previous()
                if prev and "Downloads" in prev.get_text():
                    downloads = int(text)
                elif prev and "Views" in prev.get_text():
                    views = int(text)
    # Strategy 2: look for stats in common patterns like "123 Downloads"
    if downloads is None:
        for el in soup.find_all(string=re.compile(r'\d+\s*Downloads', re.IGNORECASE)):
            m = re.search(r'(\d[\d,]*)\s*Downloads', str(el), re.IGNORECASE)
            if m:
                downloads = int(m.group(1).replace(',', ''))
                break
    if views is None:
        for el in soup.find_all(string=re.compile(r'\d+\s*Views', re.IGNORECASE)):
            m = re.search(r'(\d[\d,]*)\s*Views', str(el), re.IGNORECASE)
            if m:
                views = int(m.group(1).replace(',', ''))
                break

    # Find FREE download links: class="download-link" WITHOUT "download-link--premium"
    # Premium links point to /members/ (paywall), free ones point to
    # /project/file/download/{file_id}/{server}/
    download_files = []
    seen_file_ids = set()

    for a in soup.find_all("a", class_="download-link"):
        classes = a.get("class", [])
        is_premium = "download-link--premium" in classes
        # Include premium links only when authenticated (include_premium=True)
        if is_premium and not include_premium:
            continue
        href = a.get("href", "")
        if "/project/file/download/" not in href:
            continue
        full_url = urljoin(project_url, href)
        file_id_match = re.search(r"/download/(\d+)/", href)
        file_id = file_id_match.group(1) if file_id_match else href
        if file_id in seen_file_ids:
            continue
        seen_file_ids.add(file_id)
        download_files.append({
            "url": full_url,
            "file_id": file_id,
            "hint_filename": file_id_to_filename.get(file_id, ""),
        })

    # Fallback: look for any /project/file/download/ links
    if not download_files:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "/project/file/download/" in href:
                full_url = urljoin(project_url, href)
                file_id_match = re.search(r"/download/(\d+)/", href)
                file_id = file_id_match.group(1) if file_id_match else href
                if file_id in seen_file_ids:
                    continue
                seen_file_ids.add(file_id)
                download_files.append({
                    "url": full_url,
                    "file_id": file_id,
                    "hint_filename": file_id_to_filename.get(file_id, ""),
                })

    # Use first free download link as the primary download_url
    download_url = download_files[0]["url"] if download_files else ""

    filename = ""
    filesize = ""

    for td in soup.find_all("td"):
        td_text = td.get_text(strip=True)
        if td_text.endswith(".blend"):
            filename = td_text
        elif "MB" in td_text or "GB" in td_text:
            filesize = td_text

    if not filename:
        for text in soup.find_all(string=re.compile(r"\.blend\b")):
            match = re.search(r"(\S+\.blend)\b", str(text))
            if match:
                filename = match.group(1)
                break

    license_text = ""
    license_div = soup.find(string=re.compile(r"Creative Commons|Licence|License", re.I))
    if license_div:
        parent = license_div.find_parent()
        if parent:
            license_text = parent.get_text(strip=True)[:200]

    return {
        "project_id": project_id,
        "title": title,
        "description": description[:2000],
        "creator": creator,
        "tags": tags,
        "downloads": downloads,
        "views": views,
        "download_url": download_url,
        "download_files": download_files,
        "filename": filename,
        "filesize": filesize,
        "license": license_text,
        "source_url": project_url,
    }


def get_projects_from_rss(site_key: str = "smutbase") -> list[dict]:
    """Get latest projects from RSS feed (faster than scraping pages)."""
    rss_url = SITES[site_key]["rss"]
    projects = []

    try:
        resp = requests.get(rss_url, headers=SESSION_HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")

        for item in soup.find_all("item"):
            title = item.find("title")
            link = item.find("link")
            desc = item.find("description")

            if link:
                url = link.get_text(strip=True)
                match = re.search(
                    r"/project/([0-9a-f-]{36})/", url
                )
                if match:
                    projects.append({
                        "project_id": match.group(1),
                        "title": title.get_text(strip=True) if title else "",
                        "url": url,
                        "description_preview": (
                            desc.get_text(strip=True)[:500] if desc else ""
                        ),
                    })
    except Exception as e:
        logger.warning(f"Failed to fetch RSS for {site_key}: {e}")

    return projects


def _extract_blend_from_archive(archive_path: Path, output_dir: Path) -> list[Path]:
    """Extract .blend files from a .rar or .zip archive.

    Also handles the case where the "archive" is actually a raw .blend file
    (BlendSwap sometimes returns application/octet-stream with BLENDER magic).

    Returns list of extracted .blend file paths.
    """
    extracted = []
    suffix = archive_path.suffix.lower()

    # Check magic bytes first — may be a .blend disguised as .zip
    try:
        magic = archive_path.read_bytes()[:4]
        if magic == b'BLEN':
            # It's a raw .blend file — just rename it
            dest = output_dir / archive_path.with_suffix(".blend").name
            if not dest.exists():
                import shutil
                shutil.move(str(archive_path), str(dest))
            return [dest]
    except Exception:
        pass

    try:
        if suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                for name in zf.namelist():
                    if name.lower().endswith(".blend") and not name.startswith("__MACOSX"):
                        safe = re.sub(r'[^\w\-.]', '_', Path(name).name)
                        target = output_dir / safe
                        if not target.exists():
                            with zf.open(name) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                            extracted.append(target)

        elif suffix == ".rar" and HAS_RARFILE:
            with rarfile.RarFile(archive_path, "r") as rf:
                for name in rf.namelist():
                    if name.lower().endswith(".blend"):
                        safe = re.sub(r'[^\w\-.]', '_', Path(name).name)
                        target = output_dir / safe
                        if not target.exists():
                            with rf.open(name) as src, open(target, "wb") as dst:
                                dst.write(src.read())
                            extracted.append(target)

        elif suffix == ".rar" and not HAS_RARFILE:
            result = subprocess.run(
                ["unrar", "e", "-o-", str(archive_path), str(output_dir)],
                capture_output=True, timeout=120,
            )
            if result.returncode == 0:
                for f in output_dir.iterdir():
                    if f.suffix.lower() == ".blend" and f.stat().st_size > 50_000:
                        extracted.append(f)

    except Exception as e:
        logger.warning(f"Archive extraction failed for {archive_path.name}: {e}")

    return extracted


def download_project_file(session: requests.Session, details: dict,
                          output_dir: Path,
                          hash_registry: GlobalHashRegistry | None = None,
                          ) -> str | None:
    """Download the .blend file(s) for a project.

    Tries all free download links.  Handles .rar/.zip archives by
    extracting .blend files.  Returns path to first valid .blend,
    or None on failure.
    """
    download_files = details.get("download_files", [])
    if not download_files:
        url = details.get("download_url", "")
        if not url or "/members" in url:
            return None
        download_files = [{"url": url, "file_id": "0"}]

    project_id = details["project_id"]
    first_blend = None

    # --- Filter: prefer .blend entries; skip files that are clearly texture/material ZIPs ---
    _NON_BLEND_PAT = re.compile(
        r'texture|tex_|_tex[._]|_mat[._]|material|hdri|skybox|normal[_.]|roughness|diffuse|albedo',
        re.IGNORECASE,
    )
    blend_entries = [
        e for e in download_files
        if e.get("hint_filename", "").lower().endswith(".blend")
    ]
    zip_entries   = [
        e for e in download_files
        if not e.get("hint_filename", "").lower().endswith(".blend")
        and not _NON_BLEND_PAT.search(e.get("hint_filename", ""))
    ]
    texture_entries = [
        e for e in download_files
        if _NON_BLEND_PAT.search(e.get("hint_filename", ""))
    ]
    # Try .blend first, then ambiguous ZIPs, ignore texture-only entries
    ordered_entries = blend_entries or (zip_entries + texture_entries) or download_files
    if blend_entries and len(download_files) > 1:
        skipped = [e.get("hint_filename") or e["file_id"] for e in texture_entries + zip_entries]
        if skipped:
            logger.debug(f"[{project_id}] Skipping non-blend files: {skipped}")

    for dl_entry in ordered_entries:
        download_url = dl_entry["url"]
        file_id = dl_entry.get("file_id", "0")

        if "/members" in download_url or "/login" in download_url:
            continue

        try:
            # ── Step 1: Hit the download page and extract the real CDN URL ──
            page_resp = session.get(
                download_url,
                stream=False,
                timeout=60,
                headers={"Referer": details.get("source_url", "")},
            )
            page_resp.raise_for_status()

            real_url = None
            ct = page_resp.headers.get("content-type", "").lower()

            if "text/html" in ct:
                # Parse download page for CDN URL
                dl_soup = BeautifulSoup(page_resp.text, "html.parser")

                # Method 1: JS redirect  window.location = "https://...rar"
                for script in dl_soup.find_all("script"):
                    if script.string and "window.location" in script.string:
                        m = re.search(
                            r'window\.location\s*=\s*["\']([^"\']+)["\']',
                            script.string,
                        )
                        if m:
                            real_url = m.group(1)
                            break

                # Method 2: fallback "Click here" link
                if not real_url:
                    fallback = dl_soup.find("a", href=re.compile(
                        r"files\.sfmlab\.com|files\.open3dlab\.com|"
                        r"files\.smutba\.se|\.rar|\.zip|\.blend|\.7z"
                    ))
                    if fallback:
                        real_url = fallback["href"]

                # Method 3: any link containing known CDN domains
                if not real_url:
                    for a in dl_soup.find_all("a", href=True):
                        href = a["href"]
                        if any(cdn in href for cdn in [
                            "files.sfmlab.com", "files.open3dlab.com",
                            "files.smutba.se", "cdn.open3dlab.com/content",
                        ]):
                            real_url = href
                            break

                if not real_url:
                    # Check if redirected to login/members
                    if "/members" in str(page_resp.url) or "/login" in str(page_resp.url):
                        logger.warning(
                            f"File {file_id}: redirected to login wall"
                        )
                    else:
                        logger.warning(
                            f"File {file_id}: download page has no CDN link"
                        )
                    continue
            else:
                # Direct download (not HTML) — use response directly
                real_url = None  # handled below

            # ── Step 2: Download the actual file from CDN ──
            if real_url:
                logger.debug(f"File {file_id}: CDN URL = {real_url[:120]}...")
                resp = session.get(
                    real_url,
                    stream=True,
                    timeout=300,
                    headers={"Referer": download_url},
                )
                resp.raise_for_status()
            else:
                # Non-HTML response from step 1 — use it directly
                resp = page_resp

            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                logger.warning(
                    f"Rejected file {file_id}: CDN returned HTML. "
                    f"URL: {resp.url}"
                )
                continue

            cd = resp.headers.get("content-disposition", "")
            cd_match = re.search(r'filename="?([^"]+)"?', cd)
            if cd_match:
                fname = cd_match.group(1).strip()
            elif real_url:
                fname = Path(urlparse(real_url).path).name or f"{project_id}_{file_id}"
            else:
                fname = Path(urlparse(download_url).path).name or f"{project_id}_{file_id}"

            safe_name = re.sub(r'[^\w\-.]', '_', fname)
            output_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = output_dir / (safe_name + ".tmp")

            downloaded = 0

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)

            if not tmp_path.exists():
                continue

            if downloaded < 50_000:
                logger.warning(
                    f"Rejected: {safe_name} — only {downloaded} bytes"
                )
                tmp_path.unlink(missing_ok=True)
                continue

            with open(tmp_path, "rb") as f:
                magic = f.read(8)

            if magic[:7] == b"BLENDER" or magic[:4] == b"\x28\xb5\x2f\xfd":
                # Standard BLENDER magic, or Zstandard-compressed .blend (Blender 3.0+)
                out_path = output_dir / safe_name
                if not out_path.suffix.lower().endswith(".blend"):
                    out_path = out_path.with_suffix(".blend")
                tmp_path.rename(out_path)
                logger.info(f"Downloaded: {out_path.name} ({downloaded / 1e6:.1f}MB)")
                if not first_blend:
                    first_blend = str(out_path)

            elif magic[:4] == b"Rar!" or magic[:7] == b"Rar!\x1a\x07\x00":
                rar_path = tmp_path.with_suffix(".rar")
                tmp_path.rename(rar_path)
                blends = _extract_blend_from_archive(rar_path, output_dir)
                rar_path.unlink(missing_ok=True)
                if blends:
                    logger.info(
                        f"Extracted {len(blends)} .blend from {safe_name} "
                        f"({downloaded / 1e6:.1f}MB)"
                    )
                    if not first_blend:
                        first_blend = str(blends[0])
                else:
                    logger.warning(f"No .blend found in archive {safe_name}")

            elif magic[:2] == b"PK":
                zip_path = tmp_path.with_suffix(".zip")
                tmp_path.rename(zip_path)
                blends = _extract_blend_from_archive(zip_path, output_dir)
                zip_path.unlink(missing_ok=True)
                if blends:
                    logger.info(
                        f"Extracted {len(blends)} .blend from {safe_name} "
                        f"({downloaded / 1e6:.1f}MB)"
                    )
                    if not first_blend:
                        first_blend = str(blends[0])
                else:
                    logger.warning(f"No .blend found in ZIP {safe_name}")

            else:
                logger.warning(
                    f"Rejected: {safe_name} — unknown format "
                    f"(magic: {magic[:8]!r})"
                )
                tmp_path.unlink(missing_ok=True)

            time.sleep(3.0)

        except Exception as e:
            logger.warning(f"Download failed for file {file_id}: {e}")
            continue

    if first_blend:
        blend_path = Path(first_blend)
        if hash_registry and not hash_registry.check_and_add(blend_path):
            logger.debug(f"Duplicate file (hash match): {blend_path.name}")
            blend_path.unlink(missing_ok=True)
            return None

    return first_blend


def scrape_site(site_key: str = "smutbase",
                output_dir: str = "data/raw/smutbase",
                max_pages: int = 250,
                download_files: bool = True):
    """Main scraping loop for a SmutBase/Open3DLab site.

    Paginates through listing pages, extracts project metadata,
    and optionally downloads .blend files.
    """
    site = SITES[site_key]
    base_url = site["base_url"]
    out_path = ensure_dir(output_dir)
    meta_dir = ensure_dir(out_path / "metadata")
    files_dir = ensure_dir(out_path / "files")

    progress = load_progress(out_path / ".progress")
    session = create_session(site_key)
    hash_registry = GlobalHashRegistry(out_path.parent)

    # Use premium links if credentials are configured
    has_credentials = bool(
        os.environ.get("OPEN3DLAB_USER") and os.environ.get("OPEN3DLAB_PASSWORD")
    )
    include_premium = has_credentials

    logger.info(f"Scraping {site['name']} ({base_url})")
    logger.info(f"  Output: {out_path}")
    logger.info(f"  Max pages: {max_pages}")
    logger.info(f"  Premium downloads: {'enabled (logged in as ' + os.environ.get('OPEN3DLAB_USER','') + ')' if include_premium else 'disabled (set OPEN3DLAB_USER/OPEN3DLAB_PASSWORD in .env)'}")
    logger.info(f"  Already processed: {len(progress)} projects")
    logger.info(f"  Global hash registry: {hash_registry.count} known files")

    total_found = 0
    total_downloaded = 0
    total_metadata = 0
    consecutive_empty = 0

    for page in range(1, max_pages + 1):
        logger.info(f"  Page {page}/{max_pages}...")
        projects = get_listing_page(session, base_url, page=page)

        if not projects:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                logger.info(f"  No more projects found after page {page}")
                break
            continue
        consecutive_empty = 0

        total_found += len(projects)

        for proj in projects:
            pid = proj["project_id"]
            if pid in progress:
                continue

            time.sleep(2.0)

            details = get_project_details(session, proj["url"], pid,
                                          include_premium=include_premium)
            if not details:
                save_progress(out_path / ".progress", pid)
                continue

            save_metadata(str(meta_dir), pid, details)
            total_metadata += 1

            if download_files and details.get("download_url"):
                dl_path = download_project_file(
                    session, details, files_dir,
                    hash_registry=hash_registry,
                )
                if dl_path:
                    total_downloaded += 1
                    logger.info(
                        f"    Downloaded: {details['title'][:50]} "
                        f"({details.get('filesize', '?')})"
                    )

            save_progress(out_path / ".progress", pid)
            progress.add(pid)  # Update in-memory set to prevent re-processing

        # Save hash registry every page
        hash_registry.save()

        logger.info(
            f"  Page {page}: {len(projects)} projects, "
            f"running total: {total_metadata} metadata, "
            f"{total_downloaded} files"
        )

    logger.info(
        f"\n{site['name']} scraping complete:\n"
        f"  Found: {total_found} projects\n"
        f"  Metadata saved: {total_metadata}\n"
        f"  Files downloaded: {total_downloaded}\n"
    )


def scrape_batch(site_key: str = "smutbase",
                 output_dir: str = "data/raw/smutbase",
                 batch_size: int = 50,
                 sort_by: str = "recent") -> int:
    """Download a batch of models. Used by BackgroundDataPuller.

    sort_by: 'recent', 'popular', 'downloads'
    Returns number of files downloaded in this batch.
    """
    site = SITES[site_key]
    base_url = site["base_url"]
    out_path = ensure_dir(output_dir)
    meta_dir = ensure_dir(out_path / "metadata")
    files_dir = ensure_dir(out_path / "files")

    progress = load_progress(out_path / ".progress")
    session = create_session(site_key)
    hash_registry = GlobalHashRegistry(out_path.parent)

    downloaded = 0
    page = 1

    while downloaded < batch_size and page <= 300:
        projects = get_listing_page(session, base_url, page=page,
                                    sort_by=sort_by)
        if not projects:
            break

        for proj in projects:
            if downloaded >= batch_size:
                break

            pid = proj["project_id"]
            if pid in progress:
                continue

            time.sleep(2.5)

            # include_premium if credentials available
            _has_creds = bool(
                os.environ.get("OPEN3DLAB_USER")
                and os.environ.get("OPEN3DLAB_PASSWORD")
            )
            details = get_project_details(session, proj["url"], pid,
                                          include_premium=_has_creds)
            if not details:
                save_progress(out_path / ".progress", pid)
                continue

            save_metadata(str(meta_dir), pid, details)

            # ── Quality gate ──
            from scrapers.quality_filter import passes_quality_filter
            passed, reason = passes_quality_filter(
                title=details.get("title", ""),
                description=details.get("description", ""),
                tags=details.get("tags", []),
                downloads=details.get("downloads"),
                category=details.get("category", ""),
            )
            if not passed:
                logger.info(f"Quality skip '{details.get('title','')[:40]}': {reason}")
                save_progress(out_path / ".progress", pid)
                progress.add(pid)
                continue

            if details.get("download_url"):
                dl_path = download_project_file(
                    session, details, files_dir,
                    hash_registry=hash_registry,
                )
                if dl_path:
                    downloaded += 1

            save_progress(out_path / ".progress", pid)
            progress.add(pid)  # Update in-memory set to prevent re-processing

        page += 1

    hash_registry.save()
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Scrape .blend models from SmutBase / Open3DLab"
    )
    parser.add_argument(
        "--site", choices=["smutbase", "open3dlab"], default="smutbase",
        help="Which site to scrape",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--max-pages", type=int, default=250,
        help="Maximum listing pages to scrape",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Only save metadata, don't download files",
    )
    args = parser.parse_args()

    setup_logging(args.site)

    output = args.output or f"data/raw/{args.site}"

    scrape_site(
        site_key=args.site,
        output_dir=output,
        max_pages=args.max_pages,
        download_files=not args.no_download,
    )


if __name__ == "__main__":
    main()
