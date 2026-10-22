"""Resolve Xiaohongshu (RedNote) share links to the note page yt-dlp reads.

App share sheets copy an xhslink.com short link. For anonymous visitors it
redirects to a login page whose ``redirectPath`` carries the real note URL and
its ``xsec_token``. The pinned yt-dlp has no extractor for the short host, so
follow the redirect once and hand only a validated note URL to its
XiaoHongShu extractor. No login, cookie or signing is added here.
"""

import re
from urllib.parse import parse_qs, urlencode, urlparse

from yt_dlp.extractor.common import InfoExtractor
from yt_dlp.utils import ExtractorError

_NOTE_PATH = re.compile(r"/(?:explore|discovery/item)/([0-9a-f]{24})/?")
# Only the parameters the note page needs; share tracking IDs are dropped.
_NOTE_QUERY_KEYS = ("xsec_token", "xsec_source")


# rednote.com is the same service under its international domain.
SITE_DOMAINS = ("xiaohongshu.com", "rednote.com")


def _is_xiaohongshu_host(hostname: str) -> bool:
    hostname = (hostname or "").lower().rstrip(".")
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in SITE_DOMAINS)


def note_url_from_redirect(value: str) -> str:
    """Return a canonical note URL from a resolved share link, or ``""``.

    Note pages open without login on either domain; the canonical form uses
    xiaohongshu.com, the only one yt-dlp's extractor accepts.
    """
    try:
        parsed = urlparse(str(value or ""))
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not _is_xiaohongshu_host(parsed.hostname or ""):
        return ""
    if parsed.path.rstrip("/") in {"/login", "/website-login/captcha"}:
        redirects = parse_qs(parsed.query).get("redirectPath", [])
        return note_url_from_redirect(redirects[0]) if len(redirects) == 1 else ""
    note = _NOTE_PATH.fullmatch(parsed.path)
    if not note:
        return ""
    query = parse_qs(parsed.query)
    kept = [(key, query[key][0]) for key in _NOTE_QUERY_KEYS if len(query.get(key, [])) == 1]
    suffix = f"?{urlencode(kept)}" if kept else ""
    return f"https://www.xiaohongshu.com/discovery/item/{note[1]}{suffix}"


class HaizFlowXhsLinkIE(InfoExtractor):
    IE_NAME = "haizflow:xhslink"
    _VALID_URL = r"https?://(?:www\.)?xhslink\.com/(?P<id>[^?#]+)"

    def _real_extract(self, url):
        short_id = self._match_id(url)
        response = self._request_webpage(url, short_id, "Resolving Xiaohongshu share link")
        note_url = note_url_from_redirect(response.url)
        response.close()
        if not note_url:
            raise ExtractorError("This Xiaohongshu link does not point to a note.", expected=True)
        return self.url_result(note_url, ie="XiaoHongShu")
