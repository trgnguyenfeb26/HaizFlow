import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from yt_dlp.utils import ExtractorError

from haizflow.services import video_download
from haizflow.services.xiaohongshu_video import HaizFlowXhsLinkIE, note_url_from_redirect

NOTE_ID = "6a33cbbb000000001101f409"
NOTE_URL = f"https://www.xiaohongshu.com/discovery/item/{NOTE_ID}?xsec_token=abc%3D&xsec_source=app_share"
LOGIN_REDIRECT = (
    "https://www.xiaohongshu.com/login?redirectPath=http%3A%2F%2Fwww.xiaohongshu.com%2Fdiscovery%2Fitem%2F"
    f"{NOTE_ID}%3Fapp_platform%3Dios%26xsec_source%3Dapp_share%26type%3Dvideo%26xsec_token%3Dabc%3D"
    "%26share_id%3Dtracking"
)


@pytest.mark.parametrize("value", [
    LOGIN_REDIRECT,
    f"https://www.xiaohongshu.com/explore/{NOTE_ID}?xsec_token=abc=&xsec_source=app_share&share_id=tracking",
    f"http://xiaohongshu.com/discovery/item/{NOTE_ID}/?xsec_source=app_share&xsec_token=abc=",
])
def test_share_redirects_resolve_to_one_canonical_note(value):
    assert note_url_from_redirect(value) == NOTE_URL


@pytest.mark.parametrize("value", [
    "https://www.xiaohongshu.com/login",
    "https://www.xiaohongshu.com/login?redirectPath=https%3A%2F%2Fevil.example%2Fexplore%2F" + NOTE_ID,
    "https://www.xiaohongshu.com/user/profile/64004cdc000000001002a991",
    f"https://xiaohongshu.com.evil.example/explore/{NOTE_ID}",
    f"ftp://www.xiaohongshu.com/explore/{NOTE_ID}",
])
def test_redirects_off_a_note_are_rejected(value):
    assert note_url_from_redirect(value) == ""


def test_note_links_are_normalized_and_short_links_kept_for_resolution():
    assert video_download.validate_video_url(f"Xem nhé https://www.xiaohongshu.com/explore/{NOTE_ID}?share_id=x") == (
        f"https://www.xiaohongshu.com/discovery/item/{NOTE_ID}", "Xiaohongshu")
    assert video_download.validate_video_url("xhslink.com/o/143WGzf54Gy") == (
        "https://xhslink.com/o/143WGzf54Gy", "Xiaohongshu")
    with pytest.raises(ValueError, match="Xiaohongshu"):
        video_download.validate_video_url("https://www.xiaohongshu.com/user/profile/64004cdc000000001002a991")


def test_short_link_extractor_hands_the_note_to_yt_dlp():
    extractor = HaizFlowXhsLinkIE()
    response = Mock(url=LOGIN_REDIRECT)
    with patch.object(extractor, "_request_webpage", return_value=response):
        result = extractor._real_extract("http://xhslink.com/o/143WGzf54Gy")
    assert result["url"] == NOTE_URL
    assert result["ie_key"] == "XiaoHongShu"
    response.close.assert_called_once()


def test_short_link_to_a_non_note_page_is_an_expected_error():
    extractor = HaizFlowXhsLinkIE()
    with patch.object(extractor, "_request_webpage", return_value=Mock(url="https://www.xiaohongshu.com/explore")), \
            pytest.raises(ExtractorError, match="does not point to a note"):
        extractor._real_extract("http://xhslink.com/o/143WGzf54Gy")


def test_short_links_route_through_the_resolver_extractor():
    downloader = Mock()
    video_download._extract_video_info(downloader, "https://xhslink.com/o/abc", download=False)
    assert isinstance(downloader.add_info_extractor.call_args.args[0], HaizFlowXhsLinkIE)
    assert downloader.extract_info.call_args.kwargs["ie_key"] == HaizFlowXhsLinkIE.ie_key()


def test_xiaohongshu_inspects_as_a_browser_from_the_first_attempt():
    with patch.object(video_download, "_load_yt_dlp", return_value=SimpleNamespace()), patch.object(
        video_download, "_inspect_video_info",
        return_value={"title": "Clip", "duration": 36, "extractor_key": "XiaoHongShu", "webpage_url": NOTE_URL},
    ) as inspect:
        metadata = video_download.inspect_video_url("http://xhslink.com/o/143WGzf54Gy")
    assert inspect.call_args.kwargs["impersonate"] is True
    assert (metadata.platform, metadata.url) == ("Xiaohongshu", NOTE_URL)
    assert video_download._impersonate_on_attempt("YouTube", 0) is False


PROFILE = "https://www.xiaohongshu.com/user/profile/6125a9ee0000000001000c9e"


def profile_page(note_id):
    state = {"user": {"userPageData": {"basicInfo": {"nickname": "Kênh thử"}}, "notes": [[
        {"id": "", "xsecToken": "tok=", "noteCard": {"noteId": note_id, "type": "video", "displayTitle": "Video A",
                                                       "cover": {"urlDefault": "https://img/cover.jpg"}}},
        {"id": "", "xsecToken": "tok2=", "noteCard": {"noteId": "a" * 24, "type": "normal", "displayTitle": "Ảnh"}},
    ], []]}}
    return f"<script>window.__INITIAL_STATE__={json.dumps(state).replace('null', 'undefined')}</script>"


def test_profile_links_keep_their_share_token_and_reject_note_links():
    from haizflow.services.channel_import import validate_channel_url

    url, platform = validate_channel_url(f"{PROFILE}/?xsec_token=abc%3D&xsec_source=pc_user&from=share")
    assert (url, platform) == (f"{PROFILE}?xsec_token=abc%3D&xsec_source=pc_user", "Xiaohongshu")
    with pytest.raises(ValueError, match="profile link"):
        validate_channel_url(f"https://www.xiaohongshu.com/explore/{NOTE_ID}")


def test_profile_reader_lists_video_notes_with_their_tokens():
    from haizflow.services import social_channel

    with patch.object(social_channel, "_read", return_value=profile_page(NOTE_ID)):
        info = social_channel.inspect_profile(PROFILE, "Xiaohongshu", {}, 10, __import__("threading").Event())
    assert info["title"] == "Kênh thử"
    assert info["entries"] == [{"id": NOTE_ID, "title": "Video A", "uploader": "Kênh thử",
                                "url": f"https://www.xiaohongshu.com/discovery/item/{NOTE_ID}"
                                       "?xsec_token=tok%3D&xsec_source=pc_user",
                                "thumbnail": "https://img/cover.jpg"}]


def test_profile_reader_explains_that_hidden_note_ids_need_a_login():
    from haizflow.services import social_channel

    with patch.object(social_channel, "_read", return_value=profile_page("")), \
            pytest.raises(RuntimeError, match="logged-in"):
        social_channel.inspect_profile(PROFILE, "Xiaohongshu", {}, 10, __import__("threading").Event())


def test_rednote_domain_is_the_same_service():
    from haizflow.services import social_channel
    from haizflow.services.channel_import import validate_channel_url

    rednote_profile = PROFILE.replace("xiaohongshu.com", "rednote.com")
    # The profile stays on the pasted domain, where the user's login cookies belong.
    assert validate_channel_url(f"{rednote_profile}?xsec_token=abc")[0] == f"{rednote_profile}?xsec_token=abc"
    assert video_download.validate_video_url(f"https://www.rednote.com/explore/{NOTE_ID}?xsec_token=abc=") == (
        f"https://www.xiaohongshu.com/discovery/item/{NOTE_ID}?xsec_token=abc%3D", "Xiaohongshu")
    page = profile_page(NOTE_ID).replace('"exposed"', '"ids": new Set([]), "exposed"')
    page = page.replace('{"user"', '{"comments": {"ids": new Set(["a", "b"])}, "user"')
    with patch.object(social_channel, "_read", return_value=page):
        info = social_channel.inspect_profile(rednote_profile, "Xiaohongshu", {}, 10, __import__("threading").Event())
    assert [entry["id"] for entry in info["entries"]] == [NOTE_ID]


def test_browser_cookies_are_read_from_the_real_windows_profile(tmp_path, monkeypatch):
    import haizflow.config

    profile = tmp_path / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
    profile.mkdir(parents=True)
    monkeypatch.setattr(haizflow.config, "NATIVE_WINDOWS_USERPROFILE", str(tmp_path))
    monkeypatch.setattr(video_download.os, "name", "nt")
    options = video_download._youtube_dl_options({"cookie_browser": "chrome"})
    # Not HaizFlow's redirected %LOCALAPPDATA%, where no browser profile exists.
    assert options["cookiesfrombrowser"] == ("chrome", str(profile))
    assert video_download._youtube_dl_options({"cookie_browser": "edge"})["cookiesfrombrowser"] == ("edge",)


def test_encrypted_browser_cookies_point_to_cookies_txt():
    from haizflow.desktop.channel_import import ChannelImportCoordinator

    coordinator = ChannelImportCoordinator()
    coordinator.setCookieBrowser("chrome")
    message = coordinator._safe_message("ERROR: Failed to decrypt with DPAPI. See issue 10927")
    assert "cookies.txt" in message and "encrypts" in message
