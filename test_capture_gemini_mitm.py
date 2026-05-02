import unittest

import capture_gemini_mitm as capture


class CaptureGeminiMitmTest(unittest.TestCase):
    def test_body_text_keeps_content_up_to_500kb(self):
        content = b"a" * (400 * 1024)

        self.assertEqual(capture.body_text(content), "a" * (400 * 1024))

    def test_body_text_trims_content_over_500kb(self):
        content = b"a" * (501 * 1024)

        body = capture.body_text(content)

        self.assertEqual(len(body), 500 * 1024 + len("\n<trimmed>"))
        self.assertTrue(body.endswith("\n<trimmed>"))

    def test_multipart_body_preserves_boundary_and_trims_large_binary_part(self):
        body = (
            b"--abc123\r\n"
            b'Content-Disposition: form-data; name="metadata"\r\n\r\n'
            b'{"name":"cat"}\r\n'
            b"--abc123\r\n"
            b'Content-Disposition: form-data; name="file"; filename="cat.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
            + (b"x" * (capture.MULTIPART_PART_LIMIT + 1))
            + b"\r\n--abc123--\r\n"
        )

        text = capture.body_text(body, "multipart/form-data; boundary=abc123")

        self.assertIn("--abc123", text)
        self.assertIn('{"name":"cat"}', text)
        self.assertIn("<part-trimmed", text)
        self.assertIn("--abc123--", text)

    def test_request_event_includes_content_metadata(self):
        class Headers(dict):
            pass

        class Request:
            pretty_host = "gemini.google.com"
            method = "POST"
            pretty_url = "https://gemini.google.com/upload"
            headers = Headers({"content-type": "application/json", "content-length": "2"})
            raw_content = b"{}"

        class Flow:
            id = "flow-1"
            request = Request()
            metadata = {}

        events = []
        old_append = capture.append
        capture.append = events.append
        try:
            capture.request(Flow())
        finally:
            capture.append = old_append

        self.assertEqual(events[0]["content_type"], "application/json")
        self.assertEqual(events[0]["content_length"], 2)


if __name__ == "__main__":
    unittest.main()
