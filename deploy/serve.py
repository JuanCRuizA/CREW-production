"""Production server for CREW-production on Oracle Cloud: serves the same handler Vercel used, on port 8000.

Every path answers with the scoring contract (GET describes the model, POST scores one record).
Caddy terminates HTTPS in front of this process.
"""
import os
import sys
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "api"))
import score  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    if score.BUNDLE_ERROR is not None:
        print(f"Starting with an unusable bundle: {score.BUNDLE_ERROR}", file=sys.stderr)
    print(f"CREW-production scoring API on port {port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), score.handler).serve_forever()
