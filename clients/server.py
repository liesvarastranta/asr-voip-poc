"""Web voice client — serves HTML page + LiveKit token endpoint."""
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from livekit import api


LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
PORT = int(os.getenv("WEB_PORT", "8080"))


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/token":
            token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
                .with_identity("web-user") \
                .with_name("Web User") \
                .with_grants(api.VideoGrants(room_join=True, room="agent-room")) \
                .with_room_config(
                    api.RoomConfiguration(
                        agents=[
                            api.RoomAgentDispatch(agent_name="voice-agent-id"),
                        ],
                    ),
                ) \
                .to_jwt()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"token": token, "url": LIVEKIT_URL}).encode())
        else:
            super().do_GET()


if __name__ == "__main__":
    import os as _os
    _os.chdir(_os.path.join(_os.path.dirname(__file__), "..", "clients"))
    print(f"Serving on http://localhost:{PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
