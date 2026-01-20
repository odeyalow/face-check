import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import rtspRelay from "rtsp-relay";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const publicDir = path.join(__dirname, "public");
const knownDir = path.join(publicDir, "known");

app.use(express.static(publicDir));
app.use("/known", express.static(knownDir));

const { proxy } = rtspRelay(app);
const defaultRtspUrl = process.env.RTSP_URL;
const useNativeFfmpeg = process.env.USE_NATIVE_FFMPEG === "1";

app.get("/api/known", (_req, res) => {
  let files = [];

  if (fs.existsSync(knownDir)) {
    files = fs.readdirSync(knownDir)
      .filter((file) => /\.(png|jpe?g)$/i.test(file));
  }

  res.json({ files });
});

app.get("/known", (_req, res) => {
  let files = [];
  if (fs.existsSync(knownDir)) {
    files = fs.readdirSync(knownDir)
      .filter((file) => /\.(png|jpe?g)$/i.test(file));
  }

  const links = files
    .map((file) => `<li><a href="/known/${encodeURIComponent(file)}">${file}</a></li>`)
    .join("");

  res.send(`<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Known Faces</title>
  </head>
  <body>
    <h3>Known Faces</h3>
    <ul>${links || "<li>No files</li>"}</ul>
  </body>
</html>`);
});

app.ws("/api/stream", (ws, req) => {
  const reqUrl = new URL(req.url, `http://${req.headers.host}`);
  const rtspUrl = reqUrl.searchParams.get("url") || defaultRtspUrl;

  if (!rtspUrl) {
    ws.close(1008, "Missing RTSP url");
    return;
  }

  proxy({ url: rtspUrl, transport: "tcp", useNativeFFmpeg: useNativeFfmpeg })(ws);
});

app.listen(3000, () => console.log("http://localhost:3000"));
