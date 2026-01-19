import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import rtspRelay from "rtsp-relay";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.static(path.join(__dirname, "public")));

const { proxy } = rtspRelay(app);
const defaultRtspUrl = process.env.RTSP_URL;

app.get("/api/known", (_req, res) => {
  const knownDir = path.join(__dirname, "public", "known");
  let files = [];

  if (fs.existsSync(knownDir)) {
    files = fs.readdirSync(knownDir)
      .filter((file) => /\.(png|jpe?g)$/i.test(file));
  }

  res.json({ files });
});

app.ws("/api/stream", (ws, req) => {
  const reqUrl = new URL(req.url, `http://${req.headers.host}`);
  const rtspUrl = reqUrl.searchParams.get("url") || defaultRtspUrl;

  if (!rtspUrl) {
    ws.close(1008, "Missing RTSP url");
    return;
  }

  proxy({ url: rtspUrl, transport: "tcp" })(ws);
});

app.listen(3000, () => console.log("http://localhost:3000"));
