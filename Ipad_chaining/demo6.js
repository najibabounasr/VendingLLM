// server.js
import express from "express";
import "dotenv/config";
import path from "path";
import { fileURLToPath } from "url";
import { WebSocketServer } from "ws";
import { RealtimeAgent, RealtimeSession } from "@openai/agents/realtime";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

// serve /public
app.use(express.static(path.join(__dirname, "public")));
app.get("/", (_req, res) => res.sendFile(path.join(__dirname, "public", "index.html")));

const server = app.listen(PORT, () => console.log(`http://localhost:${PORT}`));

// single WS server
const wss = new WebSocketServer({ noServer: true });
server.on("upgrade", (req, socket, head) => {
  wss.handleUpgrade(req, socket, head, ws => wss.emit("connection", ws, req));
});

wss.on("connection", async (ws) => {
  const agent = new RealtimeAgent({
    name: "Assistant",
    instructions: "You are a helpful assistant.",
  });

  const session = new RealtimeSession(agent);
  await session.connect({ apiKey: process.env.OPENAI_API_KEY });

  // Immediately ask the model so you see something even without audio
    await session.sendText?.("Say 'ping' if you can hear me.");

    // Stream model text → browser
    session.on?.("response", (resp) => {
    // Try to pull a string; fall back to raw JSON
    // this is basically just parsing:
    const text = resp?.text ?? resp?.message ?? JSON.stringify(resp);
    ws.send(JSON.stringify({ type: "model_text", text }));
    });

    // Optional: let browser send text questions
    // (client will send {type:"ask", text:"..."} )
    ws.on("message", async (msg, isBinary) => {
    if (isBinary) {
        await session.sendAudio?.(msg);
        return;
    }
    try {
        const m = JSON.parse(msg.toString());
        if (m.type === "rate" && Number(m.sampleRate)) { clientRate = Number(m.sampleRate); return; }
        if (m.type === "stop") { const wav = pcm16ToWav(Buffer.concat(pcmChunks), clientRate); pcmChunks = []; ws.send(wav); return; }
        if (m.type === "ask" && m.text) { await session.sendText?.(m.text); return; }
    } catch { /* ignore */ }
    });

  // Buffer model audio (PCM16 mono) until client says "stop"
  let pcmChunks = [];
  let clientRate = 16000; // default

  // MODEL → buffer
  session.on("audio", (ab) => {
    pcmChunks.push(Buffer.from(new Uint8Array(ab)));
  });

  // CLIENT → MODEL  (binary = mic chunks, text = control)
  ws.on("message", async (msg, isBinary) => {
    if (isBinary) {
      // mic audio chunk -> model (raw PCM16)
      await session.sendAudio(msg);
      return;
    }
    // control messages (JSON)
    try {
      const m = JSON.parse(msg.toString());
      if (m.type === "rate" && Number(m.sampleRate)) {
        clientRate = Number(m.sampleRate);
        return;
      }
      if (m.type === "stop") {
        const wav = pcm16ToWav(Buffer.concat(pcmChunks), clientRate);
        pcmChunks = [];                // reset for next turn
        ws.send(wav);                  // one playable WAV back to client
        return;
      }
    } catch {
      // ignore non-JSON control frames
    }
  });

  ws.on("close", () => session.disconnect && session.disconnect());
});

// ---- helper: wrap PCM16LE mono -> WAV Buffer ----
function pcm16ToWav(pcmBuf, sampleRate = 16000) {
  const numChannels = 1, bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const header = Buffer.alloc(44);

  header.write('RIFF', 0);
  header.writeUInt32LE(36 + pcmBuf.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);                  // PCM chunk size
  header.writeUInt16LE(1, 20);                   // PCM format
  header.writeUInt16LE(numChannels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(bitsPerSample, 34);
  header.write('data', 36);
  header.writeUInt32LE(pcmBuf.length, 40);

  return Buffer.concat([header, pcmBuf]);
}
