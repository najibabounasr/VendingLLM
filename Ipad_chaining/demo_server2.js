import express from "express";
import { WebSocketServer } from "ws";
import {WebSocket} from "ws";
import fetch from "node-fetch";
import OpenAI from "openai";
import "dotenv/config"; 
import fs from "fs";
import { RealtimeAgent, RealtimeSession } from "@openai/agents/realtime";
// Middlewear: a function that runs BETWEEM a request (to server), and a response (to client)

const app = express(); // app is an object, that represents our web server. 
const PORT = 3000; // the local port we have running. 

// Serve static frontend files
app.use(express.static("public")); // STATIC file: files that don't change on the server side. 
// app.use( 'folder' ) --> telling express 'before doing anything else, try and serve files from the public folder'
// express.static("public") --> basically between requests and responses, express will look inside the public/ folder 

const url = "wss://api.openai.com/v1/realtime?intent=transcription"
const ws = new WebSocket(
  url, {
    headers: {
      Authorization: "Bearer " + "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",
    }
  }
)

// Listen for and parse server events
// must be right after creation of websocket: 
ws.on("message", function incoming(message) {
      const data = JSON.parse(message.toString());
      console.log(data);
      
        // Save every message to a file
        fs.appendFileSync("messages.log", JSON.stringify(data) + "\n");
});

ws.on("open", function open() {
    console.log("Connected to server.");

    // Configure the session for transcription
  ws.send(JSON.stringify({
    type: "transcription_session.update",
    input_audio_format: "pcm16",
    input_audio_transcription: {
      model: "gpt-4o-mini-transcribe",
      prompt: "",
      language: ""
    },
    turn_detection: {
      type: "server_vad",
      threshold: 0.5,
      prefix_padding_ms: 300,
      silence_duration_ms: 500
    },
    input_audio_noise_reduction: {
      type: "near_field"
    },
    include: ["item.input_audio_transcription.logprobs"]
  }));

  ws.send(JSON.stringify({
    type: "input_audio_buffer.append",
    audio: "Base64EncodedAudioData"
}));
  ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
});

// REQUIREMENTS: an open WebSocket `ws` to OpenAI, session config already sent.
// Streams 16 kHz PCM16 base64 chunks while running === true.

let running = false;
let audioCtx, mic, proc;
const chunksForSave = []; // optional: keep what you sent (base64) to save later

document.getElementById("start").onclick = async () => {
  if (running || ws.readyState !== WebSocket.OPEN) return;
  running = true;
  document.getElementById("start").disabled = true;
  document.getElementById("stop").disabled  = false;

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  mic = audioCtx.createMediaStreamSource(stream);
  proc = audioCtx.createScriptProcessor(2048, 1, 1);

  proc.onaudioprocess = (e) => {
    if (!running) return;
    const f32 = e.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    const b64 = btoa(String.fromCharCode(...new Uint8Array(pcm.buffer)));
    ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio: b64 }));
    chunksForSave.push(b64); // optional
  };

  mic.connect(proc);
  proc.connect(audioCtx.destination); // required in some browsers
};

document.getElementById("stop").onclick = async () => {
  if (!running) return;
  running = false;
  document.getElementById("start").disabled = false;
  document.getElementById("stop").disabled  = true;

  // finalize current segment for transcription
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  }

  // clean up audio nodes
  try { proc.disconnect(); proc.onaudioprocess = null; } catch {}
  try { mic.disconnect(); } catch {}
  try { await audioCtx.close(); } catch {}

  // OPTIONAL: download what you recorded as a WAV (client-side)
  // (wrap base64 PCM16 chunks into a single Blob and download)
  if (chunksForSave.length) {
    const pcmBytes = chunksForSave.map(b64 => Uint8Array.from(atob(b64), c => c.charCodeAt(0)));
    const pcm = new Uint8Array(pcmBytes.reduce((a,b)=>a+b.length,0));
    let off = 0; for (const u of pcmBytes) { pcm.set(u, off); off += u.length; }
    const wav = pcm16ToWav(pcm.buffer, 16000);
    const url = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
    const a = document.createElement("a"); a.href = url; a.download = "outgoing.wav"; a.click();
    URL.revokeObjectURL(url);
    chunksForSave.length = 0;
  }
};

// Tiny WAV header helper (PCM16 mono)
function pcm16ToWav(ab, sampleRate=16000) {
  const pcm = new Uint8Array(ab);
  const header = new ArrayBuffer(44);
  const v = new DataView(header);
  const write = (o,s) => { for (let i=0;i<s.length;i++) v.setUint8(o+i, s.charCodeAt(i)); };
  write(0,"RIFF"); v.setUint32(4, 36 + pcm.length, true);
  write(8,"WAVE"); write(12,"fmt "); v.setUint32(16,16,true);
  v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,sampleRate,true); v.setUint32(28,sampleRate*2,true);
  v.setUint16(32,2,true); v.setUint16(34,16,true);
  write(36,"data"); v.setUint32(40, pcm.length, true);
  return new Blob([header, pcm]);
}



const server = app.listen(PORT, function () {
  console.log("Server running at http://localhost:" + PORT);
});

server.on("upgrade", function (req, socket, head) {
  wss.handleUpgrade(req, socket, head, function (ws) {
    wss.emit("connection", ws, req);
  });
});