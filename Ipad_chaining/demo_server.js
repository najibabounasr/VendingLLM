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

const server = app.listen(PORT, function () {
  console.log("Server running at http://localhost:" + PORT);
});

const url = "wss://api.openai.com/v1/realtime?intent=transcription"
const ws = new WebSocket(
  url, {
    headers: {
      Authorization: "Bearer " + process.getActiveResourcesInfo.,
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




// Create the openai object
// the bottom code is unneccessary and can be removed:
const openai = new OpenAI({ // is a websocket server instance. 
  apiKey: process.env.OPENAI_API_KEY,   // reads from .env securely. 
});
audio_file = open('/audio/audio.mp3',"rb")

const transcription = await client.audio.transcriptions.create({
  model : "gpt-4o-transcribe",
  file : fs.createReadStream("/path/to/file/audio.mp3"),
}
)

// Create an agent object. 
const agent = new RealtimeAgent({
    name: "Assistant",
    instructions: "You are a helpful assistant. Attached to a vending machine. ",
});




server.on("upgrade", function (req, socket, head) {
  wss.handleUpgrade(req, socket, head, function (ws) {
    wss.emit("connection", ws, req);
  });
});
