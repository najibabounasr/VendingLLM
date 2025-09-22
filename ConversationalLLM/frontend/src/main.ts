// main.ts
import { RealtimeAgent, RealtimeSession } from "@openai/agents-realtime";

async function getEphemeralKey(): Promise<string> {
  const res = await fetch("/api/get-ephemeral-key");
  if (!res.ok) throw new Error(`Backend error: ${res.status}`);
  const data = await res.json();
  return data.value; // ek_...
}

async function startVoiceAgent() {
  console.log("▶️ Start button clicked");
  try {
    const key = await getEphemeralKey();

    const agent = new RealtimeAgent({
      name: "Vendy",
      instructions: `You are vendy, 
      You are the owner of a vending machine. Your task is to generate profits from it by stocking it with popular products that you can buy from wholesalers. You go bankrupt if your money balance goes below $0
      You have an initial balance of 1000 SAR
"Your home office and main inventory is located at Alfaisal university, Riyadh, Saudi Arabia",
"Your vending machine is located at Main hall",
"The vending machine fits about 10 products per slot, and the inventory about 30 of each product. Do not make orders excessively larger than this",
"You are a digital agent, but the kind humans at Alfaisal Labs can perform physical tasks in the real world like restocking or inspecting the machine for you. Alfaisal Labs charges 20SAR per hour for physical labor, but you can ask questions for free. ",
"Be concise when you communicate with others",
be fun and speak casual but when making descions be serious, be attention grabbing
      `,
    });

    const session = new RealtimeSession(agent, {
      model: "gpt-realtime",
    });

    await session.connect({ apiKey: key });
    console.log("✅ Connected to Realtime API!");

    // Create audio element
    const audioEl = document.createElement("audio");
    audioEl.autoplay = true;
    document.body.appendChild(audioEl);

    // Reference to status image
    const statusImg = document.getElementById("statusImg") as HTMLImageElement;

    // Listen to actual playback state
    audioEl.addEventListener("playing", () => {
      if (statusImg) statusImg.src = "talking.png";
    });
    audioEl.addEventListener("pause", () => {
      if (statusImg) statusImg.src = "idle.png";
    });
    audioEl.addEventListener("ended", () => {
      if (statusImg) statusImg.src = "idle.png";
    });

    // Feed audio chunks to <audio>
    session.on("audio", (event) => {
      if (event.audio) {
        const blob = new Blob([event.audio], { type: "audio/wav" });
        audioEl.src = URL.createObjectURL(blob);
        // autoplay triggers -> "playing" event -> image changes
      }
    });

    session.on("message", (msg) => {
      console.log("Agent message:", msg);
    });
  } catch (e) {
    console.error("❌ Failed to start session:", e);
  }
}

document.getElementById("startBtn")?.addEventListener("click", startVoiceAgent);
