# 🌙 Kimi Proxy (Portable)

This folder contains everything you need to run Moonshot Kimi-K2.5 with apps that don't support it natively (like memU Desktop).

## � Setup (Required)

> [!IMPORTANT]
> The `.env` file contains your API key and is **never committed** to the repository. You must create it yourself.

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and replace the placeholder with your real Moonshot API key:
   ```
   MOONSHOT_API_KEY=sk-your-real-key-here
   ```

## �🚀 How to Run
1. Open Terminal.
2. Drag `start.sh` into the terminal window and press Enter.
   *(Or run `./start.sh` if you are already in the folder)*
3. Keep the window open.

## ⚙️ App Configuration
Configure your Desktop App with these exact settings:

| Setting | Value |
|---------|-------|
| **Base URL** | `http://127.0.0.1:8080/v1` |
| **API Key** | `any` (Key is hidden in .env) |
| **Model** | `kimi-k2.5` |

---

## 🌟 Features

### Universal Bridge
Works with apps set to **OpenAI** OR **Anthropic (Claude)** mode. Automatically translates between both protocol formats so Kimi-K2.5 works as a drop-in replacement.

### Auto-Translation
Converts Anthropic-format requests (messages, tools, streaming SSE) to OpenAI format and back. Handles:
- Message format conversion (roles, content blocks, tool use/results)
- Tool definition translation (`input_schema` → `parameters`)
- Streaming SSE responses (Anthropic event format)
- Tool call ID bidirectional mapping between Kimi and Anthropic formats
- `reasoning_content` injection for Kimi's thinking model compatibility

### Vision Support
*Available in the paid version only.*

### 👁️ Vision Loop-Back (Autonomous Self-Vision)
*Available in the paid version only.*

### Local Embeddings
Handles embeddings locally on your CPU using `all-MiniLM-L6-v2` via `sentence-transformers`. Free, private, and fast — never sent to Moonshot.

---

## 📁 Files

| File | Purpose |
|------|---------|\
| `start.sh` | Launcher — downloads dependencies automatically |
| `main.py` | Core proxy logic — bridge, tool translation |
| `.env.example` | Template — copy to `.env` and add your API key |
| `.env` | **Your API key (create locally, never committed)** |

---

## 🔌 MCP Servers

The proxy works alongside MCP servers registered in your desktop app. The following are used by the agent:

| Server | Purpose |
|--------|---------|
| `playwright` (built-in) | Browser automation, screenshots, navigation |
| `ElevenLabs` | Text-to-speech, voice cloning, audio tools |
| `mimi-vision` | Custom vision server (*Available in the paid version only*) |
| `remotion-documentation` | Video documentation reference |
| `videodb-director` | Video database and director tools |

### mimi-vision MCP Server
*Available in the paid version only.*

---

## 🔍 Health Check
Visit `http://127.0.0.1:8080/health` to verify the proxy is running and see active ID mappings and reasoning cache size.

---

## 🔒 Security
- Your Moonshot API key is stored in `.env` and never exposed to the app.
- Local embeddings never leave your machine.
- The proxy runs on `127.0.0.1` (localhost only) — not accessible from outside your machine.
