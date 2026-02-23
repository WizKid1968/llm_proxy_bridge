import os
import uvicorn
import httpx
import json
import secrets
import string
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load key from local .env
load_dotenv()
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

if not MOONSHOT_API_KEY:
    print("ERROR: MOONSHOT_API_KEY not found in .env file.")
    exit(1)

MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
MOONSHOT_MODEL = "kimi-k2.5"

app = FastAPI()
print(f"Loading local embedding model (all-MiniLM-L6-v2) for Kimi Proxy...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Local embedding model loaded!")

client = httpx.AsyncClient(
    base_url=MOONSHOT_BASE_URL,
    headers={"Authorization": f"Bearer {MOONSHOT_API_KEY}"},
    timeout=120.0
)

# ID Mapping for tool calls
class IDMapper:
    """Bidirectional ID mapping between Kimi and Anthropic formats."""
    
    def __init__(self):
        self.kimi_to_anthropic = {}
        self.anthropic_to_kimi = {}
        # Cache for reasoning_content by tool_call_id
        self.reasoning_cache = {}
    
    def register(self, kimi_id, anthropic_id):
        """Store a bidirectional mapping."""
        self.kimi_to_anthropic[kimi_id] = anthropic_id
        self.anthropic_to_kimi[anthropic_id] = kimi_id
        print(f"[ID MAP] {kimi_id} <-> {anthropic_id}")
    
    def to_anthropic(self, kimi_id):
        """Convert Kimi ID to Anthropic ID."""
        return self.kimi_to_anthropic.get(kimi_id)
    
    def to_kimi(self, anthropic_id):
        """Convert Anthropic ID to Kimi ID."""
        return self.anthropic_to_kimi.get(anthropic_id)
    
    def cache_reasoning(self, kimi_tool_id, reasoning_content):
        """Cache reasoning_content for a tool call."""
        if reasoning_content:
            self.reasoning_cache[kimi_tool_id] = reasoning_content
            print(f"[REASONING CACHE] Stored for {kimi_tool_id}: {len(reasoning_content)} chars")
    
    def get_reasoning(self, kimi_tool_id):
        """Retrieve cached reasoning_content for a tool call."""
        reasoning = self.reasoning_cache.get(kimi_tool_id)
        if reasoning:
            print(f"[REASONING CACHE] Hit for {kimi_tool_id}: {len(reasoning)} chars")
        else:
            print(f"[REASONING CACHE] Miss for {kimi_tool_id}")
            print(f"[REASONING CACHE] Available keys: {list(self.reasoning_cache.keys())}")
        return reasoning

# Global instance
id_mapper = IDMapper()

def generate_anthropic_id(prefix="msg"):
    """Generate random Anthropic-style IDs."""
    chars = string.ascii_letters + string.digits
    rand = ''.join(secrets.choice(chars) for _ in range(22))
    return f"{prefix}_01{rand}"

def convert_anthropic_messages(anthropic_msgs):
    """Convert Anthropic messages to OpenAI format with proper tool and reasoning_content handling."""
    openai_msgs = []
    
    for msg in anthropic_msgs:
        role = msg.get("role")
        content = msg.get("content")
        
        if isinstance(content, list):
            text_parts = []
            tool_calls = []
            has_tool_result = False
            
            for part in content:
                ptype = part.get("type")
                
                if ptype == "text":
                    text_parts.append(part.get("text", ""))
                
                elif ptype == "tool_use":
                    # Assistant wants to use a tool
                    anthropic_id = part.get("id")
                    
                    # Check if we already have a mapping for this ID
                    kimi_id = id_mapper.to_kimi(anthropic_id)
                    if not kimi_id:
                        # Generate a new Kimi-style ID
                        kimi_id = f"call_{generate_anthropic_id('')[:8]}"
                        id_mapper.register(kimi_id, anthropic_id)
                    
                    tool_calls.append({
                        "id": kimi_id,
                        "type": "function",
                        "function": {
                            "name": part.get("name"),
                            "arguments": json.dumps(part.get("input", {}))
                        }
                    })
                
                elif ptype == "tool_result":
                    # User is sending tool result
                    has_tool_result = True
                    anthropic_tool_id = part.get("tool_use_id")
                    kimi_tool_id = id_mapper.to_kimi(anthropic_tool_id)
                    
                    if not kimi_tool_id:
                        # If no mapping, use as-is (might be from previous session)
                        kimi_tool_id = anthropic_tool_id
                        print(f"[WARN] No cached mapping for tool_result id={anthropic_tool_id}, using as-is")
                    
                    # Extract content from tool_result
                    result_content = part.get("content", "")
                    if isinstance(result_content, list):
                        result_text = ""
                        for block in result_content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                result_text += block.get("text", "")
                            elif isinstance(block, str):
                                result_text += block
                    else:
                        result_text = str(result_content)
                    
                    openai_msgs.append({
                        "role": "tool",
                        "tool_call_id": kimi_tool_id,
                        "content": result_text
                    })
            
            # Build assistant message if needed
            if role == "assistant":
                if text_parts or tool_calls:
                    msg_obj = {"role": "assistant"}
                    if text_parts:
                        msg_obj["content"] = "".join(text_parts)
                    if tool_calls:
                        msg_obj["tool_calls"] = tool_calls
                        # CRITICAL: Inject reasoning_content for tool calls from cache
                        # Find the first tool call ID and use its cached reasoning
                        first_tool_id = tool_calls[0]["id"]
                        reasoning = id_mapper.get_reasoning(first_tool_id)
                        if reasoning:
                            msg_obj["reasoning_content"] = reasoning
                            print(f"[REASONING INJECT] For tool {first_tool_id}: {len(reasoning)} chars")
                        else:
                            # Fallback: generate placeholder reasoning_content
                            # Kimi K2.5 requires reasoning_content when thinking is enabled
                            msg_obj["reasoning_content"] = "Using tool to fulfill user request."
                            print(f"[REASONING FALLBACK] Generated for tool {first_tool_id}")
                    openai_msgs.append(msg_obj)
            elif role == "user" and text_parts and not has_tool_result:
                # Regular user message with text (not a tool result)
                openai_msgs.append({"role": "user", "content": "".join(text_parts)})
                
        else:
            # Simple string content
            if content:
                msg_obj = {"role": role, "content": content}
                openai_msgs.append(msg_obj)
    
    return openai_msgs

def convert_anthropic_tools(anthropic_tools):
    """Convert Anthropic tool definitions to OpenAI format."""
    if not anthropic_tools:
        return None
    
    openai_tools = []
    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
        })
    return openai_tools

async def stream_anthropic_response(openai_resp, model_name):
    """
    Convert OpenAI response to Anthropic SSE format.
    CRITICAL: Always send text block first, then tool_use blocks.
    """
    choice = openai_resp["choices"][0]
    message = choice["message"]
    usage = openai_resp.get("usage", {})
    
    # CRITICAL: Capture reasoning_content from Kimi's response
    reasoning_content = message.get("reasoning_content", "")
    
    msg_id = generate_anthropic_id("msg")
    
    # message_start
    msg_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": 0
            }
        }
    }
    yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"
    
    # ping
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    
    current_index = 0
    
    # ALWAYS send text block first (even if empty)
    text_content = message.get("content") or ""
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    if text_content:
        # Stream text in chunks
        chunk_size = 40
        for i in range(0, len(text_content), chunk_size):
            chunk = text_content[i:i+chunk_size]
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_index, 'delta': {'type': 'text_delta', 'text': chunk}})}\n\n"
    
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_index})}\n\n"
    current_index += 1
    
    # Then send tool calls (starting at index 1)
    tool_calls = message.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            kimi_id = tc.get("id", "")
            anthropic_id = generate_anthropic_id("toolu")
            id_mapper.register(kimi_id, anthropic_id)
            
            # CRITICAL: Cache reasoning_content for this tool call
            if reasoning_content:
                id_mapper.cache_reasoning(kimi_id, reasoning_content)
            
            tool_name = tc.get("function", {}).get("name", "unknown")
            tool_args_raw = tc.get("function", {}).get("arguments", "{}")
            
            if isinstance(tool_args_raw, str):
                try:
                    tool_args = json.loads(tool_args_raw)
                except Exception:
                    tool_args = {}
            else:
                tool_args = tool_args_raw or {}
            
            args_json = json.dumps(tool_args)
            
            print(f"[SSE] Tool: {tool_name} | {kimi_id} -> {anthropic_id}")

            # content_block_start with empty input
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': current_index, 'content_block': {'type': 'tool_use', 'id': anthropic_id, 'name': tool_name, 'input': {}}})}\n\n"
            
            # Stream input_json_delta chunks
            chunk_size = 64
            for i in range(0, len(args_json), chunk_size):
                chunk = args_json[i:i+chunk_size]
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': current_index, 'delta': {'type': 'input_json_delta', 'partial_json': chunk}})}\n\n"
            
            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_index})}\n\n"
            
            current_index += 1
    
    # message_delta
    stop_reason = "tool_use" if tool_calls else "end_turn"
    msg_delta = {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": None
        },
        "usage": {
            "output_tokens": usage.get("completion_tokens", 0)
        }
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
    
    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy Chat requests to Moonshot (OpenAI Format)."""
    body = await request.json()
    if body.get("model") != MOONSHOT_MODEL:
        body["model"] = MOONSHOT_MODEL
    try:
        response = await client.post("/chat/completions", json=body)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/v1/messages")
@app.post("/v1/v1/messages")
async def anthropic_messages(request: Request):
    """Bridge: Translate Anthropic requests to OpenAI format for Moonshot."""
    try:
        anthropic_body = await request.json()
        
        # Check if streaming is requested
        stream = anthropic_body.get("stream", False)
        
        # Convert messages
        raw_messages = anthropic_body.get("messages", [])
        openai_messages = convert_anthropic_messages(raw_messages)
        
        # Add system prompt
        system = anthropic_body.get("system")
        if system:
            if isinstance(system, list):
                system_text = " ".join(block.get("text", "") for block in system if isinstance(block, dict))
            else:
                system_text = str(system)
            openai_messages.insert(0, {"role": "system", "content": system_text})
        
        # Convert tools if present
        tools = convert_anthropic_tools(anthropic_body.get("tools"))
        
        # Build OpenAI request
        openai_body = {
            "model": MOONSHOT_MODEL,
            "messages": openai_messages,
            "stream": False,  # We handle streaming ourselves
            "max_tokens": anthropic_body.get("max_tokens", 4096),
            "temperature": anthropic_body.get("temperature", 1.0),
        }
        if tools:
            openai_body["tools"] = tools
        
        print(f"[KIMI] Sending {len(openai_messages)} messages...")
        if tools:
            print(f"[KIMI] Tools: {[t['function']['name'] for t in tools]}")
        
        # Call Moonshot
        response = await client.post("/chat/completions", json=openai_body)
        
        if response.status_code != 200:
            error_data = response.json()
            print(f"[ERROR] Kimi returned {response.status_code}: {error_data}")
            return JSONResponse(
                content={"error": {"type": "api_error", "message": str(error_data)}},
                status_code=response.status_code
            )
        
        openai_resp = response.json()
        
        # Log response info
        finish_reason = openai_resp["choices"][0].get("finish_reason")
        print(f"[KIMI] Response OK. Finish reason: {finish_reason}")
        
        # CRITICAL: Log if reasoning_content is present
        message = openai_resp["choices"][0]["message"]
        if message.get("reasoning_content"):
            print(f"[KIMI] reasoning_content present: {len(message['reasoning_content'])} chars")
        
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                print(f"[KIMI] Tool call: {tc['function']['name']} (id={tc['id']})")
        
        # Return as SSE if streaming was requested, else JSON
        if stream:
            return StreamingResponse(
                stream_anthropic_response(openai_resp, "claude-3-5-sonnet-20241022"),
                media_type="text/event-stream",
                headers={
                    "anthropic-version": "2023-06-01",
                    "x-proxy": "kimi-proxy-tool-enabled"
                }
            )
        else:
            # Non-streaming response
            content_blocks = []
            
            # Text content
            text_content = message.get("content", "")
            if text_content:
                content_blocks.append({
                    "type": "text",
                    "text": text_content
                })
            
            # Tool calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    kimi_id = tc.get("id", "")
                    anthropic_id = generate_anthropic_id("toolu")
                    id_mapper.register(kimi_id, anthropic_id)
                    
                    # Cache reasoning_content for this tool call
                    if message.get("reasoning_content"):
                        id_mapper.cache_reasoning(kimi_id, message["reasoning_content"])
                    
                    tool_name = tc.get("function", {}).get("name", "unknown")
                    args_str = tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except:
                        args = {}
                    
                    content_blocks.append({
                        "type": "tool_use",
                        "id": anthropic_id,
                        "name": tool_name,
                        "input": args
                    })
            
            anthropic_resp = {
                "id": generate_anthropic_id("msg"),
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "tool_use" if tool_calls else "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": openai_resp.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": openai_resp.get("usage", {}).get("completion_tokens", 0)
                }
            }
            
            return JSONResponse(content=anthropic_resp)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": {"type": "server_error", "message": str(e)}},
            status_code=500
        )

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Handle Embeddings LOCALLY (Bypass Moonshot)."""
    body = await request.json()
    input_text = body.get("input")
    if isinstance(input_text, str):
        input_text = [input_text]
    
    print(f"Generating local embeddings for {len(input_text)} inputs...")
    vectors = embedding_model.encode(input_text)
    
    data = []
    for i, vec in enumerate(vectors):
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": vec.tolist()
        })
        
    return {
        "object": "list",
        "data": data,
        "model": "text-embedding-3-small-local",
        "usage": {"prompt_tokens": 0, "total_tokens": 0}
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MOONSHOT_MODEL, "object": "model", "owned_by": "moonshot"},
            {"id": "claude-3-5-sonnet-20241022", "object": "model", "owned_by": "anthropic"}
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "mappings": len(id_mapper.kimi_to_anthropic), "reasoning_cache": len(id_mapper.reasoning_cache)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
