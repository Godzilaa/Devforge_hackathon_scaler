"""
AI-Powered Code Analysis and Patching System with Ollama
Based on StackLoop architecture for automated debugging and code fixing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import subprocess
import tempfile
import shutil
from pathlib import Path
import json
import asyncio
import re
import httpx
import difflib
from datetime import datetime
from enum import Enum
from fastapi.responses import StreamingResponse

app = FastAPI(title="CodeIDE AI Agent API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-coder:1.3b"

# =============================================================================
# Models
# =============================================================================

class ErrorCategory(str, Enum):
    AUTO_FIXABLE = "auto_fixable"
    MANUAL_REQUIRED = "manual_required"
    UNKNOWN = "unknown"

class ExecutionResult(BaseModel):
    success: bool
    stdout: str
    stderr: str
    execution_time: float
    exit_code: int

class FileError(BaseModel):
    file_path: str
    line_number: Optional[int] = None
    error_message: str
    error_type: str
    error_category: ErrorCategory

class FilePatch(BaseModel):
    file_path: str
    original_content: str
    patched_content: str
    diff: str
    unified_diff: Optional[str] = None
    line_edits: Optional[List[Dict[str, Any]]] = None
    structured_fixes: Optional[List[Dict[str, Any]]] = None
    error_fixed: str
    fix_explanation: str
    confidence: float = Field(ge=0.0, le=1.0)

class AnalysisRequest(BaseModel):
    code: str
    language: str
    file_path: str = "main"

class AnalysisResponse(BaseModel):
    session_id: str
    execution_result: Optional[ExecutionResult] = None
    errors: List[FileError] = []
    patches: List[FilePatch] = []
    analysis: str
    can_auto_fix: bool

class ApplyPatchRequest(BaseModel):
    session_id: str
    patch_index: int

class ApplyPatchResponse(BaseModel):
    success: bool
    message: str
    updated_code: str

class LogEvent(BaseModel):
    """Represents a log entry from the AI fixing process"""
    timestamp: str
    level: str  # "info", "success", "warning", "error"
    message: str
    details: Optional[Dict[str, Any]] = None

class ContinuousFixRequest(BaseModel):
    code: str
    language: str
    file_path: str = "main"
    max_iterations: int = 5  # Max number of fix attempts

# =============================================================================
# In-memory storage (replace with Redis/DB for production)
# =============================================================================

sessions: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# Ollama Integration
# =============================================================================

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        
    async def check_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False

    async def ensure_model_available(self) -> bool:
        """Ensure the configured model is available in the local Ollama runtime.

        This will check the Ollama model list via HTTP and if the model is missing
        it will attempt to pull the model using the `ollama` CLI (requires the
        Ollama CLI to be installed and reachable in PATH). Returns True if the
        model is available after this call, False otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    # Normalize possible fields
                    models = data.get("models") or data.get("tags") or []
                    # models may be a list of dicts or strings
                    for m in models:
                        if isinstance(m, dict) and m.get("name") == self.model:
                            return True
                        if isinstance(m, str) and m == self.model:
                            return True
                    # If configured model is not present but other models exist,
                    # pick the first available local model and use it as a fallback.
                    if models:
                        first = models[0]
                        if isinstance(first, dict):
                            first_name = first.get("name")
                        else:
                            first_name = str(first)
                        if first_name:
                            print(f"Configured model '{self.model}' not found; falling back to local model '{first_name}'")
                            self.model = first_name
                            return True
        except Exception:
            # If the HTTP check fails, we'll still attempt a CLI pull as a fallback
            pass

        # If model is not present, try to pull it using the Ollama CLI
        try:
            proc = subprocess.run(["ollama", "pull", self.model], capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                # Give Ollama a moment to register the model, then re-check HTTP
                await asyncio.sleep(1)
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(f"{self.base_url}/api/tags")
                        if resp.status_code == 200:
                            data = resp.json()
                            models = data.get("models") or data.get("tags") or []
                            for m in models:
                                if isinstance(m, dict) and m.get("name") == self.model:
                                    return True
                                if isinstance(m, str) and m == self.model:
                                    return True
                except Exception:
                    return True
            else:
                # Pull failed; log output and return False
                print(f"ollama pull failed: {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
                return False
        except FileNotFoundError:
            # ollama CLI not found
            print("ollama CLI not found in PATH. Cannot pull model automatically.")
            return False
        except Exception as e:
            print(f"Error while trying to pull model via ollama CLI: {e}")
            return False
    
    async def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """Generate text using Ollama"""
        # Ensure the model exists before attempting generation. If missing,
        # try to pull it (CLI) and re-check.
        available = await self.ensure_model_available()
        if not available:
            raise HTTPException(status_code=503, detail=f"Ollama model '{self.model}' is not available and could not be pulled automatically.")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "system": system,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": 2048
                        }
                    }
                )

                # If Ollama responds with a specific model-missing error, attempt a CLI pull and retry once
                if response.status_code != 200:
                    try:
                        body = response.json()
                        msg = json.dumps(body)
                    except Exception:
                        msg = await response.aread() if hasattr(response, "aread") else str(response.text)

                    if response.status_code in (400, 404) or "model" in msg.lower():
                        # Try pulling the model and retrying once
                        pulled = await self.ensure_model_available()
                        if pulled:
                            # Retry generate
                            retry_resp = await client.post(
                                f"{self.base_url}/api/generate",
                                json={
                                    "model": self.model,
                                    "prompt": prompt,
                                    "system": system,
                                    "stream": False,
                                    "options": {
                                        "temperature": temperature,
                                        "num_predict": 2048
                                    }
                                }
                            )
                            if retry_resp.status_code == 200:
                                return retry_resp.json().get("response", "")
                            else:
                                raise HTTPException(status_code=500, detail="Ollama generation failed after pulling model")

                    raise HTTPException(status_code=500, detail=f"Ollama generation failed: {msg}")

                return response.json().get("response", "")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

ollama = OllamaClient()

# =============================================================================
# Code Execution in Sandbox
# =============================================================================

class SandboxExecutor:
    def __init__(self):
        self.temp_dirs: Dict[str, Path] = {}
    
    async def execute_code(self, code: str, language: str, file_path: str) -> ExecutionResult:
        """Execute code in an isolated temporary directory"""
        
        # Create isolated temp directory
        temp_dir = Path(tempfile.mkdtemp(prefix="codeide_sandbox_"))
        session_id = temp_dir.name
        self.temp_dirs[session_id] = temp_dir
        
        try:
            # Determine file extension and command
            ext_map = {
                "python": (".py", ["python"]),
                "javascript": (".js", ["node"]),
                "typescript": (".ts", ["ts-node"]),
            }
            
            extension, command = ext_map.get(language, (".txt", []))
            
            # Normalize `file_path`: strip directories and any existing extension
            # so we don't end up with filenames like `main.py.py`.
            safe_name = Path(file_path).name
            if safe_name.lower().endswith(extension):
                safe_name = safe_name[: -len(extension)]

            # Write code to file with the correct extension (use UTF-8)
            file_name = f"{safe_name}{extension}"
            code_file = temp_dir / file_name
            code_file.write_text(code, encoding='utf-8')
            
            # Execute
            start_time = datetime.now()
            # Capture raw bytes and decode explicitly with utf-8 (replace invalid)
            result = subprocess.run(
                command + [str(code_file)],
                cwd=temp_dir,
                capture_output=True,
                text=False,
                timeout=30
            )
            execution_time = (datetime.now() - start_time).total_seconds()

            # Decode outputs using UTF-8 with replacement for characters that can't be decoded
            stdout = result.stdout.decode('utf-8', errors='replace') if isinstance(result.stdout, (bytes, bytearray)) else str(result.stdout or '')
            stderr = result.stderr.decode('utf-8', errors='replace') if isinstance(result.stderr, (bytes, bytearray)) else str(result.stderr or '')

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                exit_code=result.returncode
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="Execution timed out after 30 seconds",
                execution_time=30.0,
                exit_code=-1
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution error: {str(e)}",
                execution_time=0.0,
                exit_code=-1
            )
    
    def cleanup(self, session_id: str):
        """Cleanup temporary directory"""
        if session_id in self.temp_dirs:
            try:
                shutil.rmtree(self.temp_dirs[session_id])
                del self.temp_dirs[session_id]
            except:
                pass

sandbox = SandboxExecutor()

# =============================================================================
# Error Analysis
# =============================================================================

class ErrorAnalyzer:
    @staticmethod
    def parse_errors(stderr: str, language: str) -> List[FileError]:
        """Parse errors from stderr output"""
        errors = []
        
        if language == "python":
            # Primary: try to parse standard multi-line Python traceback
            error_pattern = r'File "([^"]+)", line (\d+).*\n(?:\s+.*\n)*\s*(\w+Error|IndexError): (.+)'
            matches = re.finditer(error_pattern, stderr, re.MULTILINE)

            for match in matches:
                file_path, line, error_type, message = match.groups()
                errors.append(FileError(
                    file_path=file_path,
                    line_number=int(line),
                    error_message=message,
                    error_type=error_type,
                    error_category=ErrorAnalyzer._categorize_error(error_type)
                ))

            # Secondary: if no matches, try to find a final exception line like 'IndexError: message'
            if not errors:
                simple_pattern = r'(^|\n)(\w+Error|IndexError): (.+)'
                simple_match = re.search(simple_pattern, stderr)
                if simple_match:
                    error_type = simple_match.group(2)
                    message = simple_match.group(3).strip()
                    # Try to extract a file/line earlier in the traceback
                    file_line_pattern = r'File "([^"]+)", line (\d+)'
                    fl = re.search(file_line_pattern, stderr)
                    file_path = fl.group(1) if fl else "unknown"
                    line_no = int(fl.group(2)) if fl else None
                    errors.append(FileError(
                        file_path=file_path,
                        line_number=line_no,
                        error_message=message,
                        error_type=error_type,
                        error_category=ErrorAnalyzer._categorize_error(error_type)
                    ))
        
        elif language == "javascript":
            # JavaScript error patterns
            error_pattern = r'([^\s:]+):(\d+)\n.*?(\w+): (.+)'
            matches = re.finditer(error_pattern, stderr, re.MULTILINE)
            
            for match in matches:
                file_path, line, error_type, message = match.groups()
                errors.append(FileError(
                    file_path=file_path,
                    line_number=int(line),
                    error_message=message,
                    error_type=error_type,
                    error_category=ErrorAnalyzer._categorize_error(error_type)
                ))
        
        # If no specific errors found but stderr exists, create a generic error
        if not errors and stderr.strip():
            errors.append(FileError(
                file_path="unknown",
                line_number=None,
                error_message=stderr[:200],
                error_type="RuntimeError",
                error_category=ErrorCategory.UNKNOWN
            ))
        
        return errors
    
    @staticmethod
    def _categorize_error(error_type: str) -> ErrorCategory:
        """Categorize error type"""
        auto_fixable = {
            "SyntaxError", "IndentationError", "NameError",
            "TypeError", "AttributeError", "UnboundLocalError", "IndexError"
        }
        
        manual_required = {
            "ModuleNotFoundError", "ImportError", "FileNotFoundError",
            "PermissionError", "ConnectionError"
        }
        
        if error_type in auto_fixable:
            return ErrorCategory.AUTO_FIXABLE
        
analyzer = ErrorAnalyzer()

# =============================================================================
# AI-Powered Code Fixing
# =============================================================================

class AICodeFixer:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
    
    async def analyze_and_fix(
        self, 
        code: str, 
        errors: List[FileError], 
        language: str
    ) -> tuple[str, List[FilePatch]]:
        """Analyze errors and generate fixes"""
        
        if not errors:
            return "No errors detected. Code executed successfully!", []
        
        # Filter auto-fixable errors
        fixable_errors = [e for e in errors if e.error_category == ErrorCategory.AUTO_FIXABLE]
        
        if not fixable_errors:
            return "Errors require manual intervention (missing dependencies, system issues, etc.)", []
        
        # Generate fixes for each error
        patches = []
        for error in fixable_errors:
            patch = await self._generate_patch(code, error, language)
            if patch:
                patches.append(patch)
        
        analysis = self._create_analysis_summary(errors, patches)
        return analysis, patches
    
    async def _generate_patch(
        self, 
        code: str, 
        error: FileError, 
        language: str
    ) -> Optional[FilePatch]:
        """Generate a patch for a specific error"""
        
        system_prompt = f"""You are an expert {language} programmer. Your task is to fix code errors.
Given the original code and an error, provide the complete corrected code.
Be precise and only fix the specific error mentioned. Keep all other code unchanged.
Respond ONLY with the corrected code, no explanations."""

        user_prompt = f"""Original Code:
```{language}
{code}
```

Error Type: {error.error_type}
Error Message: {error.error_message}
{f"Line Number: {error.line_number}" if error.line_number else ""}

Provide the complete corrected code that fixes this error:"""

        try:
            corrected_code = await self.ollama.generate(user_prompt, system_prompt)
            
            # Clean up the response (remove markdown code blocks if present)
            corrected_code = self._extract_code_from_response(corrected_code, language)
            
            # Generate diff (simple), unified diff, and structured edits
            diff = self._generate_diff(code, corrected_code)
            unified = self._generate_unified_diff(code, corrected_code, file_path=error.file_path)
            line_edits = self._compute_line_edits(code, corrected_code)
            structured = self._create_structured_suggestions(line_edits, code, corrected_code)

            # Get explanation
            explanation = await self._generate_explanation(error, code, corrected_code, language)

            return FilePatch(
                file_path=error.file_path,
                original_content=code,
                patched_content=corrected_code,
                diff=diff,
                unified_diff=unified,
                line_edits=line_edits,
                structured_fixes=structured,
                error_fixed=f"{error.error_type}: {error.error_message}",
                fix_explanation=explanation,
                confidence=0.85
            )
            
        except Exception as e:
            print(f"Error generating patch: {e}")
            return None
    
    async def _generate_explanation(
        self, 
        error: FileError, 
        original: str, 
        fixed: str, 
        language: str
    ) -> str:
        """Generate explanation for the fix"""
        
        prompt = f"""Briefly explain what was wrong and how it was fixed:

Error: {error.error_type} - {error.error_message}
Language: {language}

Respond in 1-2 sentences."""

        try:
            explanation = await self.ollama.generate(prompt, temperature=0.3)
            return explanation.strip()
        except:
            return f"Fixed {error.error_type} error"
    
    @staticmethod
    def _extract_code_from_response(response: str, language: str) -> str:
        """Extract code from markdown code blocks"""
        # Try to find code blocks
        pattern = f"```{language}\\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try generic code blocks
        pattern = r"```\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return as-is if no code blocks found
        return response.strip()
    
    @staticmethod
    def _generate_diff(original: str, fixed: str) -> str:
        """Generate a simple diff representation"""
        orig_lines = original.splitlines()
        fixed_lines = fixed.splitlines()
        
        diff_lines = []
        max_len = max(len(orig_lines), len(fixed_lines))
        
        for i in range(max_len):
            orig_line = orig_lines[i] if i < len(orig_lines) else ""
            fixed_line = fixed_lines[i] if i < len(fixed_lines) else ""
            
            if orig_line != fixed_line:
                if orig_line:
                    diff_lines.append(f"- {orig_line}")
                if fixed_line:
                    diff_lines.append(f"+ {fixed_line}")
        
        return "\n".join(diff_lines) if diff_lines else "No changes"

    @staticmethod
    def _generate_unified_diff(original: str, fixed: str, file_path: str = "file") -> str:
        """Return a unified diff (unified format) between original and fixed."""
        orig_lines = original.splitlines(keepends=True)
        fixed_lines = fixed.splitlines(keepends=True)
        udiff = difflib.unified_diff(orig_lines, fixed_lines, fromfile=f"a/{file_path}", tofile=f"b/{file_path}")
        return ''.join(udiff) or "No changes"

    @staticmethod
    def _compute_line_edits(original: str, fixed: str) -> List[Dict[str, Any]]:
        """Compute a list of line-edit patches: start_line (1-based), end_line, replacement text."""
        orig_lines = original.splitlines()
        fixed_lines = fixed.splitlines()
        sm = difflib.SequenceMatcher(a=orig_lines, b=fixed_lines)
        edits: List[Dict[str, Any]] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue
            edit = {
                'op': tag,  # replace, delete, insert
                'start_line': i1 + 1,
                'end_line': i2,
                'replacement': '\n'.join(fixed_lines[j1:j2])
            }
            edits.append(edit)
        return edits

    @staticmethod
    def _create_structured_suggestions(line_edits: List[Dict[str, Any]], original: str, fixed: str) -> List[Dict[str, Any]]:
        """Convert line edits into higher-level structured suggestions."""
        suggestions: List[Dict[str, Any]] = []
        for e in line_edits:
            description = ''
            if e['op'] == 'replace':
                description = f"Replace lines {e['start_line']}-{e['end_line']} with provided replacement."
            elif e['op'] == 'delete':
                description = f"Delete lines {e['start_line']}-{e['end_line']}."
            elif e['op'] == 'insert':
                description = f"Insert at line {e['start_line']}: add provided lines."

            suggestions.append({
                'description': description,
                'start_line': e['start_line'],
                'end_line': e['end_line'],
                'replacement': e['replacement']
            })
        return suggestions
    
    @staticmethod
    def _create_analysis_summary(errors: List[FileError], patches: List[FilePatch]) -> str:
        """Create analysis summary"""
        summary = []
        summary.append(f"🔍 Found {len(errors)} error(s)")
        
        auto_fixable = sum(1 for e in errors if e.error_category == ErrorCategory.AUTO_FIXABLE)
        manual = sum(1 for e in errors if e.error_category == ErrorCategory.MANUAL_REQUIRED)
        
        if auto_fixable:
            summary.append(f"✅ {auto_fixable} can be auto-fixed")
        if manual:
            summary.append(f"⚠️ {manual} require manual intervention")
        
        if patches:
            summary.append(f"\n💡 Generated {len(patches)} patch(es)")
        
        return "\n".join(summary)

ai_fixer = AICodeFixer(ollama)

# =============================================================================
# Continuous Fixing Logic
# =============================================================================

class ContinuousCodeFixer:
    """Iteratively fixes code by applying patches and re-checking for errors"""
    
    async def fix_continuously(
        self,
        code: str,
        language: str,
        file_path: str,
        max_iterations: int = 5
    ) -> tuple[str, List[LogEvent], List[FilePatch]]:
        """Continuously fix code up to max_iterations times"""
        
        logs: List[LogEvent] = []
        all_patches: List[FilePatch] = []
        current_code = code
        iteration = 0
        
        def add_log(level: str, message: str, details: Optional[Dict[str, Any]] = None):
            logs.append(LogEvent(
                timestamp=datetime.now().isoformat(),
                level=level,
                message=message,
                details=details
            ))
        
        add_log("info", f"Starting continuous fix process (max {max_iterations} iterations)", {"language": language, "file_path": file_path})
        
        while iteration < max_iterations:
            iteration += 1
            add_log("info", f"Iteration {iteration}/{max_iterations}: Executing code...")
            
            # Execute current code
            exec_result = await sandbox.execute_code(current_code, language, file_path)
            
            add_log(
                "info" if exec_result.success else "warning",
                f"Code execution {'succeeded' if exec_result.success else 'failed'}",
                {"exit_code": exec_result.exit_code, "execution_time": exec_result.execution_time}
            )
            
            # If execution succeeded, we're done
            if exec_result.success:
                add_log("success", "✅ Code is now fixed! All iterations passed.", {})
                return current_code, logs, all_patches
            
            # Parse errors
            errors = analyzer.parse_errors(exec_result.stderr, language)
            add_log("warning", f"Found {len(errors)} error(s)", {"errors": [e.dict() for e in errors]})
            
            # Check if errors are auto-fixable
            fixable_errors = [e for e in errors if e.error_category == ErrorCategory.AUTO_FIXABLE]
            
            if not fixable_errors:
                add_log("error", "⚠️ Remaining errors require manual intervention", {"stderr": exec_result.stderr})
                return current_code, logs, all_patches
            
            # Generate patches for fixable errors
            patches = []
            for error in fixable_errors:
                add_log("info", f"Generating fix for {error.error_type}...", {"line": error.line_number})
                patch = await ai_fixer._generate_patch(current_code, error, language)
                
                if patch:
                    patches.append(patch)
                    all_patches.append(patch)
                    add_log(
                        "success",
                        f"✨ Generated patch: {patch.fix_explanation}",
                        {"confidence": patch.confidence}
                    )
                else:
                    add_log("error", f"Failed to generate patch for {error.error_type}", {})
            
            # Apply patches
            if patches:
                add_log("info", f"Applying {len(patches)} patch(es)...", {})
                # Apply the first patch (or merge if multiple)
                current_code = patches[0].patched_content
                add_log("success", "Patches applied. Retrying execution...", {})
            else:
                add_log("error", "No patches could be generated. Stopping.", {})
                return current_code, logs, all_patches
        
        add_log("warning", f"Reached maximum iterations ({max_iterations}). Stopping.", {})
        return current_code, logs, all_patches

continuous_fixer = ContinuousCodeFixer()

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "CodeIDE AI Agent API",
        "status": "running",
        "ollama_connected": await ollama.check_connection()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = await ollama.check_connection()
    return {
        "status": "healthy" if ollama_status else "degraded",
        "ollama": "connected" if ollama_status else "disconnected",
        "message": "Ollama is required for AI features" if not ollama_status else "All systems operational"
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_code(request: AnalysisRequest):
    """Analyze code, execute it, detect errors, and generate patches"""
    
    # Check Ollama connection
    if not await ollama.check_connection():
        raise HTTPException(
            status_code=503, 
            detail="Ollama is not running. Please start Ollama: 'ollama serve'"
        )
    
    # Generate session ID
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Execute code in sandbox
    execution_result = await sandbox.execute_code(
        request.code,
        request.language,
        request.file_path
    )
    
    # Parse errors if execution failed
    errors = []
    if not execution_result.success:
        errors = analyzer.parse_errors(execution_result.stderr, request.language)
    
    # Generate patches using AI
    analysis, patches = await ai_fixer.analyze_and_fix(
        request.code,
        errors,
        request.language
    )
    
    # Store session
    sessions[session_id] = {
        "code": request.code,
        "language": request.language,
        "file_path": request.file_path,
        "execution_result": execution_result,
        "errors": errors,
        "patches": patches,
        "analysis": analysis,
        "created_at": datetime.now().isoformat()
    }
    
    return AnalysisResponse(
        session_id=session_id,
        execution_result=execution_result,
        errors=errors,
        patches=patches,
        analysis=analysis,
        can_auto_fix=len(patches) > 0
    )


@app.post("/api/analyze-stream")
async def analyze_code_stream(request: AnalysisRequest):
    """Streamed analyze endpoint: yields newline-delimited JSON events until completion."""

    async def event_stream():
        # Notify start
        yield (json.dumps({"event": "started", "timestamp": datetime.utcnow().isoformat()}) + "\n")

        # Execute code in sandbox
        execution_result = await sandbox.execute_code(
            request.code,
            request.language,
            request.file_path
        )

        yield (json.dumps({"event": "execution_result", "data": execution_result.dict()}) + "\n")

        # Parse errors
        errors = []
        if not execution_result.success:
            errors = analyzer.parse_errors(execution_result.stderr, request.language)
            yield (json.dumps({"event": "errors_parsed", "data": [e.dict() for e in errors]}) + "\n")

        # If no errors, continue but report and finish
        if not errors:
            analysis = "No errors detected. Code executed successfully!"
            patches = []
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            sessions[session_id] = {
                "code": request.code,
                "language": request.language,
                "file_path": request.file_path,
                "execution_result": execution_result,
                "errors": errors,
                "patches": patches,
                "analysis": analysis,
                "created_at": datetime.now().isoformat()
            }
            yield (json.dumps({"event": "analysis_complete", "data": {"session_id": session_id, "analysis": analysis, "patches": []}}) + "\n")
            return

        # Ensure model is available (may pull via CLI)
        yield (json.dumps({"event": "model_check", "model": ollama.model}) + "\n")
        model_ok = await ollama.ensure_model_available()
        yield (json.dumps({"event": "model_available", "model": ollama.model, "available": model_ok}) + "\n")

        # Generate patches
        ai_analysis, patches = await ai_fixer.analyze_and_fix(request.code, errors, request.language)
        yield (json.dumps({"event": "ai_analysis", "analysis": ai_analysis}) + "\n")

        # Send patches one by one
        patches_data = []
        for p in patches:
            # p is a FilePatch pydantic model
            patches_data.append(p.dict())
            yield (json.dumps({"event": "patch_candidate", "patch": p.dict()}) + "\n")

        # Store session
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        sessions[session_id] = {
            "code": request.code,
            "language": request.language,
            "file_path": request.file_path,
            "execution_result": execution_result,
            "errors": errors,
            "patches": patches,
            "analysis": ai_analysis,
            "created_at": datetime.now().isoformat()
        }

        yield (json.dumps({"event": "complete", "session_id": session_id, "patch_count": len(patches)}) + "\n")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")

@app.post("/api/fix-continuously")
async def fix_code_continuously(request: ContinuousFixRequest):
    """
    Continuously fix code using AI:
    1. Execute the code
    2. If errors exist, generate patches using LLM
    3. Apply patches and re-execute
    4. Repeat until code passes or max iterations reached
    Returns streamed log events showing the entire fixing process.
    """
    
    async def event_stream():
        yield (json.dumps({"event": "started", "timestamp": datetime.utcnow().isoformat()}) + "\n")
        
        # Check Ollama connection
        if not await ollama.check_connection():
            yield (json.dumps({"event": "error", "message": "Ollama is not running"}) + "\n")
            return
        
        # Run continuous fixer
        fixed_code, logs, patches = await continuous_fixer.fix_continuously(
            request.code,
            request.language,
            request.file_path,
            request.max_iterations
        )
        
        # Stream each log event
        for log in logs:
            yield (json.dumps({"event": "log", "data": log.dict()}) + "\n")
        
        # Stream patches
        for i, patch in enumerate(patches):
            yield (json.dumps({"event": "patch", "index": i, "data": patch.dict()}) + "\n")
        
        # Create session for later reference
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        sessions[session_id] = {
            "code": request.code,
            "language": request.language,
            "file_path": request.file_path,
            "fixed_code": fixed_code,
            "logs": [log.dict() for log in logs],
            "patches": [patch.dict() for patch in patches],
            "created_at": datetime.now().isoformat()
        }
        
        yield (json.dumps({"event": "complete", "session_id": session_id, "fixed_code": fixed_code, "total_logs": len(logs)}) + "\n")
    
    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
async def apply_patch(request: ApplyPatchRequest):
    """Apply a specific patch to the code"""
    
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    patches = session["patches"]
    
    if request.patch_index < 0 or request.patch_index >= len(patches):
        raise HTTPException(status_code=400, detail="Invalid patch index")
    
    patch = patches[request.patch_index]
    
    # Mark patch as applied
    if "applied_patches" not in session:
        session["applied_patches"] = []
    session["applied_patches"].append(request.patch_index)
    
    return ApplyPatchResponse(
        success=True,
        message=f"Patch applied successfully: {patch.fix_explanation}",
        updated_code=patch.patched_content
    )

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup resources"""
    if session_id in sessions:
        sandbox.cleanup(session_id)
        del sessions[session_id]
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/models")
async def list_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return {
                    "models": [{"name": m["name"], "size": m.get("size", 0)} for m in models],
                    "current": DEFAULT_MODEL
                }
    except:
        pass
    return {"models": [], "current": DEFAULT_MODEL, "error": "Could not fetch models"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting CodeIDE AI Agent API Server")
    print("📍 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🤖 Ollama: http://localhost:11434")
    print("\nMake sure Ollama is running: ollama serve")
    uvicorn.run(app, host="0.0.0.0", port=8000)