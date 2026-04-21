"""
DeepSeek self-verification for round-3.

For each (question, answer) pair, DeepSeek performs a full web search,
produces a verification checklist with citations, and optionally revises
the answer. Results are appended to an output JSONL file.

Reads input from either a local JSONL file or a HuggingFace dataset.
The input records must contain: _id, question (or inverted_question), answer.

Works on macOS and Linux.

Usage — local JSONL:
    python deepseek_self_verify.py \\
        --input_file outputs/round-2/responses_test.jsonl \\
        --output_file outputs/round-3/verified.jsonl

Usage — HuggingFace dataset:
    python deepseek_self_verify.py \\
        --hf_dataset orbit-ai/orbit-stage-1-44k \\
        --output_file outputs/round-3/verified.jsonl

Optional flags:
    --topic              Filter HF dataset by topic field
    --max_chat_length 30 Restart browser after N items (default 30)
    --profile_path       Chrome profile directory
    --debug_wait         Print polling status while waiting for replies
"""

import argparse
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException
from tqdm import tqdm
import undetected_chromedriver as uc


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """
Question: {question}
Answer: {answer}

Given the question and answer, conduct a full web and Wikipedia search to retrieve the pages necessary to answer the question.
Next, come up with a verification list of criteria, citing each URL for each criterion.
Finally, provide a revised answer to the question (a short string) if needed, and list all the sources you cite with each URL and which verification statement it supports.

Please do the web search first!
"""


# ---------------------------------------------------------------------------
# Clipboard helper (cross-platform)
# ---------------------------------------------------------------------------

def clipboard_write(text: str) -> None:
    """Write text to the system clipboard to enable fast paste instead of send_keys."""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        elif system == "Linux":
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)
        elif system == "Windows":
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Console]::InputEncoding=[Text.Encoding]::UTF8; $input|Set-Clipboard"],
                input=text.encode("utf-8"), check=True,
            )
    except Exception as exc:
        print(f"[clipboard_write] warning: {exc}")


def default_chrome_profile_path() -> str:
    if platform.system() == "Darwin":
        return str(Path.home() / "Library" / "Application Support" / "Google" / "Chrome" / "Default")
    return str(Path.home() / ".config" / "google-chrome" / "Default")


# ---------------------------------------------------------------------------
# DeepSeek bot
# ---------------------------------------------------------------------------

class DeepSeekChatBot:
    """
    Selenium-based bot for DeepSeek Chat.
    Handles login, Expert mode, DeepThink + Smart Search toggles,
    prompt submission via clipboard paste, and structured response extraction.
    Works on macOS and Linux.
    """

    APP_URL = "https://chat.deepseek.com/"

    def __init__(self, user_data_path: str | None = None, debug_wait: bool = False):
        self.user_data_path = user_data_path or default_chrome_profile_path()
        self.debug_wait = debug_wait
        self.driver = uc.Chrome(
            user_data_dir=self.user_data_path,
            use_subprocess=False,
            headless=False,
        )
        self.is_our_first_chat = True
        self.login()
        self._ensure_expert_mode()
        self._ensure_deepthink_and_search()

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    def login(self) -> None:
        self.driver.get(self.APP_URL)
        try:
            WebDriverWait(self.driver, 5).until(EC.url_to_be(self.APP_URL))
        except TimeoutException:
            print("Please log in manually within 10 minutes...")
            WebDriverWait(self.driver, 600).until(EC.url_to_be(self.APP_URL))
        print("Login succeeded.")

    # ------------------------------------------------------------------
    # Expert mode
    # ------------------------------------------------------------------

    def _expert_mode_selected(self) -> bool:
        try:
            return bool(
                self.driver.execute_script(
                    """
                    const el = document.querySelector('[data-model-type="expert"]');
                    if (!el) return false;
                    let p = el;
                    for (let i = 0; i < 12 && p; i++, p = p.parentElement) {
                        const st = p.getAttribute('style') || '';
                        if (!st.includes('--selected-index')) continue;
                        const items = Array.from(p.querySelectorAll('[data-model-type]'));
                        const expertIdx = items.findIndex(
                            (e) => e.getAttribute('data-model-type') === 'expert'
                        );
                        const m = st.match(/--selected-index:\\s*(\\d+)/);
                        if (!m || expertIdx < 0) return false;
                        return parseInt(m[1], 10) === expertIdx;
                    }
                    const cls = (el.getAttribute('class') || '').trim().split(/\\s+/).filter(Boolean);
                    return cls.length >= 3;
                    """
                )
            )
        except Exception:
            return False

    def _ensure_expert_mode(self, timeout: float = 25.0) -> None:
        try:
            pill = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-model-type="expert"]'))
            )
        except TimeoutException:
            print("Expert mode control not found; continuing without switching.")
            return
        if self._expert_mode_selected():
            return
        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", pill)
        time.sleep(0.15)
        try:
            pill.click()
        except ElementNotInteractableException:
            self.driver.execute_script("arguments[0].click();", pill)
        time.sleep(0.35)

    # ------------------------------------------------------------------
    # DeepThink + Smart Search toggles
    # ------------------------------------------------------------------

    @staticmethod
    def _toggle_is_on(btn) -> bool:
        if btn is None:
            return False
        cls = btn.get_attribute("class") or ""
        if any(marker in cls for marker in ("--selected", "--active", "--checked", "--on")):
            return True
        if (btn.get_attribute("aria-pressed") or "").lower() == "true":
            return True
        if (btn.get_attribute("aria-checked") or "").lower() == "true":
            return True
        if (btn.get_attribute("data-state") or "").lower() in ("on", "checked", "active", "selected"):
            return True
        return False

    def _find_toggle(self, text_substrings: tuple[str, ...], transform_origin: str | None, timeout: float = 8.0):
        end = time.time() + timeout
        while time.time() < end:
            for sub in text_substrings:
                if '"' in sub or "'" in sub:
                    continue
                xpath = (
                    "//div[@role='button' and contains(@class,'ds-toggle-button')]"
                    f"[.//span[contains(normalize-space(.), '{sub}')]]"
                )
                for el in self.driver.find_elements(By.XPATH, xpath):
                    try:
                        if el.is_displayed():
                            return el
                    except Exception:
                        continue
            time.sleep(0.25)
        if transform_origin:
            sel = f"div.ds-floating-position-wrapper.ds-theme[data-transform-origin='{transform_origin}'] > button"
            try:
                el = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                if el.is_displayed():
                    return el
            except TimeoutException:
                pass
        return None

    def _ensure_toggle_on(self, btn, label: str) -> None:
        if btn is None:
            print(f"  [{label}] toggle not found; continuing.")
            return
        if self._toggle_is_on(btn):
            print(f"  [{label}] already on.")
            return
        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
        time.sleep(0.12)
        try:
            btn.click()
        except ElementNotInteractableException:
            self.driver.execute_script("arguments[0].click();", btn)
        time.sleep(0.35)
        if self._toggle_is_on(btn):
            print(f"  [{label}] turned on.")
        else:
            print(f"  [{label}] WARNING: clicked but still appears off — check UI manually.")

    def _ensure_deepthink_and_search(self, timeout: float = 25.0) -> None:
        try:
            self._get_chat_input(timeout=min(timeout, 30.0))
        except TimeoutException:
            print("Chat input not ready; skipping DeepThink/Search toggles.")
            return
        deep = self._find_toggle(("DeepThink", "深度思考"), "right", timeout=min(12.0, timeout))
        search = self._find_toggle(("Smart Search", "联网搜索", "搜索", "Search"), "left", timeout=min(12.0, timeout))
        self._ensure_toggle_on(deep, "DeepThink")
        self._ensure_toggle_on(search, "Smart Search")

    # ------------------------------------------------------------------
    # Chat input helpers
    # ------------------------------------------------------------------

    def _get_chat_input(self, timeout: float = 30.0):
        selectors = [
            (By.ID, "chat-input"),
            (By.CSS_SELECTOR, "textarea"),
            (By.CSS_SELECTOR, ".ds-scroll-area.ds-textarea textarea"),
            (By.CSS_SELECTOR, "[contenteditable='true']"),
        ]
        end = time.time() + timeout
        while time.time() < end:
            for by, sel in selectors:
                try:
                    for el in self.driver.find_elements(by, sel):
                        if el.is_displayed() and el.is_enabled():
                            return el
                except Exception:
                    continue
            time.sleep(0.2)
        raise TimeoutException(f"Chat input not found within {timeout}s")

    def _clear_chat_input(self, elem) -> None:
        try:
            elem.clear()
        except Exception:
            pass
        try:
            elem.click()
            cmd = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL
            elem.send_keys(cmd, "a")
            elem.send_keys(Keys.BACKSPACE)
        except Exception:
            try:
                self.driver.execute_script(
                    """
                    const el = arguments[0];
                    if (el.tagName && el.tagName.toLowerCase() === 'textarea') {
                        el.value = '';
                        el.dispatchEvent(new Event('input', {bubbles: true}));
                    } else { el.textContent = ''; }
                    """,
                    elem,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_new_chat(self, timeout: float = 25.0) -> None:
        """Open a fresh conversation and re-apply Expert + DeepThink + Search."""
        try:
            btn = self.driver.find_element(
                By.XPATH, '//*[text()="New chat" or text()="开启新对话"]'
            )
            self.driver.execute_script("arguments[0].scrollIntoView(true);", btn)
            try:
                btn.click()
            except ElementNotInteractableException:
                self.driver.execute_script("arguments[0].click();", btn)
        except Exception as exc:
            print(f"start_new_chat: could not click New chat button ({exc}); continuing.")
        time.sleep(0.8)
        self.is_our_first_chat = True
        try:
            self._get_chat_input(timeout=timeout)
        except TimeoutException:
            print("start_new_chat: chat input not visible yet; continuing.")
            return
        self._ensure_expert_mode()
        self._ensure_deepthink_and_search()

    def restart(self) -> None:
        """Quit and re-launch the browser, then log in again."""
        try:
            self.driver.quit()
        except Exception:
            pass
        time.sleep(2)
        self.__init__(user_data_path=self.user_data_path, debug_wait=self.debug_wait)

    # ------------------------------------------------------------------
    # Generation detection
    # ------------------------------------------------------------------

    def _is_generation_active(self) -> bool:
        xpaths = [
            "//*[self::button or @role='button'][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stop')]",
            "//*[self::button or @role='button'][contains(normalize-space(.), '停止')]",
            "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'searching')]",
            "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'reasoning')]",
            "//*[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'thinking') and not(contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'thought for'))]",
            "//*[contains(normalize-space(.), '搜索中') or contains(normalize-space(.), '思考中') or contains(normalize-space(.), '推理中')]",
        ]
        for xp in xpaths:
            try:
                for el in self.driver.find_elements(By.XPATH, xp):
                    try:
                        if el.is_displayed():
                            return True
                    except Exception:
                        continue
            except Exception:
                continue
        return False

    @staticmethod
    def _ds_markdown_count(driver) -> int:
        return len(driver.find_elements(By.CLASS_NAME, "ds-markdown"))

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _extract_response_parts(self, html: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        for span in soup.find_all("span", class_="ds-markdown-cite"):
            txt = span.get_text(strip=True)
            if txt.isdigit():
                span.string = f"[{txt}]"

        thought_summary = ""
        thought_match = soup.find(
            string=lambda s: isinstance(s, str) and ("Thought for" in s or "Thinking" in s)
        )
        if thought_match:
            thought_summary = thought_match.strip()

        urls = list(dict.fromkeys(
            (a.get("href") or "").strip()
            for a in soup.find_all("a", href=True)
            if (a.get("href") or "").startswith(("http://", "https://"))
        ))

        md_nodes = soup.select(".ds-markdown")
        if not md_nodes:
            full_md = md(str(soup)).strip()
            return {"full_markdown": full_md, "reasoning_markdown": "", "final_output_markdown": full_md, "web_urls": urls, "thought_summary": thought_summary}

        node_mds = [md(str(n)).strip() for n in md_nodes if md(str(n)).strip()]
        if not node_mds:
            full_md = md(str(soup)).strip()
            return {"full_markdown": full_md, "reasoning_markdown": "", "final_output_markdown": full_md, "web_urls": urls, "thought_summary": thought_summary}

        return {
            "full_markdown": "\n\n".join(node_mds).strip(),
            "reasoning_markdown": "\n\n".join(node_mds[:-1]).strip(),
            "final_output_markdown": node_mds[-1],
            "web_urls": urls,
            "thought_summary": thought_summary,
        }

    # ------------------------------------------------------------------
    # Prompt submission
    # ------------------------------------------------------------------

    def send_prompt(self, prompt: str) -> dict:
        if self.is_our_first_chat:
            num_history_replies = 0
        else:
            n = self._ds_markdown_count(self.driver)
            num_history_replies = 0 if n == 0 else n
            if n == 0:
                self.is_our_first_chat = True

        if num_history_replies == 0:
            self._ensure_expert_mode()
            self._ensure_deepthink_and_search()

        time.sleep(0.5)
        prompt_input = self._get_chat_input(timeout=50)
        self._clear_chat_input(prompt_input)

        clipboard_write(prompt)
        cmd = Keys.COMMAND if platform.system() == "Darwin" else Keys.CONTROL
        time.sleep(0.15)
        prompt_input.send_keys(cmd, "v")
        time.sleep(0.4)
        prompt_input.send_keys(Keys.ENTER)

        return self._get_latest_reply(num_history_replies)

    def _get_latest_reply(self, num_history_replies: int) -> dict:
        first_seen_ts = None
        for _ in range(600):
            if self._ds_markdown_count(self.driver) > num_history_replies:
                first_seen_ts = time.time()
                break
            time.sleep(0.5)
        if first_seen_ts is None:
            raise TimeoutException("No reply arrived in time.")

        prev = ""
        last_change_ts = time.time()
        stable = idle_cycles = 0
        stable_needed, idle_needed = 40, 8
        min_elapsed, forced_stagnation_seconds = 15.0, 40.0

        for poll_idx in range(600):
            replies = self.driver.find_elements(By.CLASS_NAME, "ds-markdown")
            if len(replies) <= num_history_replies:
                time.sleep(0.5)
                continue
            html = "".join((r.get_attribute("outerHTML") or "") for r in replies[num_history_replies:])
            generation_active = self._is_generation_active()

            if self.debug_wait and poll_idx % 5 == 0:
                print(
                    f"[debug] poll={poll_idx} stable={stable}/{stable_needed} "
                    f"idle={idle_cycles}/{idle_needed} active={generation_active} "
                    f"elapsed={time.time()-first_seen_ts:.1f}s stagnant={time.time()-last_change_ts:.1f}s"
                )

            if html == prev:
                stable += 1
                stagnant_for = time.time() - last_change_ts
                if stable >= stable_needed:
                    elapsed = time.time() - first_seen_ts
                    parts = self._extract_response_parts(html)
                    if parts["full_markdown"].endswith(("▍", "...")):
                        stable = idle_cycles = 0
                        time.sleep(0.5)
                        continue
                    if generation_active:
                        if stagnant_for >= forced_stagnation_seconds and elapsed >= min_elapsed:
                            self.is_our_first_chat = False
                            return parts
                        idle_cycles = 0
                        time.sleep(0.5)
                        continue
                    idle_cycles += 1
                    if elapsed >= min_elapsed and idle_cycles >= idle_needed:
                        self.is_our_first_chat = False
                        return parts
            else:
                stable = idle_cycles = 0
                last_change_ts = time.time()
            prev = html
            time.sleep(0.5)

        raise TimeoutException("Reply never stabilized.")

    def close(self) -> None:
        self.driver.quit()


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_qa_from_file(input_file: str) -> dict[str, dict]:
    """Load QA pairs from a local JSONL file. Returns {_id: {question, answer}}."""
    items: dict[str, dict] = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get("inverted_question") or obj.get("question", "")
            answer = obj.get("answer", "")
            if question and answer:
                items[obj["_id"]] = {"question": question, "answer": answer}
    return items


def load_qa_from_hf(repo_id: str, topic: str | None = None) -> dict[str, dict]:
    """Load QA pairs from a HuggingFace dataset. Returns {_id: {question, answer}}."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, split="train")
    if topic:
        ds = ds.filter(lambda x: x.get("topic") == topic)
    items: dict[str, dict] = {}
    for row in ds:
        question = row.get("inverted_question") or row.get("question", "")
        answer = row.get("answer", "")
        if question and answer:
            items[row["_id"]] = {"question": question, "answer": answer}
    return items


def load_completed(output_file: str) -> set[str]:
    completed: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("_id"):
                    completed.add(obj["_id"])
    return completed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek self-verification of QA pairs")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_file", type=str, help="Local JSONL file with {_id, question/inverted_question, answer}")
    source.add_argument("--hf_dataset", type=str, help="HuggingFace dataset repo ID (e.g. orbit-ai/orbit-20k)")

    parser.add_argument("--topic", type=str, default=None, help="Filter HF dataset by topic field")
    parser.add_argument("--output_file", type=str, required=True, help="JSONL file to append verified results to")
    parser.add_argument("--profile_path", type=str, default=None, help="Chrome profile directory (persists login)")
    parser.add_argument("--max_chat_length", type=int, default=30, help="Restart browser after this many items (default 30)")
    parser.add_argument("--debug_wait", action="store_true", help="Print polling status while waiting for replies")
    args = parser.parse_args()

    # Load QA pairs
    if args.input_file:
        print(f"Loading QA pairs from file: {args.input_file}")
        items = load_qa_from_file(args.input_file)
    else:
        print(f"Loading QA pairs from HuggingFace: {args.hf_dataset}" + (f" (topic={args.topic})" if args.topic else ""))
        items = load_qa_from_hf(args.hf_dataset, args.topic)

    print(f"Total QA pairs: {len(items):,}")

    completed = load_completed(args.output_file)
    print(f"Already completed: {len(completed):,}")

    items = {k: v for k, v in items.items() if k not in completed}
    print(f"Remaining: {len(items):,}")
    if not items:
        print("Nothing to do.")
        raise SystemExit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    bot = DeepSeekChatBot(
        user_data_path=args.profile_path,
        debug_wait=args.debug_wait,
    )
    time.sleep(5)

    count = 0
    for _id, qa in tqdm(items.items(), desc="Verifying QA"):
        count += 1
        question, answer = qa["question"], qa["answer"]

        if count % args.max_chat_length == 0 and count >= args.max_chat_length:
            bot.restart()

        if count > 1:
            bot.start_new_chat()

        prompt = PROMPT.format(question=question, answer=answer)
        try:
            parts = bot.send_prompt(prompt)

            if "the server is busy" in parts["full_markdown"].lower():
                print(f"Server busy; retrying {_id}...")
                bot.driver.refresh()
                time.sleep(10)
                parts = bot.send_prompt(prompt)
                if "server is busy" in parts["full_markdown"].lower():
                    print(f"Still busy; skipping {_id}.")
                    continue

            with open(args.output_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps({
                    "_id": _id,
                    "question": question,
                    "answer": answer,
                    "response": parts["full_markdown"],
                    "reasoning": parts["reasoning_markdown"]
                }, ensure_ascii=False) + "\n")

            time.sleep(3)

        except Exception as exc:
            print(f"Error on {_id}: {exc!r}")
            traceback.print_exc()
            try:
                bot.driver.refresh()
                time.sleep(5)
                parts = bot.send_prompt(prompt)
                with open(args.output_file, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps({
                        "_id": _id,
                        "question": question,
                        "answer": answer,
                        "response": parts["final_output_markdown"],
                        "reasoning": parts["reasoning_markdown"],
                    }, ensure_ascii=False) + "\n")
            except Exception as retry_exc:
                print(f"Retry failed for {_id}: {retry_exc!r}")

    bot.close()
