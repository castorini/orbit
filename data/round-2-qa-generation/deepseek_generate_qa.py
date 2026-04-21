"""
DeepSeek chatbot for round-2 QA generation.

Reads seeds from either a local JSONL file or the HuggingFace dataset
orbit-ai/orbit-seeds, sends each seed to DeepSeek Chat (with DeepThink +
Smart Search enabled), and appends the structured response to an output
JSONL file.

Works on macOS and Linux.

Usage — local JSONL:
    python deepseek_chatbot.py \\
        --entity_file ../round-1-seeds/entities/test.jsonl \\
        --output_file output/responses.jsonl

Usage — HuggingFace dataset:
    python deepseek_chatbot.py \\
        --hf_dataset orbit-ai/orbit-seeds \\
        --domain mathematics \\
        --output_file output/responses.jsonl

Optional flags:
    --max_chat_length 30      Restart browser after N seeds (default 30)
    --profile_path ~/.cache/deepseek_profile  Chrome profile dir
    --debug_wait              Print polling status while waiting for replies
"""

import argparse
import json
import os
import platform
import re
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
# Response parser
# ---------------------------------------------------------------------------

def parse_xml_fields(response: str) -> dict:
    """Extract inverted_question and answer from the XML block in a response."""
    def _tag(name: str) -> str | None:
        m = re.search(rf"<{name}>([\s\S]*?)</{name}>", response, re.IGNORECASE)
        return m.group(1).strip() if m else None
    return {"question": _tag("inverted_question"), "answer": _tag("answer")}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """
You are an expert multi-hop factoid question creator. You should create complex or inverted questions containing answers that are easy to verify, but hard to solve for a given seed along with the answer, verification checklist and list of evidence urls sufficient to justify the answer!

## task
<task>
The question-answer pair generated should satisfy the conditions below:
- The question should have an unique answer.
- The answer should be a short fact, either a term or a short phrase.
- The question should be very difficult to answer without the list of correct evidence urls, urging the student to search the web multiple times.
- The model answer should be easily verifiable given the list of evidence urls, and should ultimately be a substring contained in the final evidence.
- The question should be phrased so that each evidence does not give away too much information about the model answer so that the student can determine the model answer without finding all the evidences you've listed. That is, each evidence should be absolutely necessary to answer the question correctly.
- The question should be deep enough that it requires at least 5 evidence to answer the question correctly.
- You should not mention any proper nouns in the question as it gives away too much information about the model answer. Instead, describe the proper noun's properties in a way that is still enough to uniquely identify the proper noun.
</task>

## procedure
<procedure>
You would typically start with a "seed" (could be a person, event, or artifact) that the user will provide. Find several characteristics within a large search space using the "seed", and create a question from them.
</procedure>

## Required output
Please make sure you look into the seed carefully and think first before you provide your output in XML format.
Make sure the "seed" is actually hidden in the question to make it challenging and your question should not be too verbose or easy to answer.
Follow the format below by providing a verified answer to the inverted question, a verification checklist containing citations for each verification, and a list of evidence URLs:

<output>
    <inverted_question>your inverted query</inverted_question>
    <answer>the answer to the inverted query</answer>
    <verification_checklist>
        <item>bullet point 1 :cite[1]</item>
        <item>bullet point 2 :cite[1]</item>
        <item>bullet point 3 :cite[7]</item>
        etc.
    </verification_checklist>
    <evidence_urls>[1]: https://xxxx..., [2]: https://yyy..., [7]: https://zzzz...</evidence_urls>
</output>

## Example
<example>
<seed>King Crimson</seed>
<inverted_question>What is the title of a song from the self-titled 1970 album by a band whose lead singer, who was also an amateur boxer, performed on a 1971 album produced by the guitarist of a band whose 1970 album featured a pianist who also led a 50-piece jazz orchestra?</inverted_question>
<answer>The Man</answer>
<verification_checklist>
    <item>King Crimson - Lizard (album review) - Sputnikmusic: This evidence establishes that Keith Tippett was a guest pianist on King Crimson's 1970 album "Lizard" :cite[1]</item>
    <item>Centipede (band) - Wikipedia: This source connects Keith Tippett to the 50-piece jazz orchestra, Centipede, and their album "Septober Energy," produced by Robert Fripp, the guitarist for King Crimson :cite[2]</item>
    <item>Centipede: Septober Energy - Cherry Red Records: Lists Mike Patto as one of the vocalists :cite[3]</item>
    <item>PATTO discography - Prog Archives: Confirms Mike Patto was the lead singer of Patto, with a self-titled 1970 album :cite[4]</item>
    <item>Mike Patto: Confirms he was an amateur boxer and that "The Man" is a song on Patto's debut album :cite[5]</item>
</verification_checklist>
<evidence_urls>[1]: https://www.sputnikmusic.com/review/88829/King-Crimson-Lizard/, [2]: https://en.wikipedia.org/wiki/Centipede_(band), [3]: https://www.cherryred.co.uk/centipede-septober-energy-remastered-2cd-edition, [4]: https://www.progarchives.com/artist.asp?id=5116, [5]: https://www.pattofan.com/MikePatto/mike_patto.htm</evidence_urls>
</example>

NOTE: Make sure the "seed" is hidden in the question to make it challenging and do not use the "seed" as your answer! To ensure your question is difficult enough, try to answer it yourself from scratch using fewer than 5 searches — if you can, revise and try again.
<seed>{seed}</seed>
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
        # Cover all known active-state class names across DeepSeek UI versions
        if any(marker in cls for marker in ("--selected", "--active", "--checked", "--on")):
            return True
        # aria-pressed / aria-checked
        if (btn.get_attribute("aria-pressed") or "").lower() == "true":
            return True
        if (btn.get_attribute("aria-checked") or "").lower() == "true":
            return True
        # data-state used in some headless-UI based builds
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
# Seed loading
# ---------------------------------------------------------------------------

FILTER_OUT = ["Portal:", "Category:", "Template:", "File:", "Help:", "Wikipedia:"]


def load_seeds_from_file(entity_file: str) -> dict[str, str]:
    """Load seeds from a local JSONL file. Supports both `seed-url` and `seed_url`."""
    seeds: dict[str, str] = {}
    with open(entity_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seed = obj.get("seed", "")
            if seed and not any(excl in seed for excl in FILTER_OUT):
                seeds[obj["_id"]] = seed
    return seeds


def load_seeds_from_hf(repo_id: str, domain: str) -> dict[str, str]:
    """Load seeds from a HuggingFace dataset config (domain)."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, domain, split="train")
    seeds: dict[str, str] = {}
    for row in ds:
        seed = row.get("seed", "")
        if seed and not any(excl in seed for excl in FILTER_OUT):
            seeds[row["_id"]] = seed
    return seeds


def load_completed(output_file: str) -> set[str]:
    completed: set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("_id")
                if cid:
                    completed.add(cid)
    return completed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek QA generation from seeds")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--entity_file", type=str, help="Local JSONL seed file ({_id, seed, ...} per line)")
    source.add_argument("--hf_dataset", type=str, help="HuggingFace dataset repo ID (e.g. orbit-ai/orbit-seeds)")

    parser.add_argument("--domain", type=str, default=None, help="Domain/config to load when using --hf_dataset (e.g. mathematics)")
    parser.add_argument("--output_file", type=str, required=True, help="JSONL file to append responses to")
    parser.add_argument("--profile_path", type=str, default=None, help="Chrome profile directory (persists login)")
    parser.add_argument("--max_chat_length", type=int, default=30, help="Restart browser after this many seeds (default 30)")
    parser.add_argument("--debug_wait", action="store_true", help="Print polling status while waiting for replies")
    args = parser.parse_args()

    # Load seeds
    if args.entity_file:
        print(f"Loading seeds from file: {args.entity_file}")
        seeds = load_seeds_from_file(args.entity_file)
    else:
        if not args.domain:
            parser.error("--domain is required when using --hf_dataset")
        print(f"Loading seeds from HuggingFace: {args.hf_dataset} / {args.domain}")
        seeds = load_seeds_from_hf(args.hf_dataset, args.domain)

    print(f"Total seeds: {len(seeds):,}")

    completed = load_completed(args.output_file)
    print(f"Already completed: {len(completed):,}")

    seeds = {k: v for k, v in seeds.items() if k not in completed}
    print(f"Remaining: {len(seeds):,}")
    if not seeds:
        print("Nothing to do.")
        raise SystemExit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    bot = DeepSeekChatBot(
        user_data_path=args.profile_path,
        debug_wait=args.debug_wait,
    )
    time.sleep(5)

    count = 0
    for _id, seed in tqdm(seeds.items(), desc="Generating QA"):
        count += 1

        if count % args.max_chat_length == 0 and count >= args.max_chat_length:
            bot.restart()

        if count > 1:
            bot.start_new_chat()

        prompt = PROMPT.format(seed=seed)
        try:
            parts = bot.send_prompt(prompt)

            if "the server is busy" in parts["full_markdown"].lower():
                print(f"Server busy; retrying seed {_id}...")
                bot.driver.refresh()
                time.sleep(10)
                parts = bot.send_prompt(prompt)
                if "server is busy" in parts["full_markdown"].lower():
                    print(f"Still busy; skipping {_id}.")
                    continue

            with open(args.output_file, "a", encoding="utf-8") as fout:
                fout.write(json.dumps({
                    "_id": _id,
                    "seed": seed,
                    **parse_xml_fields(parts["final_output_markdown"]),
                    "response": parts["final_output_markdown"],
                    "reasoning": parts["reasoning_markdown"],
                }, ensure_ascii=False) + "\n")

            time.sleep(3)

        except Exception as exc:
            print(f"Error on seed {_id}: {exc!r}")
            traceback.print_exc()
            try:
                bot.driver.refresh()
                time.sleep(5)
                parts = bot.send_prompt(prompt)
                with open(args.output_file, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps({
                        "_id": _id,
                        "seed": seed,
                        **parse_xml_fields(parts["final_output_markdown"]),
                        "response": parts["final_output_markdown"],
                        "reasoning": parts["reasoning_markdown"],
                    }, ensure_ascii=False) + "\n")
            except Exception as retry_exc:
                print(f"Retry failed for {_id}: {retry_exc!r}")

    bot.close()
