"""
Parse four Gutenberg folklore collections into CSV datasets.

Collections:
  1. Tibetan  - "Folk tales from Tibet" by W.F. O'Connor (Gutenberg #75000)
  2. Mongolian - "Sagas from the Far East; Kalmouk and Mongolian Traditionary Tales"
                 by Rachel Harriette Busk (Gutenberg #40402)
  3. Thai/Laos - "Laos Folk-Lore of Farther India" by Katherine Neville Fleeson
                 (Gutenberg #35564) — Laos country was part of the kingdom of Siam/Thailand
  4. Shan/Burma (used as proxy for mainland SE Asia) - "Shan Folk Lore Stories
                 from the Hill and Water Country" by William Charles Griggs
                 (Gutenberg #32375)

Note on Vietnamese: No dedicated Vietnamese / Annamese folk-tale collection in English
translation exists on Project Gutenberg as of 2026. The Laos collection (35564) is
the closest mainland SE Asian English-language folklore text available. A note is
included in the output.
"""

import re
import os
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip footnote markers, illustration tags, extra whitespace."""
    # Remove [Illustration: ...] tags
    text = re.sub(r'\[Illustration:[^\]]*\]', '', text)
    # Remove footnote numeric refs like [1], [12]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove decoration lines (*** or *   *   *)
    text = re.sub(r'^\s*\*[\s\*]+\*\s*$', '', text, flags=re.MULTILINE)
    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace newlines within paragraphs with spaces
    paragraphs = re.split(r'\n\n+', text.strip())
    paras_cleaned = [' '.join(p.split()) for p in paragraphs if p.strip()]
    return ' '.join(paras_cleaned)


# ---------------------------------------------------------------------------
# 1. TIBETAN — Gutenberg #75000
#    Stories marked as:  STORY No. I.  (blank line)  TITLE IN ALL CAPS.
# ---------------------------------------------------------------------------

def parse_tibetan(filepath: str) -> list[dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Strip Gutenberg header/footer
    start = raw.find('*** START OF THE PROJECT GUTENBERG EBOOK')
    end   = raw.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    if start != -1:
        raw = raw[start:]
    if end != -1:
        raw = raw[:end]

    # Split on "STORY No. [Roman numeral]."
    # Pattern: STORY No. I.\n\nTITLE.\n\ntext...
    pattern = r'(?=^STORY No\. [IVXLC]+\.$)'
    chunks = re.split(pattern, raw, flags=re.MULTILINE)

    rows = []
    for chunk in chunks:
        if not chunk.strip().startswith('STORY'):
            continue
        lines = chunk.strip().split('\n')
        # Line 0: "STORY No. X."
        # Then blank lines, then the ALL-CAPS title
        title = ''
        text_start_idx = 1
        for i, line in enumerate(lines[1:], start=1):
            stripped = line.strip()
            if stripped and stripped == stripped.upper() and len(stripped) > 3:
                # Could be title (all caps)
                # Might span two lines for very long titles
                if not title:
                    title = stripped
                    text_start_idx = i + 1
                elif lines[i+1].strip() == lines[i+1].strip().upper() and lines[i+1].strip():
                    # continuation of title
                    title += ' ' + stripped
                else:
                    break
            elif stripped:
                if not title:
                    title = stripped  # fallback
                    text_start_idx = i
                break

        body = '\n'.join(lines[text_start_idx:])
        body_clean = clean_text(body)

        if len(body_clean) < 200:
            continue

        # Remove "THE PRINCE AND THE OGRE'S CASTLE" style noise from title
        title = title.rstrip('.')
        rows.append({
            'title': title,
            'text': body_clean,
            'region': 'tibet',
            'source': 'gutenberg_75000',
        })

    return rows


# ---------------------------------------------------------------------------
# 2. MONGOLIAN — Gutenberg #40402
#    Stories in first saga: TALE I. / TALE II. etc. followed by ALL-CAPS title.
#    Second saga: standalone ALL-CAPS title blocks.
# ---------------------------------------------------------------------------

def parse_mongolian(filepath: str) -> list[dict]:
    """
    Parse Mongolian/Kalmuck tales from Gutenberg #40402.

    Structure:
      Part 1 (Well-and-Wise-Walking Khan): TALE I. (blank) TITLE IN ALL CAPS. (blank) text...
      Part 2 (Ardschi-Bordschi saga): standalone ALL-CAPS title lines with story text below.

    The file has TWO copies of the notes section (the translator's notes repeat all
    tale numbers). We only take the FIRST occurrence of each TALE N. block.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    start_marker = raw.find('*** START OF THE PROJECT GUTENBERG EBOOK')
    end_marker   = raw.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    if start_marker != -1:
        raw = raw[start_marker:]
    if end_marker != -1:
        raw = raw[:end_marker]

    # --- Part 1: Well-and-Wise-Walking Khan ---
    # Locate the saga body (after "THE SAGA OF THE WELL-AND-WISE-WALKING KHAN.\n\nDEDICATION")
    p1_start = raw.find('\nTALE I.\n')
    # The Conclusion of the first saga signals the end of the tale-bearing section
    conclusion_start = raw.find('\nCONCLUSION OF THE ADVENTURES OF THE WELL-AND-WISE-WALKING KHAN.')
    notes_start = raw.find('\nNOTES.\n')
    if notes_start == -1:
        notes_start = raw.find('\nFOOTNOTES\n')
    # End the tale extraction at the CONCLUSION (so TALE XXIII doesn't swallow everything)
    p1_end = conclusion_start if conclusion_start != -1 else (notes_start if notes_start != -1 else len(raw))
    p1_text = raw[p1_start:p1_end]

    # Map from roman-numeral story number to its title (from TOC)
    tale_titles = {
        'I':    'The Woman Who Sought Her Husband in the Palace of Erlik Khan',
        'II':   'The Gold-Spitting Prince',
        'III':  'How the Schimnu-Khan Was Slain',
        'IV':   "The Pig's-Head Soothsayer",
        'V':    'How the Serpent-Gods Were Propitiated',
        'VI':   'The Turbulent Subject',
        'VII':  'The White Bird and His Wife',
        'VIII': 'How Ananda the Wood-Carver and Ananda the Painter Strove Against Each Other',
        'IX':   'Five to One',
        'X':    'The Biting Corpse',
        'XI':   'The Prayer Making Suddenly Rich',
        'XII':  '"Child-Intellect" and "Bright-Intellect"',
        'XIII': 'The Fortunes of Shrikantha',
        'XIV':  'The Avaricious Brother',
        'XV':   'The Use of Magic Language',
        'XVI':  'The Wife Who Loved Butter',
        'XVII': 'The Simple Husband and the Prudent Wife',
        'XVIII':'How Shanggasba Buried His Father',
        'XIX':  'The Perfidious Friend',
        'XX':   'Bhixu Life',
        'XXI':  "How the Widow Saved Her Son's Life",
        'XXII': 'The White Serpent-King',
        'XXIII':'What Became of the Red-Coloured Dog',
    }

    rows = []
    # Split on TALE [ROMAN].\n
    tale_pattern = re.compile(r'^TALE ([IVXLC]+)\.$', re.MULTILINE)
    tale_matches = list(tale_pattern.finditer(p1_text))

    for i, m in enumerate(tale_matches):
        roman = m.group(1)
        title = tale_titles.get(roman, f'Tale {roman}')
        body_start = m.end()
        body_end = tale_matches[i+1].start() if i+1 < len(tale_matches) else len(p1_text)
        body = p1_text[body_start:body_end]
        # Skip the story-specific ALL-CAPS title line at the top of the body
        # (it duplicates the title we already have)
        body_lines = body.lstrip('\n').split('\n')
        skip = 0
        for bl in body_lines:
            alpha = re.sub(r'[^A-Za-z]', '', bl.strip())
            if not bl.strip():
                skip += 1
                continue
            if alpha and alpha == alpha.upper():
                skip += 1
                continue
            break
        body = '\n'.join(body_lines[skip:])
        body_clean = clean_text(body)
        if len(body_clean) < 200:
            continue
        rows.append({
            'title': title,
            'text': body_clean,
            'region': 'mongolia',
            'source': 'gutenberg_40402',
        })

    # --- Part 2: Ardschi-Bordschi saga ---
    # Find the body of the second saga (after its header and historical notice)
    saga2_marker = 'THE BOY-KING.'
    saga2_pos = raw.find('\n' + saga2_marker + '\n')
    saga2_end = notes_start if notes_start != -1 else len(raw)

    ardschi_story_titles = [
        ('THE BOY-KING', 'The Boy-King'),
        ('THE FALSE FRIEND', 'The False Friend'),
        ('THE PRETENDED SON', 'The Pretended Son'),
        ("ARDSCHI-BORDSCHI DISCOVERS VIKRAM", "Ardschi-Bordschi Discovers Vikramaditja's Throne"),
        ('WHO INVENTED WOMAN', 'Who Invented Woman?'),
        ('THE VOICE-CHARMER', 'The Voice-Charmer'),
        ('HOW NARAN GEREL SWORE FALSELY', 'How Naran Gerel Swore Falsely and Yet Told the Truth'),
        ('THE WISE PARROT', 'The Wise Parrot'),
    ]

    if saga2_pos != -1:
        saga2_text = raw[saga2_pos:saga2_end]
        # Find each story by looking for its ALL-CAPS line
        bounds = []
        for key, friendly_title in ardschi_story_titles:
            # Search for the key (partial match to handle encoding issues)
            m = re.search(r'\n(' + re.escape(key) + r'[^\n]*)\n', saga2_text)
            if m:
                bounds.append((m.start(), friendly_title, m.end()))

        # Sort by position
        bounds.sort(key=lambda x: x[0])

        for j, (pos, friendly_title, text_start) in enumerate(bounds):
            text_end = bounds[j+1][0] if j+1 < len(bounds) else len(saga2_text)
            body = saga2_text[text_start:text_end]
            body_clean = clean_text(body)
            if len(body_clean) < 200:
                continue
            rows.append({
                'title': friendly_title,
                'text': body_clean,
                'region': 'mongolia',
                'source': 'gutenberg_40402',
            })

    return rows


# ---------------------------------------------------------------------------
# 3. THAI/LAOS — Gutenberg #35564
#    "Laos Folk-Lore of Farther India" (Laos country was part of Siam/Thailand)
#    Story titles: isolated title-case lines between blank lines
# ---------------------------------------------------------------------------

LAOS_STORY_TITLES = [
    "A Child of The Woods",
    "The Enchanted Mountain",
    "The Spirit-Guarded Cave",
    "The Mountain Spirits and the Stone Mortars",
    "Right and Might",
    "Why the Lip of the Elephant Droops",
    "How a Dead Tiger Killed the Princess",
    "The Monkeys and the Crabs",
    "The Man in the Moon",
    "The Origin of Lightning",
    "Why the Parrot and the Minor Bird but Echo the Words of Man",
    "The Fatherless Birds",
    "The Lovers' Leap",
    "The Faithful Husband",
    "The Faithful Wife",
    "An Unexpected Issue",
    "The Giants' Mountain and the Temple",
    "Cheating the Priest",
    "The Disappointed Priest",
    "The Greedy Priest",
    "The Ambitious Priest",
    "The Wizard and the Beggar",
    "A Covetous Neighbor",
    "A Lazy Man's Plot",
    "The Ungrateful Fisherman",
    "The Legend of the Rice",
    '"One Woman in Deceit and Craft is More Than a Match for Eight Men"',
    '"The Wisest Man of a Small Village is Not Equal in Wisdom to a Boy of the City Streets"',
    '"To Aid Beast is Merit; To Aid Man is but Vanity"',
    "Love's Secrets",
    "Poison-Mouth",
    "Strife and Peace",
    "The Widow's Punishment",
    "Honesty Rewarded",
    "The Justice of In Ta Pome",
    "The Words of Untold Value",
    "A Wise Philosopher",
    "The Boys Who Were Not Appreciated",
    "The Magic Well",
    "The Fortunes of Ai Powlo",
    "The Fortunes of a Lazy Beggar",
    "The Misfortunes of Paw Yan",
    "An Unfortunate Shot",
    "The Blind Man",
    "Heads I Win, Tails You Lose",
    "The Great Boaster",
    "A Clever Thief",
    "Eyeless-Needle, Rotten-Egg, Rotten-Banana, Old-Fish and Broken-Pestle",
]


def parse_laos(filepath: str) -> list[dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    start = raw.find('*** START OF THE PROJECT GUTENBERG EBOOK')
    end   = raw.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    if start != -1:
        raw = raw[start:]
    if end != -1:
        raw = raw[:end]

    # Build a regex that matches any of the known story titles
    # We escape each title and anchor to start-of-line after blank line
    escaped = [re.escape(t) for t in LAOS_STORY_TITLES]
    pattern = r'(?m)^(' + '|'.join(escaped) + r')\s*$'

    matches = list(re.finditer(pattern, raw))

    rows = []
    for i, m in enumerate(matches):
        title = m.group(1).strip().strip('"')
        text_start = m.end()
        text_end = matches[i+1].start() if i+1 < len(matches) else len(raw)
        body = raw[text_start:text_end]
        body_clean = clean_text(body)
        if len(body_clean) < 200:
            continue
        rows.append({
            'title': title,
            'text': body_clean,
            'region': 'thailand',  # Laos country was part of the kingdom of Siam (Thailand)
            'source': 'gutenberg_35564',
        })

    return rows


# ---------------------------------------------------------------------------
# 4. SHAN/BURMA — Gutenberg #32375
#    Used as a Southeast Asian proxy; Burma/Shan region bordering Thailand.
#    Story titles: ALL-CAPS lines (often in quotes).
# ---------------------------------------------------------------------------

def parse_shan(filepath: str) -> list[dict]:
    """
    Shan story titles appear as ALL-CAPS isolated lines (surrounded by blank lines)
    after the 'FOLK LORE STORIES' marker. Some are quoted like "A LAUNG KHIT."[1].
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Only parse after the body begins
    body_start = raw.find('FOLK LORE STORIES\n')
    end_marker = raw.find('*** END OF THE PROJECT GUTENBERG EBOOK')
    if body_start == -1:
        body_start = 0
    body = raw[body_start:end_marker if end_marker != -1 else len(raw)]

    lines = body.split('\n')

    # Collect story boundary line indices (all-caps lines between blank lines)
    boundary_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Remove quotes and footnote refs for caps check
        check = re.sub(r'[\[\]\d\."\']+', '', stripped).strip()
        alpha = re.sub(r'[^A-Za-z]', '', check)
        if (alpha and alpha == alpha.upper() and
                5 < len(stripped) < 100 and
                'FOLK LORE STORIES' not in stripped and
                'GLOSSARY' not in stripped and
                'ILLUSTRATIONS' not in stripped):
            prev_blank = (i == 0 or lines[i-1].strip() == '')
            next_blank = (i+1 >= len(lines) or lines[i+1].strip() == '')
            if prev_blank and next_blank:
                boundary_indices.append(i)

    # Filter out TOC/header entries: they have trailing page numbers
    SKIP_TITLES = {'INTRODUCTION', 'CONTENTS', 'LIST OF ILLUSTRATIONS',
                   'FOLK LORE STORIES', 'GLOSSARY OF TERMS'}

    rows = []
    for j, start_line in enumerate(boundary_indices):
        end_line = boundary_indices[j+1] if j+1 < len(boundary_indices) else len(lines)
        # Clean up the title
        title = lines[start_line].strip()
        # Skip TOC lines that end with page numbers like "  9" or "  92"
        if re.search(r'\s{2,}\d+\s*$', title):
            continue
        # Skip known non-story headers
        title_clean = re.sub(r'\[\d+\]', '', title).strip()
        title_clean = title_clean.strip('"').strip('.').strip()
        if title_clean in SKIP_TITLES:
            continue

        body_chunk = '\n'.join(lines[start_line+1:end_line])
        body_clean = clean_text(body_chunk)
        if len(body_clean) < 200:
            continue
        rows.append({
            'title': title_clean,
            'text': body_clean,
            'region': 'myanmar',   # Shan states of Burma/Myanmar
            'source': 'gutenberg_32375',
        })

    return rows


# ---------------------------------------------------------------------------
# Main: parse, report, save CSVs
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Parsing Gutenberg Folklore Collections")
    print("=" * 60)

    # --- Tibetan ---
    tibetan_rows = parse_tibetan(os.path.join(base, 'tibetan_75000.txt'))
    print(f"\n[1] TIBETAN — 'Folk tales from Tibet' by W.F. O'Connor (Gutenberg #75000)")
    print(f"    Stories extracted: {len(tibetan_rows)}")
    for r in tibetan_rows:
        print(f"    • {r['title'][:70]!r}  ({len(r['text'])} chars)")
    df_tibetan = pd.DataFrame(tibetan_rows)
    df_tibetan.to_csv(os.path.join(base, 'tibetan_oconnor.csv'), index=False)
    print(f"    Saved -> tibetan_oconnor.csv")

    # --- Mongolian ---
    mongolian_rows = parse_mongolian(os.path.join(base, 'mongolian_40402.txt'))
    print(f"\n[2] MONGOLIAN — 'Sagas from the Far East' by R.H. Busk (Gutenberg #40402)")
    print(f"    Stories extracted: {len(mongolian_rows)}")
    for r in mongolian_rows:
        print(f"    • {r['title'][:70]!r}  ({len(r['text'])} chars)")
    df_mongolian = pd.DataFrame(mongolian_rows)
    df_mongolian.to_csv(os.path.join(base, 'mongolian_busk.csv'), index=False)
    print(f"    Saved -> mongolian_busk.csv")

    # --- Thai/Laos ---
    laos_rows = parse_laos(os.path.join(base, 'laos_35564.txt'))
    print(f"\n[3] THAI/LAOS — 'Laos Folk-Lore of Farther India' by K.N. Fleeson (Gutenberg #35564)")
    print(f"    (Laos country was part of the kingdom of Siam/Thailand at time of writing)")
    print(f"    Stories extracted: {len(laos_rows)}")
    for r in laos_rows:
        print(f"    • {r['title'][:70]!r}  ({len(r['text'])} chars)")
    df_laos = pd.DataFrame(laos_rows)
    df_laos.to_csv(os.path.join(base, 'thai_laos_fleeson.csv'), index=False)
    print(f"    Saved -> thai_laos_fleeson.csv")

    # --- Shan/Myanmar (SE Asian proxy) ---
    shan_rows = parse_shan(os.path.join(base, 'shan_32375.txt'))
    print(f"\n[4] MYANMAR/SHAN — 'Shan Folk Lore Stories' by W.C. Griggs (Gutenberg #32375)")
    print(f"    (No dedicated Vietnamese/Annamese collection exists on Gutenberg;")
    print(f"     Shan stories from Burma — the closest available mainland SE Asian collection)")
    print(f"    Stories extracted: {len(shan_rows)}")
    for r in shan_rows:
        print(f"    • {r['title'][:70]!r}  ({len(r['text'])} chars)")
    df_shan = pd.DataFrame(shan_rows)
    df_shan.to_csv(os.path.join(base, 'myanmar_shan_griggs.csv'), index=False)
    print(f"    Saved -> myanmar_shan_griggs.csv")

    # --- Summary ---
    total = len(tibetan_rows) + len(mongolian_rows) + len(laos_rows) + len(shan_rows)
    print(f"\n{'=' * 60}")
    print(f"Total stories across all collections: {total}")
    print(f"{'=' * 60}")
