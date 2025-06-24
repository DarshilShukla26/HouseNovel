# HouseNovel

# Minneapolis 1900 Directory Extraction 

This document consolidates all scripts, configurations, and instructions you’ve developed so far for extracting text from scanned directory pages and converting that text into structured JSON.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [OCR Pipeline (`pillow_box_ocr.py`)](#ocr-pipeline)
4. [Text → JSON Parser (`text_to_json.py`)](#text-json-parser)

---

## Overview

This suite performs two main tasks:

1. **OCR Extraction**: Reads images `104.png`–`108.png` from a specified folder, detects text blocks via Pillow+OpenCV morphological operations, and runs Tesseract on each block to produce a consolidated text file (`text.txt`).
2. **Parsing**: Reads `text.txt` (lines starting with `"`), uses regex and spaCy NER to extract name, spouse, occupation, company, and address fields, then outputs a JSON array (`combined.json`).

All paths are configurable at the top of each script.

---

## Prerequisites

* **Python** 3.8+
* **Tesseract OCR**

  ```bash
  brew install tesseract
  ```
* **Python Libraries** via pip:

  ```bash
  pip install pillow opencv-python numpy pytesseract spacy
  python -m spacy download en_core_web_sm
  ```

---

## OCR Pipeline (`pillow_box_ocr.py`)

```python
#!/usr/bin/env python3
"""
Pillow + OpenCV Box‑based OCR → text.txt

1. Preprocess pages with Pillow (grayscale, sharpen, threshold)
2. Detect text boxes via OpenCV morphology
3. OCR each crop with Tesseract
4. Group lines into entries (starts with `"`, splits on `|`)
5. Write one entry per line to `text.txt`
"""
from PIL import Image, ImageOps, ImageFilter
import pytesseract, cv2, numpy as np
from pathlib import Path

# Configuration
BASE_DIR     = Path("/Users/darshilshukla/Desktop")
IMAGE_NAMES  = ["104.png","105.png","106.png","107.png","108.png"]
IMAGE_PATHS  = [BASE_DIR / name for name in IMAGE_NAMES]
OUTPUT_FILE  = BASE_DIR / "text.txt"
TS_CONFIG    = "--oem 3 --psm 6"
PADDING      = 5

# Preprocess for binary image

def preprocess(img_pil: Image.Image) -> np.ndarray:
    gray = ImageOps.grayscale(img_pil).filter(ImageFilter.SHARPEN)
    bw = gray.point(lambda x:0 if x<128 else 255, mode="1")
    return np.array(bw, dtype=np.uint8)

# Detect text boxes

def detect_boxes(bw: np.ndarray):
    inv = cv2.bitwise_not(bw)
    hor = cv2.dilate(inv, cv2.getStructuringElement(cv2.MORPH_RECT,(50,1)), iterations=2)
    ver = cv2.dilate(hor, cv2.getStructuringElement(cv2.MORPH_RECT,(1,20)), iterations=2)
    contours,_ = cv2.findContours(ver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    h,w = bw.shape
    for c in contours:
        x,y,cw,ch = cv2.boundingRect(c)
        if cw*ch<1000: continue
        xa,ya = max(0,x-PADDING), max(0,y-PADDING)
        xb,yb = min(w,x+cw+PADDING), min(h,y+ch+PADDING)
        boxes.append((xa,ya,xb-xa,yb-ya))
    boxes.sort(key=lambda b:(b[1],b[0]))
    return boxes

# OCR a box crop

def ocr_box(img_pil: Image.Image, box):
    x,y,w,h = box
    crop = img_pil.crop((x,y,x+w,y+h))
    text = pytesseract.image_to_string(crop,config=TS_CONFIG)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

# Group lines into entries

def group_entries(lines):
    entries,cur=[],None
    for line in lines:
        for seg in line.split("|"):
            seg=seg.strip()
            if not seg:
                if cur: entries.append(cur); cur=None; continue
            if seg.startswith('"'):
                if cur: entries.append(cur)
                cur=seg
            else:
                if cur: cur += " "+seg
    if cur: entries.append(cur)
    return entries

# Main

def main():
    all_lines=[]
    for p in IMAGE_PATHS:
        if not p.exists(): print(f"Missing {p}"); continue
        img=Image.open(p)
        bw=preprocess(img)
        for box in detect_boxes(bw): all_lines += ocr_box(img,box)
    entries = group_entries(all_lines)
    OUTPUT_FILE.write_text("\n".join(entries),encoding='utf-8')
    print(f"✅ {len(entries)} entries written → {OUTPUT_FILE}")

if __name__=='__main__': main()
```

---

## Text → JSON Parser (`text_to_json.py`)

```python
#!/usr/bin/env python3
"""
Text file → structured JSON

Reads lines starting with `"` from `text.txt`, extracts fields via regex + spaCy, outputs `combined.json`.
"""
import re,json,subprocess,sys
from pathlib import Path
# Auto‑install spaCy & model
try:
    import spacy
    nlp=spacy.load("en_core_web_sm")
except:
    subprocess.check_call([sys.executable,"-m","pip","install","spacy"])
    subprocess.check_call([sys.executable,"-m","spacy","download","en_core_web_sm"])
    import spacy; nlp=spacy.load("en_core_web_sm")

# Configurable paths\INPUT_FILE=Path("/Users/darshilshukla/Desktop/text.txt")
OUTPUT_FILE=Path("/Users/darshilshukla/Desktop/combined.json")
DIRECTORY_NAME="Minneapolis 1900"; PAGE_NUMBER=None
# Patterns
SPOUSE=re.compile(r"\(([^)]+)\)")
ADDR=re.compile(r"(?P<number>\d+)\s+(?P<street>[^,]+?)(?:\s*(?P<apt>apt\s*\d+))?$",re.I)
OCC_KW=["salesman","merchant","clerk","engineer","teacher","laborer","driver","barber","baker","physician","carpenter","nurse","pntr","meat ctr"]

# Parse one line

def parse_line(l):
    t=l.lstrip('"').strip();doc=nlp(t)
    # Names
    persons=[e.text for e in doc.ents if e.label_=='PERSON']
    if persons: fn,ln=persons[0].split()[0],persons[0].split()[-1]
    else: toks=t.split();fn=toks[0] if toks else None;ln=toks[1] if len(toks)>1 else None
    # Spouse
    sp=SPOUSE.search(t); spouse=sp.group(1) if sp else None
    # Addr
    m=ADDR.search(t)
    if m: num,street,apt=m.group('number'),m.group('street').strip(),m.group('apt')
    else: num,street,apt=None,None,None
    # Occupation
    occ=next((kw.title() for kw in OCC_KW if kw in t.lower()),None)
    # Company
    orgs=[e.text for e in doc.ents if e.label_=='ORG']
    comp=orgs[-1] if orgs else None
    return {"FirstName":fn,"LastName":ln,"Spouse":spouse,"Occupation":occ,
            "CompanyName":comp,"HomeAddress":{"StreetNumber":num,"StreetName":street,
            "ApartmentOrUnit":apt,"ResidenceIndicator":"h"},"WorkAddress":None,
            "Telephone":None,"DirectoryName":DIRECTORY_NAME,"PageNumber":PAGE_NUMBER}

# Main

def main():
    if not INPUT_FILE.exists(): print(f"Missing {INPUT_FILE}");return
    rec=[]
    for line in INPUT_FILE.read_text().splitlines():
        if line.strip().startswith('"'): rec.append(parse_line(line))
    OUTPUT_FILE.write_text(json.dumps(rec,indent=2))
    print(f"✅ Parsed {len(rec)} entries → {OUTPUT_FILE}")

if __name__=='__main__': main()
```
