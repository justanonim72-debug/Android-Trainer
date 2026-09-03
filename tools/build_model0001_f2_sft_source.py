#!/usr/bin/env python3
"""Build the human-first Friend-Core F2 SFT source JSONL.

Human conversational style comes from mDIA + IndoSMD TRAIN. A small bounded
project-authored deterministic protocol slice teaches tool/memory/scheduler
serialization with machine-checkable outputs; it is NOT an LLM-teacher corpus.

No project test split, external test/dev split, or Foundation-v3 validation
record is admitted into F2 train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

SEED="20260903-f2-sft-v1"
VALIDATION_BUCKETS=20  # ~5%

URL_RE=re.compile(r"https?://\S+|www\.\S+",re.I)
EMAIL_RE=re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
HANDLE_RE=re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
PHONE_RE=re.compile(r"(?<!\d)(?:\+?62|0)[\s.-]?(?:\d[\s.-]?){8,13}(?!\d)")
SPACE_RE=re.compile(r"[ \t]+")
EN_MARKERS={
    "please","sorry","thanks","thank","okay","ok","actually","literally","maybe",
    "sure","btw","anyway","weekend","meeting","deadline","work","school","friend",
    "coffee","game","movie","music","random","update","latest","today","tomorrow",
    "reminder","schedule","search","online","offline","mood","feeling","love"
}

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def norm_text(s:str)->str:
    s=unicodedata.normalize("NFKC",s).replace("\u0000"," ")
    s=s.replace("[USERNAME]"," ").replace("[URL]"," ")
    s=URL_RE.sub(" ",s); s=EMAIL_RE.sub(" ",s)
    s=HANDLE_RE.sub(" ",s); s=PHONE_RE.sub(" ",s)
    lines=[]
    for line in s.splitlines():
        line=SPACE_RE.sub(" ",line).strip()
        if line: lines.append(line)
    return "\n".join(lines).strip()

def text_sha(s:str)->str:
    return hashlib.sha256(norm_text(s).encode("utf-8")).hexdigest()

def stable_rank(rid:str)->str:
    """Deterministic ordering key independent of Python hash randomization."""
    return hashlib.sha256((SEED+":"+rid).encode("utf-8")).hexdigest()

def split_for(rid:str)->str:
    h=int(stable_rank(rid)[:16],16)
    return "validation" if h%VALIDATION_BUCKETS==0 else "train"

def looks_codeswitch(messages)->bool:
    text=" ".join(m["content"] for m in messages).lower()
    words=set(re.findall(r"[a-zA-Z]+",text))
    return len(words & EN_MARKERS)>=2

def acceptable_message(s:str)->bool:
    if not (1<=len(s)<=3500): return False
    alpha=sum(ch.isalpha() for ch in s)
    return alpha>=2

def v3_validation_text_shas(project:Path)->set[str]:
    p=project/"artifacts"/"model0001_dataset_v3"/"validation_manifest.jsonl"
    if not p.is_file():
        raise SystemExit(f"STOP: Foundation-v3 validation manifest missing: {p}")
    out=set()
    with p.open("r",encoding="utf-8") as f:
        for raw in f:
            if not raw.strip(): continue
            row=json.loads(raw)
            h=row.get("text_sha256")
            if isinstance(h,str) and len(h)==64: out.add(h)
    if not out:
        raise SystemExit("STOP: Foundation-v3 validation manifest had no hashes")
    return out

def mdia_records(path:Path,excluded:set[str]):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        candidates=[
            n for n in names
            if n.lower().endswith(".csv")
            and ("id_" in Path(n).name.lower() or "indones" in n.lower())
            and not any(x in n.lower() for x in ("translated","test","eval","valid"))
        ]
        preferred=[n for n in candidates if "cleaned_dialogue" in n.lower()]
        if preferred: candidates=preferred
        ordinal=0
        for name in sorted(set(candidates)):
            with z.open(name) as bf:
                txt=io.TextIOWrapper(bf,encoding="utf-8-sig",errors="replace",newline="")
                for row in csv.DictReader(txt):
                    user=norm_text(row.get("source_body") or "")
                    assistant=norm_text(row.get("target_body") or "")
                    if not acceptable_message(user) or not acceptable_message(assistant):
                        continue
                    combined=user+"\n"+assistant
                    if text_sha(combined) in excluded:
                        continue
                    rid=f"mdia:{ordinal:08d}"
                    messages=[
                        {"role":"user","content":user},
                        {"role":"assistant","content":assistant},
                    ]
                    language="id-en" if looks_codeswitch(messages) else "id-ID"
                    style=["natural","dialogue"]
                    if language=="id-en": style.append("code-switch")
                    yield {
                        "id":rid,"language":language,"style":style,
                        "messages":messages,
                        "source":"DoctorDream/mDIA human dialogue; F1 source reuse under SFT objective",
                        "license":"CC-BY-4.0",
                        "quality_score":0.82,
                        "split":split_for(rid),
                        "source_family":"human_natural_dialogue",
                    }
                    ordinal+=1

def indosmd_records(path:Path):
    rows=json.loads(path.read_text(encoding="utf-8"))
    for di,dialogue in enumerate(rows):
        msgs=[]
        for turn in dialogue.get("dialogue",[]):
            who=turn.get("turn")
            utter=norm_text((turn.get("data") or {}).get("utterance") or "")
            if not acceptable_message(utter): continue
            role="user" if who=="driver" else "assistant" if who=="assistant" else None
            if role: msgs.append({"role":role,"content":utter})
        if len(msgs)<2 or not any(m["role"]=="assistant" for m in msgs):
            continue
        rid=f"indosmd:{di:06d}"
        intent=((dialogue.get("scenario") or {}).get("task") or {}).get("intent")
        styles=["task-dialogue","multi-turn"]
        if intent: styles.append("intent-"+str(intent))
        yield {
            "id":rid,"language":"id-ID","style":styles,
            "messages":msgs,
            "source":"dehanalkautsar/IndoToD IndoSMD TRAIN; native-speaker annotated",
            "license":"CC-BY-SA-4.0",
            "quality_score":0.88,
            "split":split_for(rid),
            "source_family":"human_task_dialogue",
        }

def tool_call(name,args):
    return "<tool_call>"+json.dumps(
        {"name":name,"args":args},
        ensure_ascii=False,separators=(",",":")
    )+"</tool_call>"

def deterministic_protocol_records(target_count:int):
    """Bounded verifier-friendly protocol records; no LLM teacher outputs."""
    candidates=[]

    preference_values=[
      ("preference.drink","kopi tanpa gula","minuman favorit"),
      ("preference.food","mie ayam","makanan favorit"),
      ("preference.music","lofi","musik yang disukai"),
      ("preference.study_time","pagi","waktu belajar"),
      ("preference.reply_style","singkat dan santai","gaya balasan"),
      ("preference.exercise","jalan pagi","kebiasaan olahraga"),
      ("preference.snack","pisang goreng","camilan favorit"),
      ("preference.theme","mode gelap","tema aplikasi"),
      ("preference.language","Indonesia santai","gaya bahasa"),
      ("preference.focus","matematika","fokus belajar"),
    ]
    alternatives=[
      "teh tawar","nasi goreng","jazz","malam","langsung ke inti",
      "lari sore","roti bakar","mode terang","campur Indo-English","coding"
    ]
    for i,(key,value,label) in enumerate(preference_values):
        pairs=[
          ("store",f"Ingat ya, {label} gue {value}.",
           tool_call("memory_store",{"key":key,"value":value}),
           "{\"ok\":true}","Oke, gue inget."),
          ("update",f"Update ya, {label} gue sekarang {alternatives[i]}.",
           tool_call("memory_update",{"key":key,"value":alternatives[i]}),
           "{\"ok\":true}","Sip, udah gue update."),
          ("forget",f"Lupain info soal {label} gue.",
           tool_call("memory_forget",{"key":key}),
           "{\"ok\":true}","Oke, udah gue hapus dari ingatan."),
          ("lookup",f"Eh, {label} gue apa ya?",
           tool_call("memory_lookup",{"key":key}),
           json.dumps({"found":True,"value":value},ensure_ascii=False,separators=(",",":")),
           value+"."),
        ]
        for op,u,call,result,reply in pairs:
            rid=f"protocol:memory-{op}:{i:03d}"
            candidates.append({
              "id":rid,"language":"id-ID","style":["tool","memory",op],
              "messages":[
                {"role":"user","content":u},
                {"role":"assistant","content":call},
                {"role":"tool","content":result},
                {"role":"assistant","content":reply},
              ],
              "source":"project deterministic verified protocol v1",
              "license":"project-authored","quality_score":1.0,"split":split_for(rid),
              "source_family":"deterministic_protocol"
            })

    transient=[
      ("Gue lagi di halte sekarang.","Oke."),
      ("Barusan hujan deres di sini.","Waduh 😭"),
      ("Baterai gue tinggal dikit.","Cas dulu kalau bisa 😭"),
      ("Gue lagi makan bentar.","Oke, santai."),
      ("Lagi nunggu temen lima menit.","Sip."),
      ("Barusan lewat depan minimarket.","Oke."),
      ("Sekarang gue duduk di teras.","Santai dulu."),
      ("Tadi gue salah pencet tombol.","Wkwk gapapa."),
    ]
    for i,(u,a) in enumerate(transient):
        rid=f"protocol:no-memory:{i:03d}"
        candidates.append({
          "id":rid,"language":"id-ID","style":["memory","do-not-store","short"],
          "messages":[{"role":"user","content":u},{"role":"assistant","content":a}],
          "source":"project deterministic verified protocol v1",
          "license":"project-authored","quality_score":1.0,"split":split_for(rid),
          "source_family":"deterministic_protocol"
        })

    titles=["minum obat","meeting","lari","belajar","bayar kos","telepon ibu","backup file","isi bensin"]
    dates=["besok","Senin","Jumat","tanggal 5"]
    times=["06:00","09:00","19:00","20:00"]
    ordinal=0
    for title in titles:
      for date in dates:
        for time in times:
          if ordinal>=64: break
          human_time=time.replace("06:00","6 pagi").replace("09:00","9 pagi").replace("19:00","7 malam").replace("20:00","8 malam")
          u=f"Ingetin gue {title} {date} jam {human_time}."
          rid=f"protocol:schedule:{ordinal:03d}"
          candidates.append({
            "id":rid,"language":"id-ID","style":["tool","scheduler"],
            "messages":[
              {"role":"user","content":u},
              {"role":"assistant","content":tool_call("schedule_reminder",{"title":title,"when":f"{date} {time}"})},
              {"role":"tool","content":"{\"ok\":true}"},
              {"role":"assistant","content":"Siap, pengingatnya udah dipasang."},
            ],
            "source":"project deterministic verified protocol v1",
            "license":"project-authored","quality_score":1.0,"split":split_for(rid),
            "source_family":"deterministic_protocol"
          })
          ordinal+=1
        if ordinal>=64: break
      if ordinal>=64: break

    ambiguous=[
      ("Ingetin gue besok jam 8.","Jam 8 pagi atau malam?"),
      ("Pasang reminder hari Jumat.","Jam berapa hari Jumat?"),
      ("Ingetin gue tanggal 10.","Mau diingetin jam berapa tanggal 10?"),
      ("Besok ingetin gue meeting.","Meeting-nya jam berapa?"),
    ]
    for i,(u,a) in enumerate(ambiguous):
        rid=f"protocol:schedule-clarify:{i:03d}"
        candidates.append({
          "id":rid,"language":"id-ID","style":["scheduler","clarification"],
          "messages":[{"role":"user","content":u},{"role":"assistant","content":a}],
          "source":"project deterministic verified protocol v1",
          "license":"project-authored","quality_score":1.0,"split":split_for(rid),
          "source_family":"deterministic_protocol"
        })

    fresh=[
      ("harga tiket kereta","Jakarta Bandung"),
      ("cuaca","Banda Aceh"),
      ("jadwal bioskop","Jakarta"),
      ("harga emas","Indonesia"),
      ("berita teknologi","Indonesia"),
      ("versi Android terbaru",""),
      ("jadwal pertandingan","Indonesia"),
      ("harga bensin","Indonesia"),
    ]
    ordinal=0
    for subject,place in fresh:
      for temporal in ("sekarang","hari ini","terbaru"):
        q=" ".join(x for x in (subject,place,temporal) if x).strip()
        u=q.capitalize()+" gimana?"
        rid=f"protocol:search:{ordinal:03d}"
        candidates.append({
          "id":rid,"language":"id-ID","style":["tool","search","freshness"],
          "messages":[
            {"role":"user","content":u},
            {"role":"assistant","content":tool_call("web_search",{"query":q})},
            {"role":"tool","content":"{\"results\":[{\"title\":\"Hasil terbaru\",\"snippet\":\"Informasi terbaru tersedia dari sumber.\",\"source\":\"source://example\"}]}"},
            {"role":"assistant","content":"Ini butuh info terbaru; gue pakai hasil pencarian yang barusan."},
          ],
          "source":"project deterministic verified protocol v1",
          "license":"project-authored","quality_score":1.0,"split":split_for(rid),
          "source_family":"deterministic_protocol"
        })
        ordinal+=1

    privacy_places=["Banda Aceh","Jakarta","Surabaya","Medan"]
    for i,place in enumerate(privacy_places):
        rid=f"protocol:search-min-private:{i:03d}"
        candidates.append({
          "id":rid,"language":"id-ID","style":["tool","search","privacy"],
          "messages":[
            {"role":"user","content":f"Gue lagi ada urusan pribadi sama keluarga. Cuaca {place} sekarang gimana?"},
            {"role":"assistant","content":tool_call("web_search",{"query":f"cuaca {place} sekarang"})},
            {"role":"tool","content":"{\"results\":[{\"title\":\"Cuaca\",\"snippet\":\"Data cuaca terbaru.\",\"source\":\"source://example\"}]}"},
            {"role":"assistant","content":"Gue cuma kirim query cuacanya, bukan konteks pribadi lu."},
          ],
          "source":"project deterministic verified protocol v1",
          "license":"project-authored","quality_score":1.0,"split":split_for(rid),
          "source_family":"deterministic_protocol"
        })

    # Persona/config consistency: variable identity, not one hard-coded persona.
    personas=[
      ("Nara","santai"),
      ("Ari","singkat"),
      ("Mika","ramah"),
      ("Raka","santai"),
      ("Luna","ringkas"),
      ("Dio","casual"),
    ]
    for i,(name,tone) in enumerate(personas):
        rid=f"protocol:persona:{i:03d}"
        candidates.append({
          "id":rid,"language":"id-ID","style":["persona","multi-turn",tone],
          "messages":[
            {"role":"system","content":f"Nama kamu {name}. Gaya balasan {tone}. Jangan mengulang deskripsi persona tanpa perlu."},
            {"role":"user","content":"Nama lu siapa?"},
            {"role":"assistant","content":f"{name}."},
            {"role":"user","content":"Masih inget?"},
            {"role":"assistant","content":f"Iya, {name}."},
          ],
          "source":"project deterministic verified protocol v1",
          "license":"project-authored","quality_score":1.0,"split":split_for(rid),
          "source_family":"deterministic_protocol"
        })

    candidates.sort(key=lambda r:stable_rank(r["id"]))
    if target_count<=0:
        return []
    # Cycle only if the human corpus is unexpectedly enormous; every cycle gets
    # a distinct ID but retains deterministic verifier-friendly semantics.
    out=[]
    cycle=0
    while len(out)<target_count:
        for base in candidates:
            if len(out)>=target_count: break
            row=json.loads(json.dumps(base,ensure_ascii=False))
            if cycle:
                row["id"]=f"{base['id']}:v{cycle}"
                row["split"]=split_for(row["id"])
            out.append(row)
        cycle+=1
    return out

def validate_record(row):
    if row["split"] not in ("train","validation"): return False
    msgs=row["messages"]
    if not isinstance(msgs,list) or len(msgs)<2: return False
    if not any(m.get("role")=="user" for m in msgs): return False
    if not any(m.get("role")=="assistant" for m in msgs): return False
    return all(
        m.get("role") in ("system","user","assistant","tool")
        and isinstance(m.get("content"),str) and m["content"].strip()
        for m in msgs
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    args=ap.parse_args()
    project=Path(args.project).resolve()
    manifest_path=project/"data"/"raw_f2_sft_sources"/"SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit("STOP: run F2 acquisition first")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status")!="PASS":
        raise SystemExit("STOP: F2 source manifest not PASS")
    guards=manifest["hard_guards"]
    if any(guards.get(k) is not False for k in (
        "external_dev_downloaded","external_test_downloaded",
        "project_test_split_touched","openai_teacher_outputs_used",
        "llm_generated_dialogue_used")):
        raise SystemExit("STOP: F2 source hard guard failed")
    by_id={x["id"]:x for x in manifest["sources"]}
    for spec in by_id.values():
        p=Path(spec["path"])
        if not p.is_file() or sha256(p)!=spec["sha256"]:
            raise SystemExit(f"STOP: F2 source changed: {spec['id']}")

    excluded=v3_validation_text_shas(project)
    records=[]
    records.extend(mdia_records(Path(by_id["mdia_raw_reuse"]["path"]),excluded))
    records.extend(indosmd_records(Path(by_id["indotod_indosmd_train"]["path"])))
    human_count=len(records)
    if human_count<=0:
        raise SystemExit("STOP: no human F2 dialogue records")
    # ~10% of final records, capped so protocol templates cannot dominate.
    protocol_target=max(64,min(1200,int(round(human_count/9.0))))
    records.extend(deterministic_protocol_records(protocol_target))

    seen=set(); clean=[]
    for row in records:
        if not validate_record(row): continue
        sig=hashlib.sha256(json.dumps(
            row["messages"],ensure_ascii=False,sort_keys=True,separators=(",",":")
        ).encode()).hexdigest()
        if sig in seen: continue
        seen.add(sig); clean.append(row)

    split=Counter(r["split"] for r in clean)
    family=Counter(r["source_family"] for r in clean)
    lang=Counter(r["language"] for r in clean)
    if split["train"]==0 or split["validation"]==0:
        raise SystemExit("STOP: F2 source split empty")
    if family["human_natural_dialogue"]==0 or family["human_task_dialogue"]==0:
        raise SystemExit("STOP: human dialogue families missing")
    protocol_share=family["deterministic_protocol"]/max(1,len(clean))
    if protocol_share>0.13:
        raise SystemExit(f"STOP: deterministic protocol source dominance {protocol_share:.4f}")

    outdir=project/"data"/"f2_sft"
    outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/"friend_f2_sft_source.jsonl"
    clean.sort(key=lambda r:r["id"])
    with out.open("w",encoding="utf-8") as f:
        for row in clean:
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")

    report={
      "status":"PASS",
      "schema":"model0001_f2_sft_source_build_v1",
      "output":str(out),
      "sha256":sha256(out),
      "records":len(clean),
      "split_counts":dict(split),
      "source_family_counts":dict(family),
      "language_counts":dict(lang),
      "deterministic_protocol_record_fraction":protocol_share,
      "human_dialogue_record_fraction":1.0-protocol_share,
      "foundation_v3_validation_hashes_excluded":len(excluded),
      "tool_protocol":"<tool_call>{json}</tool_call>",
      "hard_guards":{
        "project_test_split_touched":False,
        "external_test_or_dev_used":False,
        "foundation_v3_validation_used_for_train":False,
        "openai_teacher_outputs_used":False,
        "human_dialogue_dominates_style_data":True
      }
    }
    rp=outdir/"F2_SOURCE_BUILD_REPORT.json"
    rp.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))

if __name__=="__main__": main()
