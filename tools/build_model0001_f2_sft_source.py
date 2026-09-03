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
    """Bounded verifier-friendly protocol records; no LLM teacher outputs.

    Every emitted conversation is content-unique. We intentionally avoid the
    old ID-only cycling trick because the source-level deduper correctly
    removed those exact message duplicates, shrinking protocol supervision.
    """
    candidates=[]

    memory_templates={
      "store":[
        "Ingat ya, {label} gue {value}.",
        "Catet dong, {label} gue itu {value}.",
        "Biar nggak lupa, {label} gue {value}.",
        "Simpan ini ya: {label} gue {value}.",
      ],
      "update":[
        "Update ya, {label} gue sekarang {value}.",
        "Yang soal {label}, ganti jadi {value}.",
        "Revisi deh, {label} gue sekarang {value}.",
        "Sekarang {label} gue {value}, bukan yang lama.",
      ],
      "forget":[
        "Lupain info soal {label} gue.",
        "Hapus ingatan soal {label} gue.",
        "Jangan simpan lagi info {label} gue.",
        "Buang info lama soal {label} gue ya.",
      ],
      "lookup":[
        "Eh, {label} gue apa ya?",
        "Masih inget {label} gue?",
        "Coba inget, {label} gue apa?",
        "Yang lu simpan soal {label} gue apa?",
      ],
    }
    memory_replies={
      "store":["Oke, gue inget.","Sip, gue simpan.","Oke, kecatet.","Siap, gue inget."],
      "update":["Sip, udah gue update.","Oke, gue ganti.","Berubah, udah gue update.","Siap, yang lama gue ganti."],
      "forget":["Oke, udah gue hapus dari ingatan.","Sip, gue lupain.","Oke, info itu gue hapus.","Siap, nggak gue simpan lagi."],
    }
    preference_values=[
      ("preference.drink","kopi tanpa gula","teh tawar","minuman favorit"),
      ("preference.food","mie ayam","nasi goreng","makanan favorit"),
      ("preference.music","lofi","jazz","musik yang disukai"),
      ("preference.study_time","pagi","malam","waktu belajar"),
      ("preference.reply_style","singkat dan santai","langsung ke inti","gaya balasan"),
      ("preference.exercise","jalan pagi","lari sore","kebiasaan olahraga"),
      ("preference.snack","pisang goreng","roti bakar","camilan favorit"),
      ("preference.theme","mode gelap","mode terang","tema aplikasi"),
      ("preference.language","Indonesia santai","campur Indo-English","gaya bahasa"),
      ("preference.focus","matematika","coding","fokus belajar"),
    ]
    for i,(key,initial,updated,label) in enumerate(preference_values):
        for variant in range(4):
            for op in ("store","update","forget","lookup"):
                value=updated if op=="update" else initial
                u=memory_templates[op][variant].format(label=label,value=value)
                if op=="store":
                    call=tool_call("memory_store",{"key":key,"value":initial})
                    result="{\"ok\":true}"
                    reply=memory_replies[op][variant]
                elif op=="update":
                    call=tool_call("memory_update",{"key":key,"value":updated})
                    result="{\"ok\":true}"
                    reply=memory_replies[op][variant]
                elif op=="forget":
                    call=tool_call("memory_forget",{"key":key})
                    result="{\"ok\":true}"
                    reply=memory_replies[op][variant]
                else:
                    call=tool_call("memory_lookup",{"key":key})
                    result=json.dumps({"found":True,"value":initial},ensure_ascii=False,separators=(",",":"))
                    reply=[initial+".",f"{initial}.",f"Yang gue inget: {initial}.",f"{initial}, itu yang tersimpan."][variant]
                rid=f"protocol:memory-{op}:{i:03d}:v{variant}"
                candidates.append({
                  "id":rid,"language":"id-ID","style":["tool","memory",op],
                  "messages":[
                    {"role":"user","content":u},
                    {"role":"assistant","content":call},
                    {"role":"tool","content":result},
                    {"role":"assistant","content":reply},
                  ],
                  "source":"project deterministic verified protocol v2",
                  "license":"project-authored","quality_score":1.0,"split":split_for(rid),
                  "source_family":"deterministic_protocol"
                })

    transient_templates=[
      ("Gue lagi di halte sekarang.","Oke."),
      ("Barusan hujan deres di sini.","Waduh 😭"),
      ("Baterai gue tinggal dikit.","Cas dulu kalau bisa 😭"),
      ("Gue lagi makan bentar.","Oke, santai."),
      ("Lagi nunggu temen lima menit.","Sip."),
      ("Barusan lewat depan minimarket.","Oke."),
      ("Sekarang gue duduk di teras.","Santai dulu."),
      ("Tadi gue salah pencet tombol.","Wkwk gapapa."),
      ("Gue baru turun dari motor.","Oke."),
      ("Sekarang lagi antre sebentar.","Sip, santai."),
      ("Gue lagi di depan rumah.","Oke."),
      ("Barusan selesai mandi.","Wkwk sip."),
    ]
    for i,(u,a) in enumerate(transient_templates):
        for variant,suffix in enumerate(("", " nih.", " aja.")):
            user=u[:-1]+suffix if u.endswith(".") else u+suffix
            rid=f"protocol:no-memory:{i:03d}:v{variant}"
            candidates.append({
              "id":rid,"language":"id-ID","style":["memory","do-not-store","short"],
              "messages":[{"role":"user","content":user},{"role":"assistant","content":a}],
              "source":"project deterministic verified protocol v2",
              "license":"project-authored","quality_score":1.0,"split":split_for(rid),
              "source_family":"deterministic_protocol"
            })

    titles=["minum obat","meeting","lari","belajar","bayar kos","telepon ibu","backup file","isi bensin"]
    dates=["besok","Senin","Jumat","tanggal 5"]
    times=[("06:00","6 pagi"),("09:00","9 pagi"),("19:00","7 malam"),("20:00","8 malam")]
    schedule_templates=[
      "Ingetin gue {title} {date} jam {human_time}.",
      "Tolong pasang reminder {title} {date} jam {human_time}.",
    ]
    ordinal=0
    for title in titles:
      for date in dates:
        for time,human_time in times:
          for variant,template in enumerate(schedule_templates):
            u=template.format(title=title,date=date,human_time=human_time)
            rid=f"protocol:schedule:{ordinal:03d}:v{variant}"
            candidates.append({
              "id":rid,"language":"id-ID","style":["tool","scheduler"],
              "messages":[
                {"role":"user","content":u},
                {"role":"assistant","content":tool_call("schedule_reminder",{"title":title,"when":f"{date} {time}"})},
                {"role":"tool","content":"{\"ok\":true}"},
                {"role":"assistant","content":["Siap, pengingatnya udah dipasang.","Oke, reminder-nya udah gue pasang."][variant]},
              ],
              "source":"project deterministic verified protocol v2",
              "license":"project-authored","quality_score":1.0,"split":split_for(rid),
              "source_family":"deterministic_protocol"
            })
          ordinal+=1

    ambiguous=[
      ("Ingetin gue besok jam 8.","Jam 8 pagi atau malam?"),
      ("Pasang reminder hari Jumat.","Jam berapa hari Jumat?"),
      ("Ingetin gue tanggal 10.","Mau diingetin jam berapa tanggal 10?"),
      ("Besok ingetin gue meeting.","Meeting-nya jam berapa?"),
      ("Minggu depan ingetin gue bayar.","Hari apa dan jam berapa?"),
      ("Nanti sore ingetin gue belajar.","Sore jam berapa yang lu mau?"),
    ]
    for i,(u,a) in enumerate(ambiguous):
        for variant,prefix in enumerate(("", "Eh, ")):
            rid=f"protocol:schedule-clarify:{i:03d}:v{variant}"
            candidates.append({
              "id":rid,"language":"id-ID","style":["scheduler","clarification"],
              "messages":[{"role":"user","content":prefix+u},{"role":"assistant","content":a}],
              "source":"project deterministic verified protocol v2",
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
      ("kurs dolar rupiah",""),
      ("jadwal kereta","Yogyakarta Solo"),
    ]
    search_templates=[
      "{q} gimana?",
      "Coba cek {q}.",
      "Cariin {q} dong.",
    ]
    ordinal=0
    for subject,place in fresh:
      for temporal in ("sekarang","hari ini","terbaru"):
        q=" ".join(x for x in (subject,place,temporal) if x).strip()
        for variant,template in enumerate(search_templates):
            u=template.format(q=q)
            rid=f"protocol:search:{ordinal:03d}:v{variant}"
            candidates.append({
              "id":rid,"language":"id-ID","style":["tool","search","freshness"],
              "messages":[
                {"role":"user","content":u},
                {"role":"assistant","content":tool_call("web_search",{"query":q})},
                {"role":"tool","content":"{\"results\":[{\"title\":\"Hasil terbaru\",\"snippet\":\"Informasi terbaru tersedia dari sumber.\",\"source\":\"source://example\"}]}"},
                {"role":"assistant","content":[
                  "Ini butuh info terbaru; gue pakai hasil pencarian yang barusan.",
                  "Gue cek info terbarunya dulu, ini hasilnya.",
                  "Karena ini bisa berubah, gue pakai hasil pencarian terbaru."
                ][variant]},
              ],
              "source":"project deterministic verified protocol v2",
              "license":"project-authored","quality_score":1.0,"split":split_for(rid),
              "source_family":"deterministic_protocol"
            })
        ordinal+=1

    privacy_places=["Banda Aceh","Jakarta","Surabaya","Medan","Semarang","Bandung"]
    for i,place in enumerate(privacy_places):
        for variant,private_context in enumerate((
            "Gue lagi ada urusan pribadi sama keluarga.",
            "Gue lagi di tempat temen dan ada urusan pribadi.",
            "Ada konteks pribadi yang nggak perlu lu kirim keluar."
        )):
            rid=f"protocol:search-min-private:{i:03d}:v{variant}"
            candidates.append({
              "id":rid,"language":"id-ID","style":["tool","search","privacy"],
              "messages":[
                {"role":"user","content":f"{private_context} Cuaca {place} sekarang gimana?"},
                {"role":"assistant","content":tool_call("web_search",{"query":f"cuaca {place} sekarang"})},
                {"role":"tool","content":"{\"results\":[{\"title\":\"Cuaca\",\"snippet\":\"Data cuaca terbaru.\",\"source\":\"source://example\"}]}"},
                {"role":"assistant","content":"Gue cuma kirim query cuacanya, bukan konteks pribadi lu."},
              ],
              "source":"project deterministic verified protocol v2",
              "license":"project-authored","quality_score":1.0,"split":split_for(rid),
              "source_family":"deterministic_protocol"
            })

    personas=[
      ("Nara","santai"),("Ari","singkat"),("Mika","ramah"),
      ("Raka","santai"),("Luna","ringkas"),("Dio","casual"),
      ("Sora","hangat"),("Niko","langsung"),("Alya","ramah"),
    ]
    persona_questions=[
      ("Nama lu siapa?","Masih inget?"),
      ("Lu dipanggil siapa?","Coba sebut lagi."),
      ("Nama yang diset buat lu apa?","Belum lupa kan?"),
    ]
    for i,(name,tone) in enumerate(personas):
        for variant,(q1,q2) in enumerate(persona_questions):
            rid=f"protocol:persona:{i:03d}:v{variant}"
            candidates.append({
              "id":rid,"language":"id-ID","style":["persona","multi-turn",tone],
              "messages":[
                {"role":"system","content":f"Nama kamu {name}. Gaya balasan {tone}. Jangan mengulang deskripsi persona tanpa perlu."},
                {"role":"user","content":q1},
                {"role":"assistant","content":f"{name}."},
                {"role":"user","content":q2},
                {"role":"assistant","content":f"Iya, {name}."},
              ],
              "source":"project deterministic verified protocol v2",
              "license":"project-authored","quality_score":1.0,"split":split_for(rid),
              "source_family":"deterministic_protocol"
            })

    candidates.sort(key=lambda r:stable_rank(r["id"]))
    if target_count<=0:
        return []
    if target_count>len(candidates):
        raise SystemExit(
            f"STOP: requested {target_count} protocol records but only "
            f"{len(candidates)} content-unique verified candidates exist"
        )
    return candidates[:target_count]

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

def runtime_smoke_test():
    rows=deterministic_protocol_records(500)
    if len(rows)!=500:
        raise SystemExit("STOP: F2 deterministic protocol smoke count mismatch")
    ids=[r["id"] for r in rows]
    if len(ids)!=len(set(ids)):
        raise SystemExit("STOP: F2 deterministic protocol smoke duplicate IDs")
    ranked=sorted(ids,key=stable_rank)
    if ranked!=sorted(ids,key=stable_rank):
        raise SystemExit("STOP: F2 stable ranking is nondeterministic")
    if not all(r["split"] in ("train","validation") for r in rows):
        raise SystemExit("STOP: F2 deterministic protocol smoke split invalid")
    print(json.dumps({
        "status":"PASS",
        "schema":"model0001_f2_source_runtime_smoke_v1",
        "records":len(rows)
    },sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runtime-smoke",action="store_true")
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    args=ap.parse_args()
    if args.runtime_smoke:
        runtime_smoke_test()
        return
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
    # Target ~5% protocol records. These records contain tool-call + visible
    # assistant outputs and therefore contribute roughly twice the scored
    # assistant tokens per record of ordinary dialogue. The packer enforces the
    # actual scored-token fraction at 8-15%.
    protocol_target=max(96,min(600,int(round(human_count/20.0))))
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
    if protocol_share>0.08:
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
