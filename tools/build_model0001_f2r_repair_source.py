#!/usr/bin/env python3
"""Build bounded replacement SFT source for Model #0001 after F2 behavior collapse.

This is NOT a continuation of frozen F2. It creates a new replacement SFT
dataset intended to train again from the promoted Foundation-v3 source.

Inputs:
- validated original F2 source JSONL;
- empirical collapse audit;
- locked post-F2 behavior suite ONLY as an exclusion list.

Policies:
- original F2 validation records are never moved into repair train;
- exact assistant-target duplicates are removed;
- high-frequency assistant-start prefixes are capped;
- very short generic acknowledgements are filtered from human style data;
- human dialogue remains the majority;
- a stronger deterministic behavior-core slice targets tool routing, persona,
  continuity, instruction following, uncertainty and basic retained skills;
- behavior-eval prompts are never copied into training;
- project/external test split is untouched.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path


SEED="20260903-f2r-source-v1"
TOKENIZER_SHA="3ab25549638ef1a0b9e718218f402c40b0633455fd2fa2ffb7fd6369ff75d5d7"
WORD_RE=re.compile(r"[A-Za-zÀ-ÿ0-9]+",re.UNICODE)
TOOL_OPEN="<tool_call>"
TOOL_CLOSE="</tool_call>"
GENERIC_FIRST={
    "sama sama","oke","ok","sip","siap","iya","ya","baik","makasih",
    "terima kasih"
}
TARGET_CORE_SCORED_FRACTION=(0.22,0.32)
TARGET_VALIDATION_FRACTION=0.05

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def norm(s:str)->str:
    return " ".join(WORD_RE.findall(s.lower()))

def words(s:str):
    return WORD_RE.findall(s.lower())

def rank(key:str)->str:
    return hashlib.sha256((SEED+":"+key).encode()).hexdigest()

def split_for(key:str)->str:
    v=int(rank(key)[:16],16)/float(16**16)
    return "validation" if v<TARGET_VALIDATION_FRACTION else "train"

def tool_call(name,args):
    return TOOL_OPEN+json.dumps(
        {"name":name,"args":args},
        ensure_ascii=False,separators=(",",":")
    )+TOOL_CLOSE

def assistant_texts(row):
    return [m["content"] for m in row["messages"] if m["role"]=="assistant"]

def scored_tokens(tok,row):
    total=0
    for text in assistant_texts(row):
        total+=len(tok.encode(text,add_special_tokens=False).ids)
    # terminal EOS is scored when record ends with assistant, as in packer.
    if row["messages"] and row["messages"][-1]["role"]=="assistant":
        total+=1
    return total

def record_signature(row):
    return hashlib.sha256(
        json.dumps(row["messages"],ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
    ).hexdigest()

def assistant_signature(row):
    return " || ".join(norm(x) for x in assistant_texts(row))

def first_prefix2(row):
    texts=assistant_texts(row)
    if not texts: return ""
    ws=words(texts[0])
    return " ".join(ws[:2])

def generic_short(row):
    texts=assistant_texts(row)
    if not texts: return True
    # Only filter one-turn human acknowledgement targets. Multi-turn/task/tool
    # records are not affected.
    if len(texts)!=1: return False
    ws=words(texts[0])
    if len(ws)>10: return False
    p2=" ".join(ws[:2])
    return p2 in GENERIC_FIRST or (ws and ws[0] in GENERIC_FIRST)

def eval_prompt_norms(suite_path:Path):
    suite=json.loads(suite_path.read_text(encoding="utf-8"))
    out=set()
    for case in suite.get("cases",[]):
        for msg in case.get("messages",[]):
            if msg.get("role")=="user":
                out.add(norm(msg.get("content","")))
    return out

def no_eval_prompt_overlap(row,excluded:set[str]):
    for msg in row["messages"]:
        if msg["role"]=="user" and norm(msg["content"]) in excluded:
            return False
    return True

def core_record(rid,messages,styles):
    return {
      "id":"f2r:"+rid,
      "language":"id-ID",
      "style":["repair-core"]+styles,
      "messages":messages,
      "source":"project deterministic F2R behavior core v1",
      "license":"project-authored",
      "quality_score":1.0,
      "split":split_for("f2r:"+rid),
      "source_family":"repair_behavior_core",
    }

def build_behavior_core():
    rows=[]

    # Memory routing + result following.
    prefs=[
      ("drink","minuman favorit","teh hijau","cokelat panas"),
      ("food","makanan favorit","soto ayam","nasi uduk"),
      ("music","musik favorit","jazz","lofi"),
      ("study","waktu belajar","pagi","malam"),
      ("style","gaya balasan","ringkas","santai"),
      ("sport","olahraga favorit","renang","lari"),
      ("snack","camilan favorit","roti bakar","pisang goreng"),
      ("theme","tema aplikasi","gelap","terang"),
      ("language","gaya bahasa","Indonesia santai","campur Indonesia Inggris"),
      ("focus","fokus belajar","fisika","matematika"),
    ]
    store_tpl=[
      "Ingat ya, {label} gue {a}.",
      "Catet dong kalau {label} gue {a}.",
      "Mulai sekarang simpan: {label} gue {a}.",
      "Biar nggak lupa, {label} gue itu {a}.",
    ]
    update_tpl=[
      "Update ya, {label} gue sekarang {b}.",
      "Yang soal {label}, ganti jadi {b}.",
      "Revisi info {label} gue jadi {b}.",
      "Sekarang {label} gue {b}, bukan yang lama.",
    ]
    lookup_tpl=[
      "Masih inget {label} gue?",
      "Coba cek ingatan, {label} gue apa?",
      "Yang lu simpan soal {label} gue apa?",
      "Eh {label} gue apa ya?",
    ]
    forget_tpl=[
      "Hapus ingatan soal {label} gue.",
      "Lupain info {label} gue.",
      "Jangan simpan lagi soal {label} gue.",
      "Buang info lama tentang {label} gue ya.",
    ]
    for pi,(key,label,a,b) in enumerate(prefs):
        memkey="preference."+key
        for vi in range(4):
            rows.append(core_record(
                f"memory-store-{pi}-{vi}",
                [
                  {"role":"user","content":store_tpl[vi].format(label=label,a=a)},
                  {"role":"assistant","content":tool_call("memory_store",{"key":memkey,"value":a})},
                  {"role":"tool","content":"{\"ok\":true}"},
                  {"role":"assistant","content":["Oke, gue simpan.","Sip, kecatet.","Siap, gue inget.","Oke, masuk ingatan."][vi]},
                ],["memory","tool","store"]))
            rows.append(core_record(
                f"memory-update-{pi}-{vi}",
                [
                  {"role":"user","content":update_tpl[vi].format(label=label,b=b)},
                  {"role":"assistant","content":tool_call("memory_update",{"key":memkey,"value":b})},
                  {"role":"tool","content":"{\"ok\":true}"},
                  {"role":"assistant","content":["Oke, gue update.","Sip, yang lama gue ganti.","Siap, udah direvisi.","Oke, sekarang gue inget yang baru."][vi]},
                ],["memory","tool","update"]))
            rows.append(core_record(
                f"memory-lookup-{pi}-{vi}",
                [
                  {"role":"user","content":lookup_tpl[vi].format(label=label)},
                  {"role":"assistant","content":tool_call("memory_lookup",{"key":memkey})},
                  {"role":"tool","content":json.dumps({"found":True,"value":a},ensure_ascii=False,separators=(",",":"))},
                  {"role":"assistant","content":[a+".",f"Yang gue inget {a}.",f"{a}, itu yang tersimpan.",f"Masih: {a}."][vi]},
                ],["memory","tool","lookup"]))
            rows.append(core_record(
                f"memory-forget-{pi}-{vi}",
                [
                  {"role":"user","content":forget_tpl[vi].format(label=label)},
                  {"role":"assistant","content":tool_call("memory_forget",{"key":memkey})},
                  {"role":"tool","content":"{\"ok\":true}"},
                  {"role":"assistant","content":["Oke, udah gue hapus.","Sip, gue lupain.","Siap, info itu dihapus.","Oke, nggak gue simpan lagi."][vi]},
                ],["memory","tool","forget"]))

    # Search routing / privacy / tool-result following.
    fresh=[
      ("cuaca Padang sekarang","Cuaca Padang sekarang gimana?"),
      ("harga emas Indonesia hari ini","Harga emas hari ini berapa?"),
      ("kurs dolar rupiah sekarang","Kurs dolar ke rupiah sekarang berapa?"),
      ("jadwal kereta Bandung Jakarta hari ini","Kereta Bandung ke Jakarta hari ini ada jam berapa?"),
      ("versi Chrome terbaru","Chrome versi terbaru sekarang apa?"),
      ("harga tiket Surabaya Bali hari ini","Tiket Surabaya Bali hari ini berapa?"),
      ("berita teknologi Indonesia hari ini","Ada berita teknologi Indonesia terbaru hari ini?"),
      ("jadwal bioskop Semarang hari ini","Bioskop Semarang hari ini tayang apa?"),
    ]
    for i,(q,u) in enumerate(fresh):
        for v in range(3):
            user=[u,"Tolong cek "+u.lower(),"Cari info terbaru: "+u.lower()][v]
            rows.append(core_record(
                f"search-{i}-{v}",
                [
                  {"role":"user","content":user},
                  {"role":"assistant","content":tool_call("web_search",{"query":q})},
                  {"role":"tool","content":"{\"results\":[{\"summary\":\"data terbaru tersedia\"}]}"},
                  {"role":"assistant","content":["Gue cek info terbarunya dulu.","Ini pakai hasil pencarian terbaru.","Karena bisa berubah, gue cek sumber terbaru."][v]},
                ],["search","tool","freshness"]))
    for i,city in enumerate(["Padang","Solo","Malang","Bogor","Batam","Pontianak"]):
        rows.append(core_record(
            f"search-private-{i}",
            [
              {"role":"user","content":f"Gue lagi ada urusan pribadi yang nggak relevan. Cek cuaca {city} sekarang dong."},
              {"role":"assistant","content":tool_call("web_search",{"query":f"cuaca {city} sekarang"})},
              {"role":"tool","content":"{\"results\":[{\"summary\":\"data cuaca terbaru\"}]}"},
              {"role":"assistant","content":"Gue cuma pakai query cuacanya, konteks pribadi lu nggak ikut dikirim."},
            ],["search","privacy","tool"]))

    # Scheduler exact + ambiguity.
    tasks=["jemur pakaian","ambil laundry","belajar fisika","telepon ayah","minum vitamin","backup foto","isi bensin","ambil paket"]
    days=["besok","Senin","Rabu","tanggal 12"]
    times=[("06:30","6.30 pagi"),("09:00","9 pagi"),("16:00","4 sore"),("20:30","8.30 malam")]
    k=0
    for task in tasks:
        for day in days:
            time,human=times[k%len(times)]
            rows.append(core_record(
                f"schedule-{k}",
                [
                  {"role":"user","content":f"Ingetin gue {task} {day} jam {human}."},
                  {"role":"assistant","content":tool_call("schedule_reminder",{"title":task,"when":f"{day} {time}"})},
                  {"role":"tool","content":"{\"ok\":true}"},
                  {"role":"assistant","content":"Siap, pengingatnya udah dipasang."},
                ],["scheduler","tool"]))
            k+=1
    ambiguous=[
      ("Besok ingetin gue bayar listrik.","Jam berapa besok?"),
      ("Hari Sabtu ingetin gue olahraga.","Sabtu jam berapa?"),
      ("Nanti ingetin gue telepon kakak.","Mau diingetin kapan?"),
      ("Minggu depan ingetin gue servis motor.","Hari dan jam berapa minggu depan?"),
      ("Tanggal 20 ingetin gue daftar ulang.","Tanggal 20 jam berapa?"),
      ("Besok jam 8 ingetin gue belajar.","Jam 8 pagi atau malam?"),
    ]
    for i,(u,a) in enumerate(ambiguous):
        rows.append(core_record(
            f"schedule-clarify-{i}",
            [{"role":"user","content":u},{"role":"assistant","content":a}],
            ["scheduler","clarification","no-tool"]))

    # Persona + relationship continuity with identities/topics disjoint from eval.
    names=["Mira","Niko","Alya","Dara","Kian","Sena","Rumi","Tara"]
    tones=["santai","ringkas","ramah","casual"]
    for i,name in enumerate(names):
        tone=tones[i%len(tones)]
        rows.append(core_record(
            f"persona-{i}",
            [
              {"role":"system","content":f"Nama kamu {name}. Balas {tone}. Jangan mengulang persona tanpa perlu."},
              {"role":"user","content":"Lu dipanggil siapa?"},
              {"role":"assistant","content":name+"."},
              {"role":"user","content":"Masih inget nama sendiri?"},
              {"role":"assistant","content":f"Iya, {name}."},
            ],["persona","continuity"]))
    topics=[
      ("sejarah","bab kerajaan"),
      ("kimia","ikatan atom"),
      ("bahasa Inggris","present tense"),
      ("coding","fungsi Python"),
      ("biologi","sel tumbuhan"),
      ("ekonomi","inflasi"),
      ("geografi","peta"),
      ("statistika","rata-rata"),
    ]
    for i,(subject,detail) in enumerate(topics):
        rows.append(core_record(
            f"continuity-{i}",
            [
              {"role":"user","content":f"Gue lagi belajar {subject}, bagian {detail}."},
              {"role":"assistant","content":f"Oke, lanjut {detail} dulu."},
              {"role":"user","content":"Tadi gue bilang belajar apa?"},
              {"role":"assistant","content":f"{subject}, bagian {detail}."},
            ],["continuity","multi-turn"]))

    # No-memory transient chatter; no tool should appear.
    transient=[
      "Gue lagi berdiri depan pagar.",
      "Barusan gue nyalain kipas.",
      "Sekarang lagi nunggu air mendidih.",
      "Tadi gue geser kursi.",
      "Gue lagi pake sandal biru sekarang.",
      "Barusan lewat motor depan rumah.",
      "Gue lagi duduk sebentar.",
      "Tadi lampu kamar gue matiin.",
      "Sekarang gue lagi antre minum.",
      "Barusan gue buka jendela.",
    ]
    for i,u in enumerate(transient):
        rows.append(core_record(
            f"transient-{i}",
            [{"role":"user","content":u},{"role":"assistant","content":["Oke.","Sip.","Wkwk oke.","Santai."][i%4]}],
            ["memory","do-not-store","no-tool"]))

    # Instruction following + concise formatting.
    for i,(u,a) in enumerate([
      ("Jawab satu kata: apakah 3 lebih kecil dari 9?","iya"),
      ("Jawab cuma angka: 6 tambah 7.","13"),
      ("Balas dua kata saja untuk bilang kamu paham.","gue paham"),
      ("Jawab ya atau tidak: apakah 10 lebih besar dari 4?","ya"),
      ("Tulis tiga warna dipisah koma.","merah, biru, hijau"),
      ("Jawab satu kata: lawan kata panas?","dingin"),
      ("Jawab cuma angka: 9 dikali 4.","36"),
      ("Balas satu kata untuk menyetujui.","iya"),
    ]):
        rows.append(core_record(
            f"instruction-{i}",
            [{"role":"user","content":u},{"role":"assistant","content":a}],
            ["instruction","concise"]))

    # Basic retention: many deterministic arithmetic examples + bounded facts.
    idx=0
    for a in range(2,13):
        for b in range(2,13):
            if (a+b+idx)%3==0:
                u=f"Berapa {a} × {b}?"
                ans=str(a*b)
            elif (a+b+idx)%3==1:
                u=f"Hitung {a}+{b}."
                ans=str(a+b)
            else:
                hi=a+b
                u=f"Berapa {hi}-{a}?"
                ans=str(b)
            rows.append(core_record(
                f"math-{idx}",
                [{"role":"user","content":u},{"role":"assistant","content":ans}],
                ["retention","math","instruction"]))
            idx+=1

    facts=[
      ("Air membeku sekitar berapa derajat Celsius?","0 derajat Celsius."),
      ("Planet tempat manusia tinggal apa?","Bumi."),
      ("Satu jam ada berapa menit?","60 menit."),
      ("Mamalia bernapas pakai apa?","Paru-paru."),
      ("Matahari terbit dari arah mana?","Timur."),
      ("Air tersusun dari hidrogen dan unsur apa?","Oksigen."),
      ("Segitiga punya berapa sisi?","Tiga sisi."),
      ("Seratus sentimeter sama dengan berapa meter?","Satu meter."),
      ("Benda jatuh ke bawah karena gaya apa?","Gravitasi."),
      ("Es yang dipanaskan akan berubah jadi apa?","Air."),
    ]
    for i,(u,a) in enumerate(facts):
        rows.append(core_record(
            f"fact-{i}",
            [{"role":"user","content":u},{"role":"assistant","content":a}],
            ["retention","general"]))

    # Uncertainty / epistemic restraint.
    situations=[
      "Temen gue belum bales dua jam. Dia pasti marah ya?",
      "Dia lihat story gue tapi nggak chat. Berarti dia benci gue?",
      "Nilai kuis gue turun sekali. Berarti gue pasti gagal semester?",
      "Orang itu senyum ke gue. Berarti dia pasti suka gue?",
      "Bos belum jawab email. Berarti gue pasti ditolak?",
      "Temen gue pendiem hari ini. Pasti gara-gara gue?",
    ]
    replies=[
      "Belum tentu. Bisa ada banyak alasan lain.",
      "Nggak bisa dipastiin cuma dari itu.",
      "Belum tentu, satu kejadian belum cukup buat nyimpulin.",
      "Bisa jadi, tapi nggak ada cukup info buat memastikan.",
    ]
    for i,u in enumerate(situations):
        rows.append(core_record(
            f"uncertainty-{i}",
            [{"role":"user","content":u},{"role":"assistant","content":replies[i%len(replies)]}],
            ["uncertainty","epistemic"]))

    # Fresh query vs stable no-search distinction.
    stable=[
      ("Kenapa gula larut dalam air?","Karena molekul air berinteraksi dengan partikel gula dan memisahkannya."),
      ("Kenapa bayangan muncul?","Karena cahaya terhalang benda sehingga area di belakangnya menerima lebih sedikit cahaya."),
      ("Kenapa baju basah bisa kering?","Karena air di baju menguap ke udara."),
      ("Kenapa suara bisa terdengar?","Karena getaran merambat melalui medium seperti udara."),
    ]
    for i,(u,a) in enumerate(stable):
        rows.append(core_record(
            f"stable-no-search-{i}",
            [{"role":"user","content":u},{"role":"assistant","content":a}],
            ["search","no-tool","general"]))

    return rows

def runtime_smoke():
    rows=build_behavior_core()
    ids=[r["id"] for r in rows]
    if len(rows)<300:
        raise SystemExit("STOP: F2R behavior core smoke produced too few records")
    if len(ids)!=len(set(ids)):
        raise SystemExit("STOP: F2R behavior core smoke duplicate ids")
    if not all(r["source_family"]=="repair_behavior_core" for r in rows):
        raise SystemExit("STOP: F2R behavior core source-family drift")
    if not all(r["split"] in ("train","validation") for r in rows):
        raise SystemExit("STOP: F2R behavior core split drift")
    print(json.dumps({
      "status":"PASS",
      "schema":"model0001_f2r_repair_runtime_smoke_v1",
      "records":len(rows),
      "train":sum(r["split"]=="train" for r in rows),
      "validation":sum(r["split"]=="validation" for r in rows)
    },sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/storage/emulated/0/Download/friend_core_corpus_bootstrap_v1")
    ap.add_argument("--collapse-audit",default="/storage/emulated/0/Download/model0001-f2-collapse-audit.json")
    ap.add_argument("--suite",default=None)
    ap.add_argument("--runtime-smoke",action="store_true")
    args=ap.parse_args()
    if args.runtime_smoke:
        runtime_smoke()
        return

    project=Path(args.project).resolve()
    source=project/"data"/"f2_sft"/"friend_f2_sft_source.jsonl"
    source_audit=source.with_suffix(source.suffix+".audit.json")
    tokenizer_path=project/"artifacts"/"tokenizer_v1"/"tokenizer.json"
    collapse_path=Path(args.collapse_audit).resolve()
    suite_path=(Path(args.suite).resolve() if args.suite else
                Path(__file__).resolve().parents[1]/"eval"/"model0001_post_f2_behavior_suite_v1.json")
    for p in (source,source_audit,tokenizer_path,collapse_path,suite_path):
        if not p.is_file(): raise SystemExit(f"STOP: missing repair input {p}")
    if sha256(tokenizer_path)!=TOKENIZER_SHA:
        raise SystemExit("STOP: tokenizer SHA drift")

    audit=json.loads(collapse_path.read_text(encoding="utf-8"))
    if audit.get("schema")!="model0001_f2_behavior_collapse_audit_v1" or audit.get("status")!="PASS":
        raise SystemExit("STOP: empirical collapse audit not PASS")
    if audit.get("behavior",{}).get("numerical_preflight",{}).get("pass") is not True:
        raise SystemExit("STOP: behavior runtime was not numerically verified")
    if audit.get("root_cause_flags",{}).get("behavior_generation_collapse_present") is not True:
        raise SystemExit("STOP: behavior report does not justify repair stage")
    if audit.get("test_split_used") is not False:
        raise SystemExit("STOP: collapse audit touched test")

    source_validation=json.loads(source_audit.read_text(encoding="utf-8"))
    if source_validation.get("status")!="PASS" or source_validation.get("sha256")!=sha256(source):
        raise SystemExit("STOP: original F2 source validation mismatch")

    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(tokenizer_path))
    excluded_prompts=eval_prompt_norms(suite_path)

    original=[]
    with source.open("r",encoding="utf-8") as f:
        for raw in f:
            if raw.strip(): original.append(json.loads(raw))

    # Old F2 validation stays holdout forever. Only old TRAIN is eligible.
    old_train=[r for r in original if r.get("split")=="train"]
    old_val=[r for r in original if r.get("split")=="validation"]
    if not old_train or not old_val:
        raise SystemExit("STOP: original F2 train/validation missing")

    exact_seen=set()
    prefix_count=collections.Counter()
    human=[]
    dropped=collections.Counter()

    # Dynamic prefix cap: max 0.5% of eligible human records, min 4, max 24.
    eligible_human=[
        r for r in old_train
        if r.get("source_family") in ("human_natural_dialogue","human_task_dialogue")
    ]
    prefix_cap=max(4,min(24,int(math.ceil(len(eligible_human)*0.005))))

    for row in sorted(eligible_human,key=lambda r:rank(r["id"])):
        if not no_eval_prompt_overlap(row,excluded_prompts):
            dropped["eval_prompt_overlap"]+=1; continue
        sig=assistant_signature(row)
        if not sig:
            dropped["no_assistant"]+=1; continue
        if sig in exact_seen:
            dropped["exact_assistant_duplicate"]+=1; continue
        if generic_short(row):
            dropped["generic_short_ack"]+=1; continue
        p2=first_prefix2(row)
        if p2 and prefix_count[p2]>=prefix_cap:
            dropped["assistant_prefix_cap"]+=1; continue
        exact_seen.add(sig)
        prefix_count[p2]+=1
        clone=json.loads(json.dumps(row,ensure_ascii=False))
        clone["id"]="f2r:human:"+row["id"]
        clone["split"]=split_for(clone["id"])
        clone["source_family"]="repair_human_"+(
            "natural" if row.get("source_family")=="human_natural_dialogue" else "task"
        )
        clone["source"]=row["source"]+"; F2R filtered from original F2 train only"
        clone["style"]=list(row.get("style",[]))+["f2r-filtered"]
        human.append(clone)

    core=build_behavior_core()
    core=[r for r in core if no_eval_prompt_overlap(r,excluded_prompts)]
    if not human or not core:
        raise SystemExit("STOP: repair human/core pool empty")

    # Token-aware balance. Keep all behavior core, then select human records in
    # deterministic order until core is ~27% of scored assistant tokens.
    core_train=[r for r in core if r["split"]=="train"]
    human_train=[r for r in human if r["split"]=="train"]
    core_train_tokens=sum(scored_tokens(tok,r) for r in core_train)
    target_core=0.27
    desired_human_tokens=int(core_train_tokens*(1-target_core)/target_core)
    selected_human=[]
    ht=0
    for row in sorted(human_train,key=lambda r:rank("select:"+r["id"])):
        t=scored_tokens(tok,row)
        selected_human.append(row); ht+=t
        if ht>=desired_human_tokens: break
    if not selected_human:
        raise SystemExit("STOP: no human train records selected")

    # Validation is separately deterministic and never copied to train.
    selected_ids={r["id"] for r in selected_human}
    human_val=[r for r in human if r["split"]=="validation"]
    core_val=[r for r in core if r["split"]=="validation"]

    train=selected_human+core_train
    validation=human_val+core_val
    all_rows=train+validation

    # Hard exact message dedupe across repair dataset.
    seen_messages=set()
    dedup=[]
    for row in sorted(all_rows,key=lambda r:rank("final:"+r["id"])):
        sig=record_signature(row)
        if sig in seen_messages:
            dropped["repair_exact_message_duplicate"]+=1
            continue
        seen_messages.add(sig); dedup.append(row)
    train=[r for r in dedup if r["split"]=="train"]
    validation=[r for r in dedup if r["split"]=="validation"]
    if not train or not validation:
        raise SystemExit("STOP: repair split empty")

    train_core_tokens=sum(scored_tokens(tok,r) for r in train if r["source_family"]=="repair_behavior_core")
    train_total_tokens=sum(scored_tokens(tok,r) for r in train)
    core_fraction=train_core_tokens/max(1,train_total_tokens)
    if not (TARGET_CORE_SCORED_FRACTION[0] <= core_fraction <= TARGET_CORE_SCORED_FRACTION[1]):
        raise SystemExit(
            f"STOP: repair behavior-core scored fraction {core_fraction:.6f} "
            f"outside {TARGET_CORE_SCORED_FRACTION}"
        )

    human_fraction=1.0-core_fraction
    if human_fraction<0.68:
        raise SystemExit("STOP: human dialogue no longer clear majority")

    # Evaluation prompts must not appear exactly in repair train.
    for row in train:
        if not no_eval_prompt_overlap(row,excluded_prompts):
            raise SystemExit("STOP: behavior eval prompt leaked into repair train")

    outdir=project/"data"/"f2r_repair"
    outdir.mkdir(parents=True,exist_ok=True)
    out=outdir/"friend_f2r_repair_source.jsonl"
    with out.open("w",encoding="utf-8") as f:
        for row in sorted(dedup,key=lambda r:(r["split"],rank(r["id"]))):
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")

    families=collections.Counter(r["source_family"] for r in dedup)
    split_counts=collections.Counter(r["split"] for r in dedup)
    report={
      "status":"PASS",
      "schema":"model0001_f2r_repair_source_build_v1",
      "output":str(out),
      "sha256":sha256(out),
      "source_checkpoint_policy":"RETRAIN_FROM_FOUNDATION_V3_NOT_F2_FINAL",
      "original_f2_source_sha256":sha256(source),
      "collapse_audit_sha256":sha256(collapse_path),
      "behavior_suite_sha256":sha256(suite_path),
      "prefix_cap_per_first_two_words":prefix_cap,
      "dropped":dict(dropped),
      "records":len(dedup),
      "split_counts":dict(split_counts),
      "source_family_counts":dict(families),
      "train_scored_assistant_tokens":train_total_tokens,
      "train_behavior_core_scored_tokens":train_core_tokens,
      "train_behavior_core_scored_fraction":core_fraction,
      "train_human_scored_fraction":human_fraction,
      "hard_guards":{
        "old_f2_validation_used_for_train":False,
        "behavior_eval_prompts_used_for_train":False,
        "exact_assistant_duplicates_filtered":True,
        "high_frequency_prefixes_capped":True,
        "generic_short_ack_targets_filtered":True,
        "human_dialogue_majority":True,
        "openai_teacher_outputs_used":False,
        "project_test_split_used":False,
        "training_started":False
      }
    }
    rp=outdir/"F2R_SOURCE_REPORT.json"
    rp.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
