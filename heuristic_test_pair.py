import subprocess
import sys
import os
import csv
import random
from collections import defaultdict
from itertools import combinations

# --- AUTO-INSTALLER ---
def maintain_dependencies():
    for lib in ['numpy', 'scipy', 'gensim']:
        try:
            __import__(lib)
        except ImportError:
            print(f"📦 Installing '{lib}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

maintain_dependencies()

import numpy as np
from gensim.models import FastText

# =============================================================================
# HEURISTIC STRATEGIES
# =============================================================================
#
# 1. MORPHOLOGICAL  — isiZulu noun class prefixes shared between words
#    Score band: 0.5–0.9  (same prefix = likely same category/class)
#
# 2. SEMANTIC CATEGORY — words grouped by topic (food, family, animals, etc.)
#    Score band: 0.6–1.0  (same category = semantically related)
#
# 3. CO-OCCURRENCE — words appearing in same sentence in corpus
#    Score band: 0.4–0.8  (contextually related but not always synonyms)
#
# 4. RANDOM (negative controls) — arbitrary pairs from different categories
#    Score band: 0.0–0.3  (expected to be unrelated)
#
# Heuristic scores are ESTIMATES used as soft labels.
# They should be validated by human annotators before use as gold standard.
# =============================================================================

# --- CORPUS ---
SAMPLE_CORPUS = """umfazi nendoda bahamba esikoleni
ingane idla ukudla kwayo
inja ikati zidlala eyadini
isikole isikhungo semfundo
ikhaya indlu yomndeni
umfula ulwandle amanzi
uthisha umfundi bafunda
isitsha indishi kudla
ibhola umdlalo imidlalo
umuntu ubuntu abantu
itheku idolobha amadolobha
incwadi iphepha ukubhala
ikhompiyutha ikhibhodi theknoloji
indiza imoto isitimela ukuhamba
ucingo ukuxhumana uxhumano
umabonakude umsakazo ezindaba
udokotela umhlengikazi ukwelapha
inkampani amasheya ukuhweba
ibhange imali ukonga
ukhuni ihlathi amahlathi
inkosi indlovukazi umbuso
inyoni iqhude izilwane
umfana mfowethu umndeni
imali idola ingcebo impahla
ihlosi isilwane izilwane
isinkwa ibhotela ukudla
ikhukhamba izambane imifino
umtapo wezincwadi incwadi
imeya inkosi amadolobha
isikweletu imali ukuboleka
umasipala uhulumeni isikhungo
inkohlakalo icala ububi
inyuvesi isikole ukufunda
isivivinyo umatric ukuhlola
ingoma umculo icwecwe
umrepha umculi umculo
ikhwaya umbhalo amaculo
amaphoyisa abasolwa icala
isibhamu inhlamvu udubula
isiteshi inkantolo amaphoyisa
ilokishi idolobha indawo
ihhotela isivakashi ukulala
itekisi imoto ukuhamba
isifo impilo ukugula
ubumnandi ukujabula injabulo
isizwe isibongo isiko
umhlaba umnotho impilo
inkatha isizwe uzalo
umngane umuntu ubuntu
amazambane imifino ukudla
ubisi inyama ukudla
izinkomo izimvu izilwane
amahhashi izinja izilwane
izintaba imifula indawo
ulwandle amanzi ukujula
ukusebenza umsebenzi isikhungo
ingqalasizinda ukuthuthukiswa ukwakha
umuntu abantu ubuntu
ubukhosi ubuhle ubungane
ukudla ukuphuza impilo
inkosi yamazwe uhulumeni
impumelelo imfundo ulwazi
umphakathi imiphakathi abantu
amasiko amazulu imvunulo
izithakazelo isibongo isiko
ukucula ingoma umculo
imithi imvelo ihlathi
ukupheka ukudla ukwenza
"""

# --- ISIZULU NOUN CLASS PREFIXES ---
# isiZulu organises nouns into classes by prefix
NOUN_CLASS_PREFIXES = {
    'um': 'class1_singular',    # umuntu, umfazi
    'aba': 'class2_plural',     # abantu, abafazi
    'imi': 'class3_plural',     # imifula, imidlalo
    'i':  'class5_singular',    # ibhola, ingane (also catches 'in', 'im')
    'in': 'class9_singular',    # inkosi, indlu
    'im': 'class9_singular',    # imali, impilo
    'iz': 'class10_plural',     # izilwane, izintaba
    'u':  'class11',            # ubuntu, ulwandle
    'is': 'class7_singular',    # isikole, isifo
    'iz': 'class8_plural',      # izinto
    'uk': 'class15_infinitive', # ukudla, ukuhamba (verb infinitives)
    'ama': 'class6_plural',     # amadolobha, amaculo
}

# --- SEMANTIC CATEGORIES ---
SEMANTIC_CATEGORIES = {
    'food':        ['ukudla', 'isinkwa', 'ibhotela', 'ubisi', 'inyama', 'amazambane',
                    'imifino', 'ikhukhamba', 'isitsha', 'indishi', 'ukuphuza'],
    'family':      ['umfazi', 'indoda', 'ingane', 'umfana', 'mfowethu', 'umndeni',
                    'umuntu', 'abantu', 'uzalo'],
    'animals':     ['inja', 'ikati', 'inyoni', 'iqhude', 'ihlosi', 'izilwane',
                    'izinkomo', 'izimvu', 'amahhashi', 'izinja'],
    'education':   ['isikole', 'inyuvesi', 'umfundi', 'uthisha', 'incwadi',
                    'iphepha', 'ukubhala', 'ukufunda', 'isivivinyo', 'umatric',
                    'umtapo', 'ulwazi', 'impumelelo', 'imfundo'],
    'government':  ['inkosi', 'imeya', 'uhulumeni', 'umasipala', 'umbuso',
                    'indlovukazi', 'inkohlakalo', 'icala', 'inkatha', 'isizwe'],
    'transport':   ['imoto', 'indiza', 'isitimela', 'itekisi', 'ukuhamba', 'uhambo'],
    'money':       ['imali', 'ibhange', 'isikweletu', 'ukuboleka', 'ukonga',
                    'inkampani', 'amasheya', 'idola', 'ingcebo'],
    'health':      ['udokotela', 'umhlengikazi', 'ukwelapha', 'isifo', 'impilo',
                    'ukugula'],
    'music':       ['ingoma', 'umculo', 'icwecwe', 'umrepha', 'umculi',
                    'ikhwaya', 'umbhalo', 'amaculo'],
    'police_law':  ['amaphoyisa', 'abasolwa', 'icala', 'isibhamu', 'inhlamvu',
                    'isiteshi', 'inkantolo'],
    'nature':      ['ihlathi', 'ukhuni', 'umfula', 'ulwandle', 'amanzi',
                    'izintaba', 'imifula', 'imvelo', 'imithi'],
    'technology':  ['ikhompiyutha', 'ikhibhodi', 'ucingo', 'ukuxhumana',
                    'umabonakude', 'umsakazo', 'theknoloji'],
    'places':      ['idolobha', 'ilokishi', 'itheku', 'ikhaya', 'indlu',
                    'ihhotela', 'isiteshi', 'isikhungo'],
    'culture':     ['amasiko', 'amazulu', 'imvunulo', 'izithakazelo', 'isibongo',
                    'isiko', 'ubuntu', 'ubukhosi'],
    'sport':       ['ibhola', 'umdlalo', 'imidlalo', 'ukudlala'],
}

# =============================================================================
# HELPER: COSINE SIMILARITY
# =============================================================================
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return float(dot / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

# =============================================================================
# HEURISTIC 1: MORPHOLOGICAL PAIRS
# Group words by noun class prefix → pairs within same class
# =============================================================================
def get_morphological_pairs(vocab, target=1200):
    prefix_groups = defaultdict(list)
    for word in vocab:
        for prefix in sorted(NOUN_CLASS_PREFIXES.keys(), key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                prefix_groups[prefix].append(word)
                break

    pairs = []
    for prefix, words in prefix_groups.items():
        if len(words) < 2:
            continue
        noun_class = NOUN_CLASS_PREFIXES[prefix]
        for w1, w2 in combinations(words, 2):
            pairs.append((w1, w2, 'morphological', noun_class, prefix))

    random.shuffle(pairs)
    return pairs[:target]

# =============================================================================
# HEURISTIC 2: SEMANTIC CATEGORY PAIRS
# Pairs within same semantic group = highly related
# Pairs across different groups = less related
# =============================================================================
def get_semantic_pairs(vocab_set, target=1200):
    pairs = []
    categories = list(SEMANTIC_CATEGORIES.keys())

    # Within-category pairs (similar)
    for cat, words in SEMANTIC_CATEGORIES.items():
        valid = [w for w in words if w in vocab_set]
        for w1, w2 in combinations(valid, 2):
            pairs.append((w1, w2, 'semantic_within', cat, cat))

    # Across-category pairs (dissimilar — acts as soft negative)
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            c1, c2 = categories[i], categories[j]
            w1_pool = [w for w in SEMANTIC_CATEGORIES[c1] if w in vocab_set]
            w2_pool = [w for w in SEMANTIC_CATEGORIES[c2] if w in vocab_set]
            if w1_pool and w2_pool:
                w1 = random.choice(w1_pool)
                w2 = random.choice(w2_pool)
                pairs.append((w1, w2, 'semantic_across', f'{c1}_vs_{c2}', ''))

    random.shuffle(pairs)
    return pairs[:target]

# =============================================================================
# HEURISTIC 3: CO-OCCURRENCE PAIRS
# Words appearing in the same sentence are contextually related
# =============================================================================
def get_cooccurrence_pairs(sentences, vocab_set, window=5, target=1200):
    cooccur = defaultdict(int)
    for sent in sentences:
        valid = [w for w in sent if w in vocab_set]
        for i, w1 in enumerate(valid):
            for w2 in valid[max(0, i-window): i+window+1]:
                if w1 != w2:
                    pair = tuple(sorted([w1, w2]))
                    cooccur[pair] += 1

    # Sort by frequency — higher co-occurrence = more related
    sorted_pairs = sorted(cooccur.items(), key=lambda x: x[1], reverse=True)
    pairs = []
    for (w1, w2), freq in sorted_pairs:
        pairs.append((w1, w2, 'cooccurrence', f'freq={freq}', ''))

    random.shuffle(pairs)
    return pairs[:target]

# =============================================================================
# HEURISTIC 4: RANDOM NEGATIVE CONTROL PAIRS
# Words from different semantic categories — expected low similarity
# =============================================================================
def get_random_pairs(vocab, semantic_words, target=1400):
    # Prefer pairing words from very different semantic groups
    cat_words = list(SEMANTIC_CATEGORIES.keys())
    pairs = []
    attempts = 0

    while len(pairs) < target and attempts < target * 10:
        attempts += 1
        w1, w2 = random.sample(vocab, 2)
        if w1 == w2:
            continue
        # Check they don't share same semantic category
        shared_cat = False
        for words in SEMANTIC_CATEGORIES.values():
            if w1 in words and w2 in words:
                shared_cat = True
                break
        label = 'random_unrelated' if not shared_cat else 'random_mixed'
        pairs.append((w1, w2, label, '', ''))

    return pairs[:target]

# =============================================================================
# ASSIGN HEURISTIC SCORE BAND
# =============================================================================
def heuristic_score_band(strategy, model, w1, w2):
    """
    Assign a heuristic similarity score based on:
    1. The strategy that generated the pair (prior)
    2. The actual cosine similarity from the model (evidence)
    
    Final score blends both: score = 0.4*prior + 0.6*cosine
    This ensures scores are grounded in the model but guided by linguistic knowledge.
    """
    # Strategy priors (expected similarity range midpoint)
    priors = {
        'morphological':    0.65,
        'semantic_within':  0.75,
        'semantic_across':  0.30,
        'cooccurrence':     0.55,
        'random_unrelated': 0.15,
        'random_mixed':     0.25,
    }
    prior = priors.get(strategy, 0.40)

    try:
        vec1 = model.wv[w1]
        vec2 = model.wv[w2]
        cosine = cosine_similarity(vec1, vec2)
        # Blend: weight cosine more heavily than prior
        blended = round(0.3 * prior + 0.7 * cosine, 6)
        return cosine, blended
    except Exception as e:
        return None, None

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    CORPUS_FILE = 'isizulu_corpus.txt'
    OUTPUT_CSV  = 'isizulu_heuristic_5000_pairs.csv'
    NUM_PAIRS   = 5000
    random.seed(42)

    # Write corpus
    if not os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_CORPUS)
        print(f"✅ Corpus written to {CORPUS_FILE}")

    # Load corpus
    print(f"\n📂 Loading corpus...")
    sentences = []
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.lower().strip().split()
            if tokens:
                sentences.append(tokens)
    print(f"✅ {len(sentences)} sentences loaded")

    # Train FastText
    print(f"\n🚀 Training FastText model...")
    model = FastText(
        sentences=sentences,
        vector_size=200,
        window=7,
        min_count=1,
        epochs=100,
        sg=1,
        workers=4,
        min_n=3,
        max_n=6,
        word_ngrams=1,
        bucket=2000000
    )
    vocab     = list(model.wv.key_to_index.keys())
    vocab_set = set(vocab)
    print(f"✅ FastText trained. Vocabulary: {len(vocab)} words")

    # --- GENERATE PAIRS BY STRATEGY ---
    print(f"\n🔀 Generating pairs using 4 heuristic strategies...")

    morph_pairs   = get_morphological_pairs(vocab, target=1200)
    sem_pairs     = get_semantic_pairs(vocab_set, target=1200)
    coocc_pairs   = get_cooccurrence_pairs(sentences, vocab_set, target=1200)
    rand_pairs    = get_random_pairs(vocab, vocab_set, target=1400)

    print(f"   Morphological pairs:   {len(morph_pairs)}")
    print(f"   Semantic pairs:        {len(sem_pairs)}")
    print(f"   Co-occurrence pairs:   {len(coocc_pairs)}")
    print(f"   Random (control) pairs:{len(rand_pairs)}")

    all_pairs = morph_pairs + sem_pairs + coocc_pairs + rand_pairs
    random.shuffle(all_pairs)

    # Deduplicate
    seen = set()
    unique_pairs = []
    for entry in all_pairs:
        key = tuple(sorted([entry[0], entry[1]]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(entry)

    # Trim or pad to NUM_PAIRS
    unique_pairs = unique_pairs[:NUM_PAIRS]
    print(f"\n✅ {len(unique_pairs)} unique pairs after deduplication")

    # --- SCORE ALL PAIRS ---
    print(f"\n📐 Scoring all pairs...")
    results = []
    cosines, blended_scores = [], []
    strategy_counts = defaultdict(int)

    for entry in unique_pairs:
        w1, w2, strategy, detail, prefix = entry
        cosine, blended = heuristic_score_band(strategy, model, w1, w2)

        in_vocab = (
            "Both"    if w1 in vocab_set and w2 in vocab_set else
            "Partial" if w1 in vocab_set or  w2 in vocab_set else
            "N-gram"
        )

        results.append({
            'word1':             w1,
            'word2':             w2,
            'strategy':          strategy,
            'detail':            detail,
            'cosine_similarity': round(cosine, 6)   if cosine  is not None else 'ERROR',
            'heuristic_score':   round(blended, 6)  if blended is not None else 'ERROR',
            'in_vocab':          in_vocab,
        })
        strategy_counts[strategy] += 1
        if cosine  is not None: cosines.append(cosine)
        if blended is not None: blended_scores.append(blended)

    # --- PRINT SUMMARY ---
    print("\n" + "="*80)
    print("HEURISTIC PAIR GENERATION — SUMMARY")
    print("="*80)

    print(f"\n📊 STRATEGY BREAKDOWN:")
    for strat, count in sorted(strategy_counts.items()):
        print(f"   {strat:<25} {count:>5} pairs")

    arr_b = np.array(blended_scores)
    arr_c = np.array(cosines)
    print(f"\n📈 HEURISTIC SCORE STATISTICS (blended):")
    print(f"   Min:    {arr_b.min():.4f}  |  Max:    {arr_b.max():.4f}")
    print(f"   Mean:   {arr_b.mean():.4f} |  Median: {np.median(arr_b):.4f}")
    print(f"   Std:    {arr_b.std():.4f}")

    print(f"\n📈 COSINE SCORE DISTRIBUTION:")
    bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ['0.0–0.2', '0.2–0.4', '0.4–0.6', '0.6–0.8', '0.8–1.0']
    for i in range(len(labels)):
        count = int(np.sum((arr_c >= bins[i]) & (arr_c < bins[i+1])))
        bar = '█' * int(count / len(cosines) * 40)
        print(f"   {labels[i]}: {bar} {count} ({count/len(cosines)*100:.1f}%)")

    print(f"\n⚠️  SCORING NOTE:")
    print(f"   Heuristic scores = 0.4 × strategy_prior + 0.6 × cosine_similarity")
    print(f"   Strategy priors:")
    print(f"      semantic_within   → 0.75  (same topic = similar)")
    print(f"      morphological     → 0.65  (same noun class prefix)")
    print(f"      cooccurrence      → 0.55  (appear in same sentence)")
    print(f"      semantic_across   → 0.30  (different topics)")
    print(f"      random_mixed      → 0.25")
    print(f"      random_unrelated  → 0.15  (negative controls)")
    print(f"\n   ⚡ These are SOFT LABELS — human validation recommended")
    print(f"      before using as gold standard in a paper.")
    print("="*80)

    # --- SAVE CSV ---
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'word1','word2','strategy','detail',
            'cosine_similarity','heuristic_score','in_vocab'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Saved {len(results)} pairs → '{OUTPUT_CSV}'")
    print(f"🎉 Done!")
    sys.stdout.flush()