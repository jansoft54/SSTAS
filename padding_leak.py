from model.bert import ActionBERT, ActionBERTConfig, checkLeakage
import torch
import torch.nn.functional as F


def test_padding_and_batch_mixing(model, input_dim=2048):
    model.eval()
    # Dropout ausschalten ist wichtig für Determinismus
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0

    device = next(model.parameters()).device
    print(f"--- Starte 3-Wege-Diagnose auf {device} ---")

    # -------------------------------------------------
    # 1. DATEN VORBEREITEN
    # -------------------------------------------------

    # Video A: Lang (11.000 Frames)
    L_a = 15000
    feat_a = torch.randn(1, L_a, input_dim).to(device)
    # Padding-Maske für Video A: Alles True (15.000 mal echte Daten)
    mask_a = torch.ones(1, L_a, dtype=torch.bool, device=device)
    # Patch-Maske für Video A (für das 80/10/10 BERT Masking)

    # -------------------------------------------------
    # 2. VIDEO B VORBEREITEN (L_b = 12000)
    # -------------------------------------------------
    L_b = 12000
    # Die "reinen" Daten von Video B
    feat_b_raw = torch.randn(1, L_b, input_dim).to(device)
    mask_b_raw = torch.ones(1, L_b, dtype=torch.bool,
                            device=device)  # Alles True

    # Padding-Länge berechnen
    diff = L_a - L_b  # 3000 Frames Padding

    # Video B Padded: Daten zuerst, dann 3000 Nullen (xxxx0000)
    feat_b_padded = torch.cat([
        feat_b_raw,
        torch.zeros(1, diff, input_dim, device=device)
    ], dim=1)

    # Padding-Maske B: 12000x True (Daten), dann 3000x False (Padding)
    mask_b_padded = torch.cat([
        mask_b_raw,
        torch.zeros(1, diff, dtype=torch.bool, device=device)
    ], dim=1)

    # Patch-Maske B: Wir maskieren im Padding-Bereich nichts (False)

    # -------------------------------------------------
    # 3. BATCH ERSTELLEN
    # -------------------------------------------------
    # Batch besteht aus [Video A, Video B_padded]

    print(mask_b_padded[0])
    batch_feat = torch.cat([feat_a, feat_b_padded], dim=0)
    batch_mask = torch.cat([mask_a, mask_b_padded], dim=0)     # Padding Maske

    # -------------------------------------------------
    # 2. MODEL RUNS
    # -------------------------------------------------

    with torch.no_grad():
        # RUN 1: Reference (Video B pur, ohne Padding)
        # So sollte das Ergebnis idealerweise aussehen.
        out_b_pure = model(feat_b_raw, None, mask_b_raw, _run_name="b_pure")
        logits_b_pure = out_b_pure["refine_logits"]

        # RUN 2: Single Padded (Video B alleine, aber mit Nullen aufgefüllt)
        # Testet: Verändern die Nullen am Ende das Ergebnis am Anfang?
        out_b_padded_single = model(
            feat_b_padded, None, mask_b_padded, _run_name="b_single_padded")
        logits_b_padded_single = out_b_padded_single["refine_logits"]

        # RUN 3: Batched (Video A und Video B zusammen)
        # Testet: Verändert die Anwesenheit von Video A das Ergebnis von Video B?
        out_batch = model(batch_feat, None, batch_mask, _run_name="batched")
        logits_batch = out_batch["refine_logits"]

    # -------------------------------------------------
    # 3. VERGLEICHE & ANALYSE
    # -------------------------------------------------

    # Analyse 1: Padding Leakage
    # Vergleich: B(Pure) vs. B(Single Padded)
    # Wir schneiden vom Padded-Output nur den validen Teil (0 bis L_b) ab.
    diff_padding = (
        logits_b_pure - logits_b_padded_single[:, :L_b]).abs().max().item()

    # Analyse 2: Batch Contamination
    # Vergleich: B(Single Padded) vs. B(Batch)
    # Wir vergleichen Video B im Single-Mode gegen Video B im Batch-Mode.
    # Da beide gepaddet sind, isolieren wir hier rein den Effekt von Video A.
    diff_batch = (logits_b_padded_single -
                  logits_batch[1:2]).abs().max().item()

    print("-" * 60)
    print(
        f"Diff 1: Padding Leakage (Pure vs. Padded Single):  {diff_padding:.8f}")
    print(
        f"Diff 2: Batch Leakage   (Padded Single vs. Batch): {diff_batch:.8f}")
    print("-" * 60)

    # --- DIAGNOSE ---
    if diff_padding > 1e-5:
        print("❌ FEHLERQUELLE: PADDING")
        print("   Das Modell ist nicht robust gegen Nullen am Ende.")
        print("   -> Problem liegt in GlobalAttention, LayerNorm oder Residuals.")
        print("   -> Lösung: Output Cleaning (x = x * mask) nach jedem Layer erzwingen.")

    if diff_batch > 1e-5:
        print("❌ FEHLERQUELLE: BATCH MIXING")
        print("   Das Modell vermischt Informationen zwischen Video A und B.")
        print("   -> Problem liegt meist an falschem .view() oder .reshape() in GlobalAttention.")

    else:
        print("✅ ALLES SAUBER: Das Modell ist mathematisch korrekt.")
        print("   Wenn BS=4 trotzdem schlechter ist, liegt es am Training (Loss/Statistik), nicht am Modell.")


# --- RUN ---
bert_conf = ActionBERTConfig(
    total_classes=14 + 30,
    input_dim=2048,
    d_model=256,
    num_heads=8,
    num_layers=4,
    dropout=0.0  # Dropout im Config auch aus
)
model = ActionBERT(config=bert_conf)

# Lade Gewichte falls vorhanden
try:
    path = "./output/actionbert_second_try.pth"
    state_dict = torch.load(path, map_location=torch.device('cuda'))
   # model.load_state_dict(state_dict, strict=False)
    print("Weights loaded.")
except:
    print("No weights loaded, using random init.")

model = model.to('cuda')
model.eval()
test_padding_and_batch_mixing(model)
checkLeakage()
