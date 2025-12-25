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
    mask_a = torch.ones(1, L_a).bool().to(device)
    
    # Video B: Kurz (9.000 Frames)
    L_b = 12000
    feat_b_raw = torch.randn(1, L_b, input_dim).to(device)
    mask_b_raw = (torch.randperm(L_b, device=device) < (L_b // 2)).view(1, -1)
    # Video B Padded: Auf Länge von A aufgefüllt (mit 0.0)
    # Das simuliert, wie Video B im Batch aussehen MUSS.
    feat_b_padded = torch.cat([feat_b_raw, torch.zeros(1, L_a - L_b, input_dim).to(device)], dim=1)
    mask_b_padded = torch.cat([mask_b_raw, torch.zeros(1, L_a - L_b).bool().to(device)], dim=1)
    
    # Batch: A und B(padded) zusammen
    batch_feat = torch.cat([feat_a, feat_b_padded], dim=0)
    batch_mask = torch.cat([mask_a, mask_b_padded], dim=0)

    # -------------------------------------------------
    # 2. MODEL RUNS
    # -------------------------------------------------
    
    with torch.no_grad():
        # RUN 1: Reference (Video B pur, ohne Padding)
        # So sollte das Ergebnis idealerweise aussehen.
        _, logits_b_pure, _ = model(feat_b_raw, None, mask_b_raw, _run_name="b_pure")

        # RUN 2: Single Padded (Video B alleine, aber mit Nullen aufgefüllt)
        # Testet: Verändern die Nullen am Ende das Ergebnis am Anfang?
        _, logits_b_padded_single, _ = model(feat_b_padded, None, mask_b_padded, _run_name="b_single_padded")

        # RUN 3: Batched (Video A und Video B zusammen)
        # Testet: Verändert die Anwesenheit von Video A das Ergebnis von Video B?
        _, logits_batch, _ = model(batch_feat, None, batch_mask, _run_name="batched")

    # -------------------------------------------------
    # 3. VERGLEICHE & ANALYSE
    # -------------------------------------------------
    
    # Analyse 1: Padding Leakage
    # Vergleich: B(Pure) vs. B(Single Padded)
    # Wir schneiden vom Padded-Output nur den validen Teil (0 bis L_b) ab.
    diff_padding = (logits_b_pure - logits_b_padded_single[:, :L_b]).abs().max().item()
    
    # Analyse 2: Batch Contamination
    # Vergleich: B(Single Padded) vs. B(Batch)
    # Wir vergleichen Video B im Single-Mode gegen Video B im Batch-Mode.
    # Da beide gepaddet sind, isolieren wir hier rein den Effekt von Video A.
    diff_batch = (logits_b_padded_single - logits_batch[1:2]).abs().max().item()
    
    print("-" * 60)
    print(f"Diff 1: Padding Leakage (Pure vs. Padded Single):  {diff_padding:.8f}")
    print(f"Diff 2: Batch Leakage   (Padded Single vs. Batch): {diff_batch:.8f}")
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
    dropout=0.0 # Dropout im Config auch aus
)
model = ActionBERT(config=bert_conf)

# Lade Gewichte falls vorhanden
try:
    path = "./output/actionbert_second_try.pth"
    state_dict = torch.load(path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict, strict=False)
    print("Weights loaded.")
except:
    print("No weights loaded, using random init.")

model = model.to('cuda')
model.eval()
test_padding_and_batch_mixing(model)
checkLeakage()